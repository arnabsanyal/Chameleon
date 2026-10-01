"""
Ablation 3, row "LCM BAQ on" (v2): enable BOS-Aware Quantization in the
Chameleon-on-MixDQ fake-quant path.

WHY v1 WAS A NO-OP: MixDQ's BAQ lives exclusively in the accelerated integer
qlinear forward (`if not self.bos: ... else: <BOS bypass>`). Under
--chameleon-weights every layer takes the `use_fake_quant_w4` branch
(`_fake_quant_a8(x)` → `F.linear` → return), which exits BEFORE the BOS branch,
so bos on/off changed nothing — byte-identical images (caught by md5 diff).
The `bos` flag IS stamped on attn2.to_k/to_v modules by `from_float`; it is
simply never read on the fake path.

WHAT THIS PATCH DOES: wraps the dynamic pipeline module's QuantizedLinear
forward so that, when `use_fake_quant_w4` and `self.bos` are both set, token 0
(the BOS position of the 77-token text embedding entering cross-attn to_k/to_v)
is exempted from A8 fake-quant and computed in fp16 with the true folded
weights. This is BAQ's exact semantics — MixDQ substitutes a pre-computed fp16
output for token 0; we compute it live (strictly more faithful, since our
folded weights differ from the ones their `bos_pre_computed` was baked with),
and it lifts MixDQ's batch-1 constraint, so --batch-size 16 stays valid.

The pipeline class only exists after diffusers materialises the custom
pipeline, so we hook `DiffusionPipeline.from_pretrained`: on first load we
locate the dynamic module via `type(pipe).__module__` and patch its
QuantizedLinear class (instances are created later, in quantize_unet, so the
class-level patch covers them all).

Run WITHOUT --no-bos so quantize_unet stamps bos=True on the to_k/to_v layers.
"""

import sys

import torch
import torch.nn.functional as F

from _shim import parse_target, run_cli


def main():
    cfg = parse_target()
    assert cfg["module"] == "chameleon_lcm_generation", \
        "lcm_enable_baq is LCM-only; pass --target lcm"
    assert "--no-bos" not in cfg["forwarded"], \
        "lcm_enable_baq needs bos=True; drop --no-bos from the forwarded args"

    from diffusers import DiffusionPipeline
    orig_fp = DiffusionPipeline.from_pretrained.__func__

    def from_pretrained_baq(cls, *args, **kwargs):
        pipe = orig_fp(cls, *args, **kwargs)
        mod = sys.modules[type(pipe).__module__]
        QL = getattr(mod, "QuantizedLinear", None)
        if QL is None:
            print("[ablation patch] WARNING: QuantizedLinear not found in "
                  f"{type(pipe).__module__}; BAQ patch NOT applied")
            return pipe
        if getattr(QL, "_baq_fakequant_patched", False):
            return pipe
        orig_forward = QL.forward

        def forward_with_baq(self, x):
            if (self.use_fake_quant_w4 and getattr(self, "bos", False)
                    and x.dim() == 3 and x.shape[1] > 1):
                x0 = x[:, :1, :]                       # BOS token: fp16, unquantised
                xq = self._fake_quant_a8(x[:, 1:, :])  # rest: per-tensor A8 fake-quant
                x_fake = torch.cat([x0, xq], dim=1)
                return F.linear(x_fake, self.weight, self.bias)
            return orig_forward(self, x)

        QL.forward = forward_with_baq
        QL._baq_fakequant_patched = True
        print("[ablation patch] BAQ enabled in fake-quant path: BOS token "
              "exempt from A8 quant on bos-flagged layers (attn2 to_k/to_v)")
        return pipe

    DiffusionPipeline.from_pretrained = classmethod(from_pretrained_baq)
    run_cli(cfg)


if __name__ == "__main__":
    main()
