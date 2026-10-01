"""
Ablation 3, row "DiT macro-only": keep macro-format routing, FREEZE micro-scale.

Still a meaningful ablation post per-tensor revert. The three DiT rows isolate the
two moving parts of the activation quantiser:
    macro_plus_micro : temporal format LUT  +  dynamic per-sample scale (canonical)
    micro_only       : single fixed format  +  dynamic per-sample scale
    macro_only       : temporal format LUT  +  STATIC (frozen) scale   ← this file
Together they attribute the FID contribution of (a) temporal format routing and
(b) dynamic micro-scaling separately.

Post-revert the canonical path is `pertensor_quant_activation(x, fmt_code)`, which
derives a fresh per-tensor scale/zp from each sample's min/max on every forward
("micro-scaling"). The old patch froze `_compute_dynamic_params`, a method removed
in the per-tensor revert. We now freeze the scale at the function level: cache the
(scale, zp) computed on the FIRST forward per format code, then reuse it for the
rest of the run — a static-scale PTQ baseline that keeps the temporal LUT.

`pertensor_quant_activation` is a module global that DiT's forward looks up by
name at call time (same module, no from-import trap), so replacing
`chameleon_dit_quant.pertensor_quant_activation` takes effect immediately.
"""

import torch

from _shim import parse_target, import_target, run_cli


def main():
    cfg = parse_target()
    assert cfg["module"] == "chameleon_dit_generation", \
        "dit_macro_only is DiT-only; pass --target dit"
    quant_mod, _ = import_target(cfg)

    q         = quant_mod
    INT8_ASYM = q.FORMAT_INT8_ASYM
    FP8_E4M3  = q.FORMAT_FP8_E4M3

    cache: dict = {}   # fmt_code -> (scale, zp|None), frozen on first observation

    def pertensor_static(x, fmt_code):
        orig_dtype = x.dtype
        x_f = x.float()
        if fmt_code not in cache:
            x_min, x_max = x_f.min().item(), x_f.max().item()
            if fmt_code == INT8_ASYM:
                scale = max((x_max - x_min) / 255.0, 1e-8)
                zp    = float(round(-x_min / scale)) - 128.0
                cache[fmt_code] = (scale, zp)
            else:
                max_abs = max(abs(x_min), abs(x_max), 1e-8)
                fmax    = q.FP8_E4M3_MAX if fmt_code == FP8_E4M3 else q.FP8_E5M2_MAX
                cache[fmt_code] = (max_abs / fmax, None)
        scale, zp = cache[fmt_code]
        if fmt_code == INT8_ASYM:
            result = (torch.clamp(torch.round(x_f / scale + zp), -128, 127) - zp) * scale
        elif fmt_code == FP8_E4M3:
            result = q._fp8_e4m3_fake_quant(x_f / scale) * scale
        else:  # FORMAT_FP8_E5M2
            result = q._fp8_e5m2_fake_quant(x_f / scale) * scale
        return result.to(orig_dtype)

    quant_mod.pertensor_quant_activation = pertensor_static
    print("[ablation patch] DiT micro-scale FROZEN to first-observation per "
          "format (macro-routing only)")
    run_cli(cfg)


if __name__ == "__main__":
    main()
