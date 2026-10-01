"""
Ablation 3, row "DiT micro-only": collapse the macro-format LUT to a single
format (FP8 E4M3) for every bucket and every layer, leaving dynamic
micro-scaling intact. Matches a Q-DiT-style baseline that uses dynamic
quantisation but no temporal macro routing.

IMPLEMENTATION NOTE (v2 — the v1 patch was a silent no-op): v1 replaced
`route_format_by_kurtosis_snr`, but ablation runs load pre-calibrated weights
via --load-weights, and `load_weight_quantization` restores the baked
`format_lut` from the JSON file — the routing function never executes at
generation time, so v1 changed nothing (byte-identical images vs
macro_plus_micro; caught by md5 diff). The baked LUT is ~99% E5M2, so forcing
E4M3 IS a real perturbation. v2 therefore wraps `load_weight_quantization` and
flattens every layer's `format_lut` AFTER the restore, which is the state the
generation loop actually reads.
"""

from _shim import parse_target, import_target, run_cli


def main():
    cfg = parse_target()
    assert cfg["module"] == "chameleon_dit_generation", \
        "dit_micro_only is DiT-only; pass --target dit"
    quant_mod, _ = import_target(cfg)
    # NB: for DiT, "selector" and "quant" are both chameleon_dit_quant.

    fp8_e4m3 = quant_mod.FORMAT_FP8_E4M3
    Q = quant_mod.ChameleonDiTQuantizer
    orig_load = Q.load_weight_quantization

    def load_and_flatten(self, load_path):
        orig_load(self, load_path)
        n = 0
        for layer in self.quant_layers:
            layer.fake_quant.format_lut.fill_(fp8_e4m3)
            layer.fake_quant.enabled = True
            n += 1
        print(f"[ablation patch] flattened format_lut of {n} layers to "
              f"FP8_E4M3 (code={fp8_e4m3}) AFTER weight load; micro-scale only")
        assert n > 0, "no quant layers found after load — patch would be a no-op"

    Q.load_weight_quantization = load_and_flatten
    # Belt-and-braces: also patch routing, so a calibration-path run (no
    # --load-weights) is covered too.
    quant_mod.route_format_by_kurtosis_snr = lambda kurtosis, diffusion_snr: fp8_e4m3
    run_cli(cfg)


if __name__ == "__main__":
    main()
