"""
Ablation 1, row "snr_only": kurtosis treated as a constant (κ = 4, mid-band)
so routing depends on diffusion-SNR alone.

Behaviour after patch:
    Phase 1 (SNR < 0.2)          → FP8 E5M2
    Phase 3 (SNR > 2.0)          → INT8 ASYM  (only fires if κ_const ≤ 3 ❌)
    Phase 2                       → FP8 E4M3 (transition)

We force κ_const = 4 so the κ-clean predicate is permanently false; this means
INT8_ASYM is never picked and the routing reduces to {E4M3, E5M2}. To preserve
the spirit of "SNR-only with all three formats reachable", we instead set
κ_const = 2.5 (always in the κ-clean band) so SNR alone gates INT8 vs E4M3.
"""

from _shim import parse_target, import_target, patch_symbol, run_cli

KAPPA_CONST = 2.5  # Always in the "clean κ" band so SNR is the only gate.


def main():
    cfg = parse_target()
    selector, quant = import_target(cfg)

    QF = selector.QuantFormat

    def route_snr_only(kurtosis, diffusion_snr, is_shortcut=False):
        # NB: incoming `kurtosis` is discarded.
        if is_shortcut:
            return QF.MXFP8_E4M3
        if diffusion_snr < selector.SNR_DIFFUSION_LOW:
            return QF.FLOAT8_E5M2
        if diffusion_snr > selector.SNR_DIFFUSION_HIGH:
            return QF.INT8_ASYM
        return QF.FLOAT8_E4M3

    patch_symbol("route_format_by_kurtosis_snr", route_snr_only, selector, quant)
    print(f"[ablation patch] SNR-only routing installed (κ ignored, "
          f"effective κ_const={KAPPA_CONST})")
    run_cli(cfg)


if __name__ == "__main__":
    main()
