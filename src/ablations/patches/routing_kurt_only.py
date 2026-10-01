"""
Ablation 1, row "kurt_only": SNR thresholds set to ±∞ so routing depends on
kurtosis alone.

Behaviour after patch:
    Phase 1 (κ > 5)              → FP8 E5M2
    Phase 3 (κ ≤ 3)              → INT8 ASYM
    Phase 2 (3 < κ ≤ 5)          → FP8 E4M3 (transition)
    Shortcut layers              → MXFP8 E4M3 (unchanged)
"""

from _shim import parse_target, import_target, patch_symbol, run_cli


def main():
    cfg = parse_target()
    selector, quant = import_target(cfg)

    QF = selector.QuantFormat

    def route_kurt_only(kurtosis, diffusion_snr, is_shortcut=False):
        if is_shortcut:
            return QF.MXFP8_E4M3
        if kurtosis > selector.KURTOSIS_HIGH:
            return QF.FLOAT8_E5M2
        if kurtosis <= selector.KURTOSIS_LOW:
            return QF.INT8_ASYM
        return QF.FLOAT8_E4M3

    # Patch selector AND quant (quant did `from selector import route_...`).
    patch_symbol("route_format_by_kurtosis_snr", route_kurt_only, selector, quant)
    print("[ablation patch] κ-only routing installed (SNR ignored)")
    run_cli(cfg)


if __name__ == "__main__":
    main()
