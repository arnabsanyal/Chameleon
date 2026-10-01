"""
Ablation 1 addendum: bump κ thresholds (3, 5) → (3.5, 5.5), keep SNR thresholds
fixed. Demonstrates routing robustness to threshold choice.

This patch leaves route_format_by_kurtosis_snr unchanged but mutates the
module-level KURTOSIS_LOW / KURTOSIS_HIGH constants the function reads from.
"""

from _shim import parse_target, import_target, patch_symbol, run_cli


def main():
    cfg = parse_target()
    selector, quant = import_target(cfg)

    # route_format_by_kurtosis_snr reads these from its own (selector) globals,
    # so selector is the essential target; patch quant too for uniformity + the
    # loud no-op guard.
    patch_symbol("KURTOSIS_LOW", 3.5, selector, quant)
    patch_symbol("KURTOSIS_HIGH", 5.5, selector, quant)
    print("[ablation patch] κ thresholds perturbed: (3.0, 5.0) → (3.5, 5.5)")
    run_cli(cfg)


if __name__ == "__main__":
    main()
