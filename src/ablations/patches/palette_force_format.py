"""
Ablation 2, single-format baselines: every layer forced to one activation
format regardless of κ or SNR. Shortcut layers keep MXFP8 unless --include-
shortcut is passed.

    python palette_force_format.py --target sdxl --format INT8_ASYM \\
        [--include-shortcut] -- <forwarded CLI args>

Recognised --format values: INT8_ASYM, FLOAT8_E4M3, FLOAT8_E5M2, MXFP8_E4M3.
"""

import argparse
import sys

from _shim import parse_target, import_target, patch_symbol, run_cli


def main():
    cfg = parse_target()
    # Pull out shim-specific flags from leftover (after --target was consumed).
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True,
                   choices=["INT8_ASYM", "FLOAT8_E4M3",
                            "FLOAT8_E5M2", "MXFP8_E4M3"])
    p.add_argument("--include-shortcut", action="store_true",
                   help="Also force shortcut layers to --format")
    args = p.parse_args(cfg["leftover"])

    selector, quant = import_target(cfg)
    target_format = getattr(selector.QuantFormat, args.format)
    include_shortcut = args.include_shortcut

    def route_constant(kurtosis, diffusion_snr, is_shortcut=False):
        if is_shortcut and not include_shortcut:
            return selector.QuantFormat.MXFP8_E4M3
        return target_format

    patch_symbol("route_format_by_kurtosis_snr", route_constant, selector, quant)
    print(f"[ablation patch] forced single format = {args.format}"
          f" (shortcut overridden = {include_shortcut})")
    run_cli(cfg)


if __name__ == "__main__":
    main()
