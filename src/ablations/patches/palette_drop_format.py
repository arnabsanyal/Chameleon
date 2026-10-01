"""
Ablation 2, drop-one rows: remove one activation format from the palette.
Whenever the routing function would have returned the dropped format, fall
back to the nearest neighbour:

    INT8_ASYM   → FLOAT8_E4M3   (next-cleanest)
    FLOAT8_E4M3 → FLOAT8_E5M2   (next-noisiest)
    FLOAT8_E5M2 → FLOAT8_E4M3   (next-cleanest)
    MXFP8_E4M3  → FLOAT8_E4M3   (kills shortcut treatment — the headline row)

For W4 weight palette drops (NF4, MXFP4_E2M1) the substitution happens at the
weight-quantization stage, NOT here; pass `--weight-drop NF4` instead and the
shim will edit the candidate list inside the quant module.

    python palette_drop_format.py --target sdxl --drop MXFP8_E4M3 -- ...
    python palette_drop_format.py --target sdxl --weight-drop NF4   -- ...
"""

import argparse

from _shim import parse_target, import_target, patch_symbol, run_cli

ACT_FALLBACK = {
    "INT8_ASYM":   "FLOAT8_E4M3",
    "FLOAT8_E4M3": "FLOAT8_E5M2",
    "FLOAT8_E5M2": "FLOAT8_E4M3",
    "MXFP8_E4M3":  "FLOAT8_E4M3",
}


def main():
    cfg = parse_target()
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--drop",        choices=list(ACT_FALLBACK))
    g.add_argument("--weight-drop", choices=["NF4", "MXFP4_E2M1",
                                             "MXINT8", "INT8_SYM"])
    args = p.parse_args(cfg["leftover"])

    selector, quant = import_target(cfg)

    if args.drop:
        QF = selector.QuantFormat
        dropped  = getattr(QF, args.drop)
        fallback = getattr(QF, ACT_FALLBACK[args.drop])
        original = selector.route_format_by_kurtosis_snr

        def route_drop(kurtosis, diffusion_snr, is_shortcut=False):
            f = original(kurtosis, diffusion_snr, is_shortcut)
            return fallback if f == dropped else f

        patch_symbol("route_format_by_kurtosis_snr", route_drop, selector, quant)
        print(f"[ablation patch] dropped activation format {args.drop} → "
              f"{ACT_FALLBACK[args.drop]}")

    else:
        # The actual constants live in snr_format_selector and are imported
        # into chameleon_quant via `from snr_format_selector import ...`. Both
        # names refer to the SAME list object, so an in-place .remove() on
        # one is seen by the other AND by SNRFormatSelector.__init__ when it
        # later does `self.formats = list(WEIGHT_FORMATS_4BIT)`.
        target = args.weight_drop
        QF = selector.QuantFormat
        target_fmt = getattr(QF, target)

        modified = False
        for attr in ("WEIGHT_FORMATS_4BIT", "WEIGHT_FORMATS_8BIT"):
            lst = getattr(selector, attr, None)
            if lst is None:
                continue
            if target_fmt in lst:
                before = [f.name for f in lst]
                lst.remove(target_fmt)        # in-place; both views update
                after = [f.name for f in lst]
                print(f"[ablation patch] {attr}: {before} → {after}")
                modified = True
        if not modified:
            raise RuntimeError(
                f"weight format {target} not found in any palette; "
                f"available: WEIGHT_FORMATS_4BIT={[f.name for f in selector.WEIGHT_FORMATS_4BIT]}, "
                f"WEIGHT_FORMATS_8BIT={[f.name for f in selector.WEIGHT_FORMATS_8BIT]}"
            )

    run_cli(cfg)


if __name__ == "__main__":
    main()
