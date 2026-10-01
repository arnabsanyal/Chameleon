"""
Chameleon weight fold for the MixDQ pipeline ("Chameleon-on-MixDQ").

This is the bridge that lets Chameleon's *adaptive weight palette* ride on top
of MixDQ's *calibrated activation quantisation*.  MixDQ's fake-quant path folds
each weight with a single symmetric per-output-channel scale (`delta_list`); this
module replaces only that fold with Chameleon's SNR-driven format search, while
MixDQ keeps full ownership of the activation path (static calibrated `act_scales`
+ BOS + mixed precision — the part that already reaches ~21.4 FID on SDXL-Turbo).

Design rationale (why this and not the standalone Chameleon-LCM quantiser):
  The standalone Chameleon-LCM re-derives activation scales with dynamic
  per-sample min/max micro-scaling, which is outlier-sensitive in the 1-step
  Turbo regime and over-saturates (FID 36.6/43.9 at W8A8/W4A8).  MixDQ's static
  calibrated activation scales are robust there.  So we keep MixDQ's activations
  untouched and contribute only the weight palette:
    - W8: per-output-channel SNR selection over {INT8_SYM, INT8_ASYM, MXINT8}
    - W4: per-layer (group_size, format) search over the widened palette
          {INT4_ASYM, NF4, FP4_E2M1, MXINT4, MXFP4_E2M1} x {32,64,128,192,288}

Both `select_w8_formats_per_channel` and `group_format_quantize` already flatten
to (out, -1), so 2-D Linear and 4-D Conv2d weights are handled identically.

Usage (from the patched MixDQ pipeline.py, gated by MIXDQ_CHAMELEON_WEIGHTS=1):
    from chameleon_weight_fold import chameleon_fake_quant_weight
    w_fake, info = chameleon_fake_quant_weight(weight.float(), w_bit)
"""

import os
import sys
import torch

# Weight palette primitives live next to this file (chameleon_lcm_quant.py) —
# single source of truth for format definitions and SNR search.  Ensure that
# dir is importable even when this module is loaded from inside the MixDQ
# HF-cache pipeline (which sets MIXDQ_CHAMELEON_SRC to point here).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from chameleon_lcm_quant import (  # noqa: E402
    WeightFormat,
    quantize_int8_symmetric,
    quantize_int8_asymmetric,
    mx_block_quantize_per_row,
    group_format_quantize,
    search_w4_format_per_layer,
    select_w8_formats_per_channel,
)


def _fold_w8_per_channel(weight: torch.Tensor):
    """W8: assemble the dequantised weight from per-output-channel format codes.

    Mirrors ChameleonLCMQuantizedLinear._quantize_weights (the W8 branch), kept
    here so the fold is callable without instantiating a wrapper module.
    """
    codes = select_w8_formats_per_channel(weight)
    q_w = torch.zeros_like(weight)
    for fmt_val in set(codes):
        mask = [i for i, c in enumerate(codes) if c == fmt_val]
        subset = weight[mask]
        if fmt_val == int(WeightFormat.INT8_SYM):
            q_w[mask] = quantize_int8_symmetric(subset)[0]
        elif fmt_val == int(WeightFormat.INT8_ASYM):
            q_w[mask] = quantize_int8_asymmetric(subset)[0]
        elif fmt_val == int(WeightFormat.MXINT8):
            q_w[mask] = mx_block_quantize_per_row(subset, use_fp8=False)
    return q_w, {"codes": codes}


def chameleon_fake_quant_weight(weight: torch.Tensor, weight_bits: int):
    """Return (dequantised fake weight, info dict) using Chameleon's palette.

    weight       : fp tensor, (out, in) for Linear or (out, in, kh, kw) for Conv2d.
    weight_bits  : 8 -> per-output-channel SNR selection;
                   4 -> per-layer (group_size, format) search.
    The returned tensor matches `weight`'s shape (and is cast back to its dtype
    by the caller).
    """
    w = weight.float()
    if weight_bits == 4:
        g, f, snr = search_w4_format_per_layer(w)
        q_w = group_format_quantize(w, g, f)
        return q_w, {"group_size": g, "format": int(f), "snr_db": snr}
    # default: 8-bit per-output-channel
    return _fold_w8_per_channel(w)
