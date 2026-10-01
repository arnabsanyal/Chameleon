"""
Chameleon-LCM weight palette (Chameleon-on-MixDQ).

Self-contained weight-format primitives for the few-step Chameleon-LCM model.
This is the *weight* half of the framework; the *activation* half is owned by
MixDQ (calibrated static act_scales + BOS + mixed precision), so unlike the old
standalone Chameleon-LCM there is **no** kurtosis/SNR activation routing here.

What lives here:
  - WeightFormat enum + the palettes WEIGHT_FORMATS_{8,4}BIT and GROUP_SIZES_W4
  - INT8 / MX / group (INT4/NF4/FP4/MXINT4/MXFP4) quant primitives
  - group_format_quantize, search_w4_format_per_layer, select_w8_formats_per_channel

Consumers:
  - chameleon_weight_fold.chameleon_fake_quant_weight (the MixDQ pipeline hook)
  - src/ablations (weight-palette ablations patch WEIGHT_FORMATS_* here)

History: extracted from the old standalone src/chameleon-lcm/generation/
chameleon_lcm_quant.py, dropping everything activation-related (the dynamic
per-sample micro-scaling that over-saturated and lost ~15 FID points to MixDQ).
"""

import math
from enum import IntEnum
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ── Weight format enum ────────────────────────────────────────────────────────

class WeightFormat(IntEnum):
    INT8_SYM    = 0
    INT8_ASYM   = 1
    MXINT8      = 2
    INT4_ASYM   = 3
    NF4         = 4
    FP4_E2M1    = 5
    MXINT4      = 6   # Microscaling INT4: per-32-block shared power-of-2 exp + INT4 sym
    MXFP4_E2M1  = 7   # Microscaling FP4 E2M1: per-32-block shared exp + FP4 element


WEIGHT_FORMATS_8BIT = [WeightFormat.INT8_SYM, WeightFormat.INT8_ASYM, WeightFormat.MXINT8]
# Widened W4 palette (paper §4.2 F^(4)_w). In the 1-step regime there is no
# temporal averaging of weight-quant error, so residual quality is dominated by
# the weight palette — block-scaled MX formats and larger group sizes pay off.
WEIGHT_FORMATS_4BIT = [
    WeightFormat.INT4_ASYM, WeightFormat.NF4, WeightFormat.FP4_E2M1,
    WeightFormat.MXINT4, WeightFormat.MXFP4_E2M1,
]
GROUP_SIZES_W4 = [32, 64, 128, 192, 288]

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
MX_BLOCK_SIZE = 32

# NF4 quantile-optimal levels (QLoRA, Dettmers et al. 2023)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
])

# FP4 E2M1 representable values (1s 2e 1m)
FP4_E2M1_LEVELS = torch.tensor([
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
])


# ── INT8 primitives ──────────────────────────────────────────────────────────

def quantize_int8_symmetric(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8: q in [-128, 127]."""
    if tensor.ndim >= 2:
        flat = tensor.reshape(tensor.shape[0], -1)  # reshape: may be non-contiguous
        max_abs = flat.abs().amax(dim=1).clamp(min=1e-8)
    else:
        max_abs = tensor.abs().max().clamp(min=1e-8)
    scale = max_abs / 127.0
    if tensor.ndim >= 2:
        shape = [-1] + [1] * (tensor.ndim - 1)
        q = torch.clamp(torch.round(tensor / scale.view(shape)), -128, 127)
        return q * scale.view(shape), scale
    q = torch.clamp(torch.round(tensor / scale), -128, 127)
    return q * scale, scale


def quantize_int8_asymmetric(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel asymmetric INT8: q in [0, 255]."""
    if tensor.ndim >= 2:
        flat = tensor.reshape(tensor.shape[0], -1)
        t_min = flat.amin(dim=1)
        t_max = flat.amax(dim=1)
    else:
        t_min, t_max = tensor.min(), tensor.max()
    scale = ((t_max - t_min) / 255.0).clamp(min=1e-8)
    zp = torch.clamp(torch.round(-t_min / scale), 0, 255)
    if tensor.ndim >= 2:
        shape = [-1] + [1] * (tensor.ndim - 1)
        q = torch.clamp(torch.round(tensor / scale.view(shape) + zp.view(shape)), 0, 255)
        return (q - zp.view(shape)) * scale.view(shape), scale, zp
    q = torch.clamp(torch.round(tensor / scale + zp), 0, 255)
    return (q - zp) * scale, scale, zp


# ── MX block primitives ──────────────────────────────────────────────────────

def mx_block_quantize_per_row(weight: torch.Tensor, use_fp8: bool = False,
                              block_size: int = MX_BLOCK_SIZE) -> torch.Tensor:
    """Vectorised per-output-channel MX quantization (blocks formed within each row)."""
    orig_shape = weight.shape
    n_out = orig_shape[0]
    flat = weight.reshape(n_out, -1)
    in_flat = flat.shape[1]
    pad = (block_size - in_flat % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    n_blocks = flat.shape[1] // block_size
    blocks = flat.reshape(n_out, n_blocks, block_size)

    max_abs = blocks.abs().amax(dim=2, keepdim=True).clamp(min=1e-30)
    scale_block = 2.0 ** torch.floor(torch.log2(max_abs))
    normed = blocks / scale_block

    if use_fp8:
        q = torch.clamp(normed, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        exp = torch.floor(torch.log2(q.abs().clamp(min=1e-30))).clamp(min=-7)
        lsb = (2.0 ** exp) / 8.0
        q = torch.round(q / lsb.clamp(min=1e-30)) * lsb
    else:
        int8_scale = 2.0 / 127.0
        q = torch.clamp(torch.round(normed / int8_scale), -128, 127) * int8_scale

    result = (q * scale_block).reshape(n_out, n_blocks * block_size)
    if pad:
        result = result[:, :in_flat]
    return result.reshape(orig_shape)


# ── Group-based quantization (for W4A8) ──────────────────────────────────────

def _reshape_into_groups(weight: torch.Tensor, group_size: int):
    """Reshape weight [out, in, ...] into groups along input dim."""
    d_out = weight.shape[0]
    d_in_flat = weight[0].numel()
    flat = weight.reshape(d_out, d_in_flat)
    pad = (group_size - d_in_flat % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    n_groups = flat.shape[1] // group_size
    grouped = flat.reshape(d_out * n_groups, group_size)
    return grouped, d_out, d_in_flat, pad


def _group_quant_int4_asym(grouped: torch.Tensor) -> torch.Tensor:
    """Per-group asymmetric INT4: q in [0, 15]."""
    g_min = grouped.amin(dim=1, keepdim=True)
    g_max = grouped.amax(dim=1, keepdim=True)
    scale = ((g_max - g_min) / 15.0).clamp(min=1e-8)
    zp = torch.clamp(torch.round(-g_min / scale), 0, 15)
    q = torch.clamp(torch.round(grouped / scale + zp), 0, 15)
    return (q - zp) * scale


def _group_quant_nf4(grouped: torch.Tensor) -> torch.Tensor:
    """NormalFloat4: snap to 16 quantile-optimal levels per group."""
    levels = NF4_LEVELS.to(grouped.device, grouped.dtype)
    max_abs = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normed = grouped / max_abs
    diffs = (normed.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    idx = diffs.argmin(dim=-1)
    return levels[idx] * max_abs


def _group_quant_fp4_e2m1(grouped: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1: snap to representable values per group."""
    levels = FP4_E2M1_LEVELS.to(grouped.device, grouped.dtype)
    max_abs = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 6.0
    scaled = grouped / scale
    diffs = (scaled.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    idx = diffs.argmin(dim=-1)
    return levels[idx] * scale


def _group_quant_mx4(grouped: torch.Tensor, use_fp4: bool,
                     block_size: int = MX_BLOCK_SIZE) -> torch.Tensor:
    """Microscaling 4-bit within each group row (MXINT4 / MXFP4 E2M1)."""
    n_rows, gs = grouped.shape
    pad = (block_size - gs % block_size) % block_size
    g = F.pad(grouped, (0, pad)) if pad else grouped
    n_blocks = g.shape[1] // block_size
    blocks = g.reshape(n_rows * n_blocks, block_size)

    max_abs = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    scale_block = (2.0 ** torch.floor(torch.log2(max_abs)))
    normed = blocks / scale_block

    if use_fp4:
        levels = FP4_E2M1_LEVELS.to(normed.device, normed.dtype)
        diffs = (normed.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
        q = levels[diffs.argmin(dim=-1)]
    else:
        int4_scale = 2.0 / 7.0
        q = torch.clamp(torch.round(normed / int4_scale), -8, 7) * int4_scale

    result = (q * scale_block).reshape(n_rows, n_blocks * block_size)
    if pad:
        result = result[:, :gs]
    return result


def group_format_quantize(weight: torch.Tensor, group_size: int,
                          fmt: WeightFormat) -> torch.Tensor:
    """Quantize weight with group-based format; return dequantized tensor."""
    grouped, d_out, d_in_flat, pad = _reshape_into_groups(weight, group_size)

    if fmt == WeightFormat.INT4_ASYM:
        q = _group_quant_int4_asym(grouped)
    elif fmt == WeightFormat.NF4:
        q = _group_quant_nf4(grouped)
    elif fmt == WeightFormat.FP4_E2M1:
        q = _group_quant_fp4_e2m1(grouped)
    elif fmt == WeightFormat.MXINT4:
        q = _group_quant_mx4(grouped, use_fp4=False)
    elif fmt == WeightFormat.MXFP4_E2M1:
        q = _group_quant_mx4(grouped, use_fp4=True)
    else:
        raise ValueError(f"Unsupported group format: {fmt}")

    n_groups = grouped.shape[0] // d_out
    result = q.reshape(d_out, n_groups * group_size)
    if pad:
        result = result[:, :d_in_flat]
    return result.reshape(weight.shape)


# ── SNR + format search ──────────────────────────────────────────────────────

def compute_snr_db(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """SNR in dB between original and quantized tensors."""
    signal = (original.float() ** 2).mean()
    noise = ((original.float() - quantized.float()) ** 2).mean()
    if noise < 1e-12:
        return 100.0
    return 10.0 * math.log10(float(signal / noise))


def select_w8_formats_per_channel(weight: torch.Tensor,
                                  formats: List[WeightFormat] = WEIGHT_FORMATS_8BIT
                                  ) -> List[int]:
    """Per-output-channel SNR-based W8 format selection (vectorised over formats)."""
    n_out = weight.shape[0]
    w2d = weight.reshape(n_out, -1).float()
    signal = (w2d ** 2).mean(dim=1).clamp(min=1e-12)

    snr_per_fmt = []
    for fmt in formats:
        if fmt == WeightFormat.INT8_SYM:
            q_w = quantize_int8_symmetric(weight)[0]
        elif fmt == WeightFormat.INT8_ASYM:
            q_w = quantize_int8_asymmetric(weight)[0]
        elif fmt == WeightFormat.MXINT8:
            q_w = mx_block_quantize_per_row(weight, use_fp8=False)
        else:
            continue
        noise = ((w2d - q_w.reshape(n_out, -1).float()) ** 2).mean(dim=1).clamp(min=1e-12)
        snr_per_fmt.append(10.0 * torch.log10(signal / noise))

    snr_stack = torch.stack(snr_per_fmt, dim=0)
    best_idx = snr_stack.argmax(dim=0).tolist()
    fmt_codes = [int(f) for f in formats if f in
                 (WeightFormat.INT8_SYM, WeightFormat.INT8_ASYM, WeightFormat.MXINT8)]
    return [fmt_codes[i] for i in best_idx]


def search_w4_format_per_layer(weight: torch.Tensor,
                               group_sizes: List[int] = GROUP_SIZES_W4,
                               formats: List[WeightFormat] = WEIGHT_FORMATS_4BIT
                               ) -> Tuple[int, WeightFormat, float]:
    """Exhaustive (group_size, format) search for W4A8. Returns best (g, f, snr)."""
    best_snr, best_g, best_f = -1.0, group_sizes[0], formats[0]
    for g in group_sizes:
        for f in formats:
            q_w = group_format_quantize(weight, g, f)
            snr = compute_snr_db(weight, q_w)
            if snr > best_snr:
                best_snr, best_g, best_f = snr, g, f
    return best_g, best_f, best_snr
