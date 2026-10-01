"""
Chameleon-DiT: Adaptive Quantization for Diffusion Transformers (PixArt-alpha)

Fuses two PTQ approaches for DiTs (§5 of the Chameleon-DiT design doc):

  Weights — Evolutionary "Format + Granularity" Search (§5.2):
    For each linear layer, an exhaustive greedy search finds the optimal
    (g_l, f_l) tuple: group size g ∈ {32,64,128,192,288} and format
    f ∈ {INT4, NF4, FP4} that maximises quantisation SNR.
    Weights are grouped along the INPUT-channel dimension.

  Activations — Macro-Routing + Micro-Scaling (§5.3):
    • Macro-routing (Chameleon's job): AOT kurtosis/SNR profiling of
      input activations → per-bucket format LUT.
        Bucket ≈ t → T  (pure noise)  : FP8 E5M2
        Bucket ≈ t → T/2 (transition) : FP8 E4M3
        Bucket ≈ t → 0  (clean signal): INT8 ASYM
    • Micro-scaling: dynamic per-forward scaling in the macro-routed format.
      act_mode (default 'pertensor'):
        'pertensor'   — one global scale/zp for the whole tensor. CANONICAL.
        'group-macro' — group-wise quant in the macro format. ABLATION.
        'group-int8'  — group-wise asym INT8 everywhere (routing off). ABLATION.
      [2026-06: we tried Q-DiT-style group-wise activations expecting a W4A8 win;
       it REGRESSED on PixArt-alpha (W8A8 24.29→26.69, W4A8 23.83→24.34) on BOTH
       FID and CLIP, and group-int8 collapsed Chameleon-DiT's W8A8 onto Q-DiT
       (27.59≈27.69). Per-tensor was a feature, not a bug — reverted to default.
       The W4A8 gap vs Q-DiT (23.83 vs 22.24) is weight-side, not activation-side.]

Architectural fork (§5.1):
  is_unet → Classic Chameleon (SDXL, see src/sdxl-chameleon)
  is_dit  → Chameleon-DiT  (this module)
  No MX shortcut logic in DiT mode.

Reference: Chameleon-DiT design doc §5 (chameleon-q-dit-extension-prompt.pdf)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from enum import Enum
from diffusers import PixArtAlphaPipeline
from tqdm import tqdm
import os
import json
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402



# ── Search space ───────────────────────────────────────────────────────────────

GROUP_SIZE_SEARCH: List[int] = [32, 64, 128, 192, 288]


class WeightFormat(Enum):
    """Weight formats searched by Chameleon-DiT (§5.2), across 4/6/8-bit widths."""
    # 4-bit
    INT4_ASYM = "int4_asym"    # Asymmetric INT4: uniform, per-group
    NF4       = "nf4"          # NormalFloat4: 16 quantile-optimal levels
    FP4_E2M1  = "fp4_e2m1"    # FP4 E2M1: 1s 2e 1m, pos. {0,.5,1,1.5,2,3,4,6}
    # 6-bit
    INT6_SYM  = "int6_sym"     # Symmetric INT6: scale = max_abs/31, q ∈ [-32, 31]
    INT6_ASYM = "int6_asym"    # Asymmetric INT6: scale = (max-min)/63, q ∈ [0, 63]
    # 8-bit
    INT8_SYM  = "int8_sym"     # Symmetric INT8: scale = max_abs/127, q ∈ [-128, 127]
    INT8_ASYM = "int8_asym"    # Asymmetric INT8: scale = (max-min)/255, q ∈ [0, 255]
    MXINT8    = "mxint8"       # Microscaling INT8: per-group shared-exp + INT8 sym


# Per-bit-width format search lists
FORMAT_SEARCH_4BIT: List[WeightFormat] = [
    WeightFormat.INT4_ASYM,
    WeightFormat.NF4,
    WeightFormat.FP4_E2M1,
]
FORMAT_SEARCH_6BIT: List[WeightFormat] = [
    WeightFormat.INT6_SYM,
    WeightFormat.INT6_ASYM,
]
FORMAT_SEARCH_8BIT: List[WeightFormat] = [
    WeightFormat.INT8_SYM,
    WeightFormat.INT8_ASYM,
    WeightFormat.MXINT8,
]

FORMAT_SEARCH_BY_BITS: Dict[int, List[WeightFormat]] = {
    4: FORMAT_SEARCH_4BIT,
    6: FORMAT_SEARCH_6BIT,
    8: FORMAT_SEARCH_8BIT,
}

# Default (backward-compat alias)
FORMAT_SEARCH: List[WeightFormat] = FORMAT_SEARCH_4BIT


# ── Activation format codes (macro-routing LUT) ────────────────────────────────

FORMAT_INT8_ASYM  = 0   # Clean signal: symmetric integer
FORMAT_FP8_E4M3   = 1   # Transition: float8 E4M3
FORMAT_FP8_E5M2   = 2   # Pure noise: float8 E5M2

FORMAT_NAMES = {
    FORMAT_INT8_ASYM: "INT8_ASYM",
    FORMAT_FP8_E4M3:  "FP8_E4M3",
    FORMAT_FP8_E5M2:  "FP8_E5M2",
}

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


# ── Kurtosis / diffusion-SNR routing thresholds ────────────────────────────────

KURTOSIS_HIGH      = 5.0
KURTOSIS_LOW       = 3.0
DIFF_SNR_LOW       = 0.2
DIFF_SNR_HIGH      = 2.0


# ── Lookup tables ──────────────────────────────────────────────────────────────

# NF4 quantile levels (16 values from QLoRA paper, optimal for N(0,1))
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,  0.33791524171829224,
     0.44070982933044434,  0.5626170039176941,   0.7229568362236023,   1.0,
])

# FP4 E2M1 representable values (1s 2e 1m, bias=1)
FP4_E2M1_LEVELS = torch.tensor([
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
])


# ── Group quantisation primitives (vectorised) ─────────────────────────────────

def _group_quant_int4(grouped: torch.Tensor) -> torch.Tensor:
    """Asymmetric INT4 for a (N, group_size) matrix — one row per group.
    Returns dequantised tensor of the same shape (fake-quant)."""
    qmax  = 15  # 4-bit unsigned
    g_min = grouped.min(dim=1, keepdim=True)[0]
    g_max = grouped.max(dim=1, keepdim=True)[0]
    scale = (g_max - g_min).clamp(min=1e-8) / qmax
    zp    = torch.clamp(torch.round(-g_min / scale), 0, qmax)
    q     = torch.clamp(torch.round(grouped / scale + zp), 0, qmax)
    return (q - zp) * scale


def _group_quant_nf4(grouped: torch.Tensor) -> torch.Tensor:
    """NF4 quantisation for a (N, group_size) matrix — one row per group.
    Each row is normalised by its max_abs, snapped to NF4 levels, scaled back."""
    max_abs   = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    normed    = grouped / max_abs               # → [-1, 1]
    levels    = NF4_LEVELS.to(grouped.device, grouped.dtype)
    mids      = (levels[:-1] + levels[1:]) / 2.0
    indices   = torch.bucketize(normed.contiguous(), mids)
    return levels[indices] * max_abs


def _group_quant_fp4(grouped: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 quantisation for a (N, group_size) matrix — one row per group.
    Each row is scaled into [-6, 6], snapped to FP4 levels, scaled back."""
    max_abs   = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    scale     = max_abs / 6.0
    normed    = grouped / scale                 # → [-6, 6]
    levels    = FP4_E2M1_LEVELS.to(grouped.device, grouped.dtype)
    mids      = (levels[:-1] + levels[1:]) / 2.0
    indices   = torch.bucketize(normed.contiguous(), mids)
    return levels[indices] * scale


def _group_quant_int6_sym(grouped: torch.Tensor) -> torch.Tensor:
    """Symmetric INT6 for a (N, group_size) matrix — one row per group.
    q ∈ [-32, 31], scale = max_abs / 31."""
    max_abs = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    scale   = max_abs / 31.0
    q       = torch.clamp(torch.round(grouped / scale), -32, 31)
    return q * scale


def _group_quant_int6_asym(grouped: torch.Tensor) -> torch.Tensor:
    """Asymmetric INT6 for a (N, group_size) matrix — one row per group.
    q ∈ [0, 63], scale = (max - min) / 63."""
    g_min = grouped.min(dim=1, keepdim=True)[0]
    g_max = grouped.max(dim=1, keepdim=True)[0]
    scale = (g_max - g_min).clamp(min=1e-8) / 63.0
    zp    = torch.clamp(torch.round(-g_min / scale), 0, 63)
    q     = torch.clamp(torch.round(grouped / scale + zp), 0, 63)
    return (q - zp) * scale


def _group_quant_int8_sym(grouped: torch.Tensor) -> torch.Tensor:
    """Symmetric INT8 for a (N, group_size) matrix — one row per group.
    q ∈ [-128, 127], scale = max_abs / 127."""
    max_abs = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    scale   = max_abs / 127.0
    q       = torch.clamp(torch.round(grouped / scale), -128, 127)
    return q * scale


def _group_quant_int8_asym(grouped: torch.Tensor) -> torch.Tensor:
    """Asymmetric INT8 for a (N, group_size) matrix — one row per group.
    q ∈ [0, 255], scale = (max - min) / 255."""
    g_min = grouped.min(dim=1, keepdim=True)[0]
    g_max = grouped.max(dim=1, keepdim=True)[0]
    scale = (g_max - g_min).clamp(min=1e-8) / 255.0
    zp    = torch.clamp(torch.round(-g_min / scale), 0, 255)
    q     = torch.clamp(torch.round(grouped / scale + zp), 0, 255)
    return (q - zp) * scale


def _group_quant_mxint8(grouped: torch.Tensor) -> torch.Tensor:
    """Microscaling INT8 — per-group shared exponent + INT8 sym.
    In the group-quantisation context each group IS the MX block:
    scale = 2^floor(log2(max_abs)), then INT8 sym within the block."""
    max_abs    = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-10)
    eblock     = torch.floor(torch.log2(max_abs))
    block_scale = torch.pow(2.0, eblock)
    normed     = grouped / block_scale           # values land in (-2, 2)
    int8_scale = 2.0 / 127.0
    q_int      = torch.clamp(torch.round(normed / int8_scale), -128.0, 127.0)
    return q_int * int8_scale * block_scale


# ── Weight-level helpers ───────────────────────────────────────────────────────

def _reshape_weight_into_groups(
    weight: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Reshape W ∈ R^{d_out × d_in_flat} into (d_out * num_groups, group_size).
    Returns (grouped, d_out, d_in_flat, pad).
    """
    d_out    = weight.shape[0]
    w_flat   = weight.reshape(d_out, -1).float()
    d_in     = w_flat.shape[1]

    pad = (group_size - d_in % group_size) % group_size
    if pad:
        w_flat = F.pad(w_flat, (0, pad))

    n_groups = w_flat.shape[1] // group_size
    grouped  = w_flat.reshape(d_out * n_groups, group_size)
    return grouped, d_out, d_in, pad


# Formats whose scale is derived from the asymmetric [min, max] range; the rest
# use a symmetric max-abs range.  Used by the clip-ratio search (#2).
_ASYM_FORMATS = {WeightFormat.INT4_ASYM, WeightFormat.INT6_ASYM, WeightFormat.INT8_ASYM}


def _clip_groups(grouped: torch.Tensor, clip_ratio: float, fmt: WeightFormat) -> torch.Tensor:
    """Shrink each group's range by `clip_ratio` before quantisation.

    Clipping trades a little clipping error for much lower rounding error on the
    bulk of the distribution — the per-group analogue of AWQ/OMSE clip search.
    Because the format primitives recompute their scale from the (clipped) group
    values, clamping here is sufficient to realise clipped quantisation.
    """
    if clip_ratio >= 1.0:
        return grouped
    if fmt in _ASYM_FORMATS:
        g_min = grouped.min(dim=1, keepdim=True)[0] * clip_ratio
        g_max = grouped.max(dim=1, keepdim=True)[0] * clip_ratio
        return torch.clamp(grouped, min=g_min, max=g_max)
    bound = grouped.abs().max(dim=1, keepdim=True)[0] * clip_ratio
    return torch.clamp(grouped, min=-bound, max=bound)


def group_format_quantize(
    weight:     torch.Tensor,
    group_size: int,
    fmt:        WeightFormat,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Quantise weight W along the input-channel dimension using (group_size, fmt).
    `clip_ratio` < 1.0 shrinks each group's range before quantising (#2).
    Returns dequantised float tensor of the same shape.
    """
    grouped, d_out, d_in, _pad = _reshape_weight_into_groups(weight, group_size)
    grouped = _clip_groups(grouped, clip_ratio, fmt)

    if fmt == WeightFormat.INT4_ASYM:
        dq = _group_quant_int4(grouped)
    elif fmt == WeightFormat.NF4:
        dq = _group_quant_nf4(grouped)
    elif fmt == WeightFormat.FP4_E2M1:
        dq = _group_quant_fp4(grouped)
    elif fmt == WeightFormat.INT6_SYM:
        dq = _group_quant_int6_sym(grouped)
    elif fmt == WeightFormat.INT6_ASYM:
        dq = _group_quant_int6_asym(grouped)
    elif fmt == WeightFormat.INT8_SYM:
        dq = _group_quant_int8_sym(grouped)
    elif fmt == WeightFormat.INT8_ASYM:
        dq = _group_quant_int8_asym(grouped)
    elif fmt == WeightFormat.MXINT8:
        dq = _group_quant_mxint8(grouped)
    else:
        raise ValueError(f"Unknown WeightFormat: {fmt}")

    # Remove padding columns and restore shape
    dq_flat = dq.reshape(d_out, -1)[:, :d_in]
    return dq_flat.reshape(weight.shape)


def compute_snr_db(
    original: torch.Tensor,
    quantized: torch.Tensor,
) -> float:
    """SNR in dB: 10·log10(E[x²] / E[(x-x̂)²])."""
    signal = torch.mean(original.float() ** 2)
    noise  = torch.mean((original.float() - quantized.float()) ** 2)
    if noise < 1e-10:
        return float('inf')
    return (10 * torch.log10(signal / noise + 1e-10)).item()


# Default clip-ratio grid for the (optional) clip search (#2). 1.0 == no clipping.
CLIP_RATIO_SEARCH: List[float] = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75]


def _err_sig(weight: torch.Tensor, dq: torch.Tensor,
             input_importance: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (error, signal) energies — output-error-weighted if importance given."""
    w = weight.float()
    d = dq.float()
    if input_importance is None:
        return ((w - d) ** 2).sum(), (w ** 2).sum()
    imp = input_importance.to(w.device, torch.float32).clamp(min=0).reshape(1, -1)
    return (((w - d) ** 2) * imp).sum(), ((w ** 2) * imp).sum()


def compute_layer_snr(
    weight:           torch.Tensor,
    group_size:       int,
    fmt:              WeightFormat,
    input_importance: Optional[torch.Tensor] = None,
    clip_ratio:       float = 1.0,
) -> float:
    """Quantisation SNR (dB) for a weight tensor under (g, f, clip).

    Default: weight-only SNR, 10·log10(‖W‖²/‖W−Ŵ‖²). With `input_importance`
    (per-input-channel E[x_i²]) it becomes the INPUT-AWARE / output-error
    objective 10·log10(Σ_i E[x_i²]‖W[:,i]‖² / Σ_i E[x_i²]‖ΔW[:,i]‖²) — the
    diagonal-Hessian (GPTQ/Q-DiT) criterion.
    """
    with torch.no_grad():
        dq = group_format_quantize(weight, group_size, fmt, clip_ratio=clip_ratio)
    err, sig = _err_sig(weight, dq, input_importance)
    return (10.0 * torch.log10((sig + 1e-12) / (err + 1e-12))).item()


def search_best_format_granularity(
    weight:           torch.Tensor,
    group_sizes:      List[int]         = GROUP_SIZE_SEARCH,
    formats:          List[WeightFormat] = None,
    verbose:          bool               = False,
    input_importance: Optional[torch.Tensor] = None,
    clip_ratios:      Optional[List[float]] = None,
) -> Tuple[int, WeightFormat, float, float, float]:
    """
    Exhaustive search over (g, f[, clip]) tuples — the Chameleon-DiT step.

    `input_importance` → output-error-aware selection (#1/#2 criterion).
    `clip_ratios`      → also search the weight clip ratio (#2); default [1.0].
    Returns (best_group_size, best_format, best_clip, best_snr_db, best_err)
    where best_err is the (importance-weighted) residual error — used by the
    mixed-precision allocator to score per-layer sensitivity.
    """
    if formats is None:
        formats = FORMAT_SEARCH
    if clip_ratios is None:
        clip_ratios = [1.0]

    best_snr  = -float('inf')
    best_g    = group_sizes[0]
    best_f    = formats[0]
    best_clip = clip_ratios[0]
    best_err  = float('inf')

    for g in group_sizes:
        for f in formats:
            for c in clip_ratios:
                try:
                    with torch.no_grad():
                        dq = group_format_quantize(weight, g, f, clip_ratio=c)
                    err, sig = _err_sig(weight, dq, input_importance)
                    snr = (10.0 * torch.log10((sig + 1e-12) / (err + 1e-12))).item()
                except Exception:
                    snr, err = -float('inf'), torch.tensor(float('inf'))
                if verbose:
                    print(f"    g={g:3d}, f={f.value:<12}, clip={c:.2f}: SNR={snr:.2f} dB")
                if snr > best_snr:
                    best_snr  = snr
                    best_g    = g
                    best_f    = f
                    best_clip = c
                    best_err  = float(err)

    return best_g, best_f, best_clip, best_snr, best_err


def allocate_mixed_precision(
    layer_stats:  Dict[str, dict],
    target_bits:  float,
    base_bits:    int = 4,
    promote_bits: int = 8,
) -> set:
    """Greedy mixed-precision allocation under an average-bit budget (#1).

    `layer_stats[name]` must provide {'params', 'err_base', 'err_hi'} — the
    importance-weighted residual error at base_bits and promote_bits.  Layers are
    promoted base→promote in order of error-reduction PER EXTRA BIT
    (gain/cost density) until the parameter-weighted average bit-width reaches
    `target_bits`.  Returns the set of promoted layer names.
    """
    total_params = sum(s['params'] for s in layer_stats.values())
    extra_budget = (target_bits - base_bits) * total_params      # extra bit·params allowed
    if extra_budget <= 0 or promote_bits <= base_bits:
        return set()

    ranked = []
    for name, s in layer_stats.items():
        cost = (promote_bits - base_bits) * s['params']
        gain = max(s['err_base'] - s['err_hi'], 0.0)             # output-error removed
        if cost > 0:
            ranked.append((gain / cost, cost, name))
    ranked.sort(reverse=True)                                    # densest gain first

    promoted, spent = set(), 0.0
    for _density, cost, name in ranked:
        if spent + cost <= extra_budget:
            promoted.add(name)
            spent += cost
    return promoted


# ── Activation routing helpers ─────────────────────────────────────────────────

def compute_kurtosis(tensor: torch.Tensor) -> float:
    """Excess-free kurtosis κ = E[(X-μ)⁴] / σ⁴ (returns 3.0 on degenerate input)."""
    x  = tensor.float().flatten()
    n  = x.numel()
    if n < 2:
        return 3.0
    m1 = x.mean().item()
    m2 = (x ** 2).mean().item()
    m3 = (x ** 3).mean().item()
    m4 = (x ** 4).mean().item()

    sigma2 = m2 - m1 ** 2
    if sigma2 < 1e-10:
        return 3.0

    central_m4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return central_m4 / (sigma2 ** 2)


def compute_diffusion_snr(
    timestep:       int,
    alphas_cumprod: torch.Tensor,
) -> float:
    """Diffusion SNR at timestep t: SNR(t) = ᾱ_t / (1 - ᾱ_t)."""
    t     = min(timestep, len(alphas_cumprod) - 1)
    alpha = alphas_cumprod[t].float().item()
    denom = 1.0 - alpha
    if denom < 1e-10:
        return 0.0
    return alpha / denom


def route_format_by_kurtosis_snr(
    kurtosis:     float,
    diffusion_snr: float,
) -> int:
    """
    Macro activation-format routing (§5.3):
      Pure noise  : κ > 5.0  OR  SNR(t) < 0.2  →  FP8_E5M2
      Transition  : 3.0 < κ ≤ 5.0               →  FP8_E4M3
      Clean signal: κ ≤ 3.0  AND SNR(t) > 2.0   →  INT8_ASYM

    Returns an integer format code.
    """
    if kurtosis > KURTOSIS_HIGH or diffusion_snr < DIFF_SNR_LOW:
        return FORMAT_FP8_E5M2
    if kurtosis <= KURTOSIS_LOW and diffusion_snr > DIFF_SNR_HIGH:
        return FORMAT_INT8_ASYM
    return FORMAT_FP8_E4M3


# ── Per-bucket activation statistics ──────────────────────────────────────────

class PerBucketActivationStats:
    """
    Tracks activation statistics per timestep bucket using running moments.
    Collects n, Σx, Σx², Σx³, Σx⁴, min, max per bucket (O(1) memory).
    """

    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets
        self.n        = [0]             * num_buckets
        self.sum      = [0.0]           * num_buckets
        self.sum_sq   = [0.0]           * num_buckets
        self.sum_cub  = [0.0]           * num_buckets
        self.sum_quad = [0.0]           * num_buckets
        self.min_val  = [float('inf')]  * num_buckets
        self.max_val  = [float('-inf')] * num_buckets

    def update(self, activations: torch.Tensor, bucket_idx: int):
        if bucket_idx < 0 or bucket_idx >= self.num_buckets:
            return
        with torch.no_grad():
            flat = activations.float().flatten()
            nn   = flat.numel()
            if nn == 0:
                return
            b = bucket_idx
            self.n[b]        += nn
            self.sum[b]      += flat.sum().item()
            self.sum_sq[b]   += (flat ** 2).sum().item()
            self.sum_cub[b]  += (flat ** 3).sum().item()
            self.sum_quad[b] += (flat ** 4).sum().item()
            bmin = flat.min().item()
            bmax = flat.max().item()
            if bmin < self.min_val[b]: self.min_val[b] = bmin
            if bmax > self.max_val[b]: self.max_val[b] = bmax

    def has_data(self, bucket_idx: int) -> bool:
        return self.n[bucket_idx] > 0

    def compute_kurtosis(self, bucket_idx: int) -> float:
        b  = bucket_idx
        nn = self.n[b]
        if nn < 2:
            return 3.0
        m1 = self.sum[b]      / nn
        m2 = self.sum_sq[b]   / nn
        m3 = self.sum_cub[b]  / nn
        m4 = self.sum_quad[b] / nn
        sigma2 = m2 - m1 ** 2
        if sigma2 < 1e-10:
            return 3.0
        central_m4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        return central_m4 / (sigma2 ** 2)


# ── FP8 helpers ────────────────────────────────────────────────────────────────

def _fp8_e4m3_fake_quant(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, 'float8_e4m3fn'):
        orig = x.dtype
        return x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(orig)
    max_val = FP8_E4M3_MAX
    clamped  = x.clamp(-max_val, max_val)
    sign     = torch.sign(clamped)
    abs_val  = clamped.abs()
    min_sub  = 2.0 ** -9
    mask_z   = abs_val < min_sub
    safe     = abs_val.clamp(min=min_sub)
    exp      = torch.floor(torch.log2(safe))
    m        = safe / (2.0 ** exp)
    qm       = torch.round(m * 8.0) / 8.0
    result   = sign * qm * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


def _fp8_e5m2_fake_quant(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, 'float8_e5m2'):
        orig = x.dtype
        return x.clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX).to(torch.float8_e5m2).to(orig)
    max_val = FP8_E5M2_MAX
    clamped  = x.clamp(-max_val, max_val)
    sign     = torch.sign(clamped)
    abs_val  = clamped.abs()
    min_sub  = 2.0 ** -16
    mask_z   = abs_val < min_sub
    safe     = abs_val.clamp(min=min_sub)
    exp      = torch.floor(torch.log2(safe))
    m        = safe / (2.0 ** exp)
    qm       = torch.round(m * 4.0) / 4.0
    result   = sign * qm * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


# ── Group-wise activation quantisation (Q-DiT style) ──────────────────────────
#
# DiT activations have large per-channel variance: a few outlier channels would
# dominate a single per-tensor scale and crush the rest toward zero.  Q-DiT's key
# idea is to quantise activations *group-wise* along the channel dim, so each
# group of `act_group_size` channels gets its own scale/zp.  The old Chameleon-DiT
# path used a single global min/max (per-tensor) — that was the regression vs
# Q-DiT.  We restore group-wise micro-scaling here while keeping Chameleon's
# adaptive weight palette and (optionally) its per-bucket macro format routing.

DEFAULT_ACT_GROUP_SIZE = 128      # Q-DiT default (only used by the group-* modes)

# Activation quant modes.  PER-TENSOR is the DEFAULT and the canonical model:
# empirically it beats group-wise on PixArt-alpha (group-wise added per-group
# scale-estimation noise this model doesn't need, and at W8A8 it erased
# Chameleon-DiT's advantage over Q-DiT — see the group-wise ablation below).
ACT_MODE_PERTENSOR   = "pertensor"     # DEFAULT: per-tensor dynamic scale, macro-routed format
ACT_MODE_GROUP_MACRO = "group-macro"   # ablation: group-wise quant in the macro-routed format
ACT_MODE_GROUP_INT8  = "group-int8"    # ablation: group-wise asym INT8 everywhere (routing off)
ACT_MODES = (ACT_MODE_PERTENSOR, ACT_MODE_GROUP_MACRO, ACT_MODE_GROUP_INT8)
DEFAULT_ACT_MODE = ACT_MODE_PERTENSOR


def pertensor_quant_activation(x: torch.Tensor, fmt_code: int) -> torch.Tensor:
    """Per-tensor dynamic activation fake-quant in the routed format (the
    canonical Chameleon-DiT path).  A single scale/zp is derived from the whole
    tensor's min/max (asym INT8) or max-abs (FP8) on this forward pass.

    Empirically this beats the group-wise variants on PixArt-alpha — the group-*
    modes are kept only as a (negative) ablation.
    """
    orig_dtype = x.dtype
    x_f   = x.float()
    x_min = x_f.min().item()
    x_max = x_f.max().item()

    if fmt_code == FORMAT_INT8_ASYM:
        scale  = max((x_max - x_min) / 255.0, 1e-8)
        zp     = float(round(-x_min / scale)) - 128.0
        q      = torch.clamp(torch.round(x_f / scale + zp), -128, 127)
        result = (q - zp) * scale
    else:
        max_abs = max(abs(x_min), abs(x_max), 1e-8)
        if fmt_code == FORMAT_FP8_E4M3:
            scale  = max_abs / FP8_E4M3_MAX
            result = _fp8_e4m3_fake_quant(x_f / scale) * scale
        else:  # FORMAT_FP8_E5M2
            scale  = max_abs / FP8_E5M2_MAX
            result = _fp8_e5m2_fake_quant(x_f / scale) * scale

    return result.to(orig_dtype)


def _group_fp8(grouped: torch.Tensor, e5m2: bool) -> torch.Tensor:
    """Per-group symmetric FP8 fake-quant for a (N, group_size) matrix."""
    max_abs = grouped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    fmax    = FP8_E5M2_MAX if e5m2 else FP8_E4M3_MAX
    scale   = max_abs / fmax
    normed  = grouped / scale
    q       = _fp8_e5m2_fake_quant(normed) if e5m2 else _fp8_e4m3_fake_quant(normed)
    return q * scale


def group_quant_activation(x: torch.Tensor, fmt_code: int,
                           group_size: int = DEFAULT_ACT_GROUP_SIZE) -> torch.Tensor:
    """Group-wise (sample-wise) activation fake-quant in the routed format.

    x of shape (..., d_in) is grouped along d_in into chunks of `group_size`;
    each group's scale/zp is computed from its own min/max (asym INT8) or
    max-abs (FP8) on this forward pass.  Returns a dequantised tensor of the
    same shape and dtype.  Falls back to tensor-wise when group_size >= d_in.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    d_in = orig_shape[-1]
    xf   = x.float().reshape(-1, d_in)                 # (N, d_in)
    n    = xf.shape[0]

    gs = group_size if (group_size and group_size < d_in) else d_in
    pad = (gs - d_in % gs) % gs
    if pad:
        xf = F.pad(xf, (0, pad))
    n_groups = xf.shape[1] // gs
    grouped  = xf.reshape(n * n_groups, gs)            # one row per group

    if fmt_code == FORMAT_INT8_ASYM:
        dq = _group_quant_int8_asym(grouped)
    elif fmt_code == FORMAT_FP8_E4M3:
        dq = _group_fp8(grouped, e5m2=False)
    else:  # FORMAT_FP8_E5M2
        dq = _group_fp8(grouped, e5m2=True)

    dq = dq.reshape(n, n_groups * gs)
    if pad:
        dq = dq[:, :d_in]
    return dq.reshape(orig_shape).to(orig_dtype)


# ── Dynamic activation fake-quant (DiT mode) ──────────────────────────────────

class ChameleonDiTDynamicQuantize(nn.Module):
    """
    Dynamic activation quantisation for DiT layers.

    Macro-routing: format code per timestep bucket (AOT LUT).
    Micro-scaling: scale/zp computed on-the-fly from the current sample's
                   min/max — no static scale LUT (§5.3).

    Buffers:
        format_lut  [num_buckets]  int32  — macro format code per bucket
    """

    def __init__(self, num_timesteps: int = 1000, num_buckets: int = 10,
                 act_group_size: int = DEFAULT_ACT_GROUP_SIZE,
                 act_mode: str = DEFAULT_ACT_MODE):
        super().__init__()
        self.num_timesteps  = num_timesteps
        self.num_buckets    = num_buckets
        self.bucket_size    = num_timesteps // num_buckets
        self.act_group_size = act_group_size
        if act_mode not in ACT_MODES:
            raise ValueError(f"act_mode must be one of {ACT_MODES}, got {act_mode}")
        self.act_mode = act_mode

        # Default: FP8_E4M3 for all buckets (overwritten during calibration)
        self.register_buffer(
            'format_lut',
            torch.ones(num_buckets, dtype=torch.int32) * FORMAT_FP8_E4M3,
        )
        self.enabled: bool = False

    def calibrate_bucket(self, bucket_idx: int, format_code: int):
        """Set macro format for one bucket (no scale/zp — those are dynamic)."""
        self.format_lut[bucket_idx] = format_code

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Dynamic activation fake-quant in the routed format.

        act_mode:
          'pertensor'   (default): one global scale/zp for the whole tensor.
          'group-macro' (ablation): group-wise quant in the bucket's macro format.
          'group-int8'  (ablation): group-wise asym INT8 everywhere (routing off).
        Per-tensor is the canonical Chameleon-DiT model; the group-* modes are
        kept as a documented (negative) ablation.
        """
        if not self.enabled:
            return x

        if self.act_mode == ACT_MODE_GROUP_INT8:
            fmt_code = FORMAT_INT8_ASYM
        else:
            bucket_idx = min(t // self.bucket_size, self.num_buckets - 1)
            fmt_code   = int(self.format_lut[bucket_idx].item())

        if self.act_mode == ACT_MODE_PERTENSOR:
            return pertensor_quant_activation(x, fmt_code)
        return group_quant_activation(x, fmt_code, self.act_group_size)

    def lut_summary(self, layer_name: str = ""):
        prefix = f"[{layer_name}] " if layer_name else ""
        print(f"\n{prefix}Activation Macro-Format LUT (DiT):")
        for b in range(self.num_buckets):
            t_lo = b * self.bucket_size
            t_hi = (b + 1) * self.bucket_size - 1
            fmt_code = int(self.format_lut[b].item())
            fmt_name = FORMAT_NAMES.get(fmt_code, f"UNKNOWN({fmt_code})")
            print(f"  Bucket {b:2d} | t={t_lo:4d}-{t_hi:4d} | {fmt_name}")


# ── Quantised linear layer ─────────────────────────────────────────────────────

class ChameleonDiTQuantizedLinear(nn.Module):
    """
    Chameleon-DiT quantised nn.Linear.

    Weights:      pre-quantised with the layer-optimal (g_l, f_l) pair.
                  Groups are along the input-channel dim.
    Activations:  input is fake-quantised GROUP-WISE (Q-DiT style) using the
                  macro-format (LUT, option 2) or asym INT8 (option 1), with
                  per-group dynamic scale per forward pass.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        group_size:     int,
        weight_format:  WeightFormat,
        num_buckets:    int = 10,
        act_group_size: int = DEFAULT_ACT_GROUP_SIZE,
        act_mode:       str = DEFAULT_ACT_MODE,
        clip_ratio:     float = 1.0,
    ):
        super().__init__()
        self.layer         = original_layer
        self.group_size    = group_size
        self.weight_format = weight_format
        self.clip_ratio    = clip_ratio

        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features

        self._quantize_and_store_weights()

        self.fake_quant = ChameleonDiTDynamicQuantize(
            num_buckets=num_buckets,
            act_group_size=act_group_size,
            act_mode=act_mode,
        )

        self.activation_stats: Optional[PerBucketActivationStats] = None
        self.calibrating      = False
        self.quantized        = True
        self.current_timestep = 0

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            layer = object.__getattribute__(self, '_modules').get('layer')
            if layer is not None:
                return getattr(layer, name)
        except (KeyError, AttributeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def _quantize_and_store_weights(self):
        w = self.layer.weight.data
        with torch.no_grad():
            qw = group_format_quantize(w, self.group_size, self.weight_format,
                                       clip_ratio=self.clip_ratio)
        # group_format_quantize works in float32 internally; cast back to the
        # model's dtype (e.g. float16) so F.linear dtype is consistent.
        self.register_buffer('quantized_weight', qw.to(w.dtype))

    def set_timestep(self, timestep: int):
        self.current_timestep = timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            # Collect INPUT activation stats per bucket (those are what we quantise)
            if self.activation_stats is not None:
                bsz = self.fake_quant.bucket_size
                nb  = self.fake_quant.num_buckets
                b   = min(self.current_timestep // bsz, nb - 1)
                self.activation_stats.update(x, b)
            # Pass through with original weights (unquantised) for clean stats
            return self.layer(x)

        if self.quantized:
            # Macro-route + dynamic micro-scale for activation
            x_q    = self.fake_quant(x, self.current_timestep)
            return F.linear(x_q, self.quantized_weight, self.layer.bias)

        return self.layer(x)


# ── Main quantiser ─────────────────────────────────────────────────────────────

# Default skip patterns (sensitive layers — embedders and output head)
DEFAULT_SKIP_PATTERNS = ['proj_out', 'adaln_single']


class ChameleonDiTQuantizer:
    """
    Chameleon-DiT quantiser for PixArt-alpha.

    Step 1 — Weight quantisation:
        Exhaustive per-layer (g, f) search → inject ChameleonDiTQuantizedLinear.

    Step 2 — Activation calibration:
        Collect per-bucket activation stats → kurtosis/SNR routing → format LUT.
        Micro-scale is computed dynamically at inference (not stored in LUT).
    """

    def __init__(
        self,
        model_id:     str              = "PixArt-alpha/PixArt-XL-2-1024-MS",
        device:       str              = "cuda",
        dtype:        torch.dtype      = torch.float16,
        group_sizes:  List[int]        = None,
        formats:      List[WeightFormat] = None,
        num_buckets:  int              = 10,
        weight_bits:  int              = 4,
        act_group_size: int            = DEFAULT_ACT_GROUP_SIZE,
        act_mode:     str              = DEFAULT_ACT_MODE,
        input_aware_weights: bool      = False,
        input_aware_samples: int       = 16,
        clip_search:  bool             = False,
        mixed_precision: bool          = False,
        mp_target_bits: float          = 4.5,
        mp_promote_to: int             = 8,
    ):
        if weight_bits not in FORMAT_SEARCH_BY_BITS:
            raise ValueError(f"weight_bits must be one of {list(FORMAT_SEARCH_BY_BITS.keys())}, got {weight_bits}")
        if act_mode not in ACT_MODES:
            raise ValueError(f"act_mode must be one of {ACT_MODES}, got {act_mode}")
        self.model_id    = model_id
        self.device      = device
        self.dtype       = dtype
        self.weight_bits = weight_bits
        self.group_sizes = group_sizes if group_sizes is not None else GROUP_SIZE_SEARCH
        # If no explicit format list, pick by bit-width
        self.formats     = formats if formats is not None else FORMAT_SEARCH_BY_BITS[weight_bits]
        self.num_buckets = num_buckets
        self.act_group_size = act_group_size
        self.act_mode    = act_mode
        self.input_aware_weights = input_aware_weights
        self.input_aware_samples = input_aware_samples
        if mp_promote_to not in FORMAT_SEARCH_BY_BITS:
            raise ValueError(f"mp_promote_to must be one of {list(FORMAT_SEARCH_BY_BITS.keys())}, got {mp_promote_to}")
        self.clip_search     = clip_search
        self.mixed_precision = mixed_precision
        self.mp_target_bits  = mp_target_bits
        self.mp_promote_to   = mp_promote_to
        self.mp_promoted: set       = set()
        self.mp_avg_bits: float     = float(weight_bits)

        self.pipe:         Optional[PixArtAlphaPipeline]       = None
        self.quant_layers: List[ChameleonDiTQuantizedLinear]   = []
        self.layer_names:  List[str]                            = []

        # Per-layer search results: {layer_name: (group_size, fmt, snr)}
        # name -> (group_size, weight_format, clip_ratio, snr_db)
        self.layer_search_results: Dict[str, Tuple[int, WeightFormat, float, float]] = {}

        self.weight_quantization_complete    = False
        self.activation_calibration_complete = False

        self._hook_installed = False

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self):
        """Load PixArt-alpha pipeline."""
        print(f"Loading PixArt-alpha from {self.model_id}...")
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        self.pipe.transformer.eval()

        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            print(f"Pipeline on {torch.cuda.get_device_name(dev_idx)}")
        return self.pipe

    # ── Timestep hook ──────────────────────────────────────────────────────────

    def _install_timestep_hook(self):
        """
        Monkeypatch pipe.transformer.forward to broadcast the current timestep
        to all quantised layers before each denoising step.
        This is the clean alternative to a manual denoising loop — works for
        both calibration and generation without reimplementing the pipeline.
        """
        if self._hook_installed or self.pipe is None:
            return

        original_forward = self.pipe.transformer.forward
        quant_layers     = self.quant_layers

        def hooked_forward(*args, timestep=None, **kwargs):
            if timestep is not None:
                t_tensor = timestep
                if isinstance(t_tensor, torch.Tensor):
                    t_val = int(t_tensor.flatten()[0].item())
                else:
                    t_val = int(t_tensor)
                for layer in quant_layers:
                    layer.set_timestep(t_val)
            return original_forward(*args, timestep=timestep, **kwargs)

        self.pipe.transformer._original_forward = original_forward
        self.pipe.transformer.forward           = hooked_forward
        self._hook_installed = True

    def _remove_timestep_hook(self):
        if self.pipe is not None and hasattr(self.pipe.transformer, '_original_forward'):
            self.pipe.transformer.forward = self.pipe.transformer._original_forward
            del self.pipe.transformer._original_forward
            self._hook_installed = False

    # ── Input importance (for input-aware weight selection) ───────────────────

    def _collect_input_importance(
        self,
        candidates:          List[Tuple[str, nn.Module]],
        num_prompts:         int   = 16,
        num_inference_steps: int   = 20,
        guidance_scale:      float = 4.5,
    ) -> Dict[str, torch.Tensor]:
        """Pre-pass on the ORIGINAL (unquantised) model to estimate, per layer,
        the per-input-channel activation energy E[x_i²].  Used to make the (g,f)
        search output-error-aware (diagonal-Hessian / GPTQ-style criterion)."""
        acc: Dict[str, Optional[torch.Tensor]] = {name: None for name, _ in candidates}

        def make_hook(name):
            def hook(_module, inputs):
                x  = inputs[0].detach().float()
                xe = (x * x).reshape(-1, x.shape[-1]).sum(dim=0)   # (d_in,)
                acc[name] = xe if acc[name] is None else acc[name] + xe
            return hook

        handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in candidates]
        prompts = self._get_calibration_prompts(num_prompts)
        print(f"Input-aware weight selection: importance pre-pass over "
              f"{len(prompts)} prompts ...")
        try:
            with torch.no_grad():
                for p in tqdm(prompts, desc="Importance pre-pass"):
                    self.pipe(
                        prompt=p,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(0),
                    )
        finally:
            for h in handles:
                h.remove()
        # Replace any layer that never fired with uniform importance (= weight-only).
        return {name: acc[name] for name, _ in candidates}

    # ── Step 1: weight quantisation ───────────────────────────────────────────

    def inject_weight_quantization(
        self,
        skip_patterns: Optional[List[str]] = None,
        verbose:       bool = False,
    ):
        """
        Per-layer (g, f) search + quantised-layer injection.

        For each eligible nn.Linear in transformer_blocks, exhaustively tries all
        (group_size, format) combinations and selects the one with the highest SNR.
        """
        if self.pipe is None:
            self.load_model()

        skip = skip_patterns if skip_patterns is not None else DEFAULT_SKIP_PATTERNS

        print(f"\n{'='*60}")
        print("CHAMELEON-DiT STEP 1: WEIGHT QUANTISATION")
        print(f"{'='*60}")
        print(f"Weight bits           : {self.weight_bits} (W{self.weight_bits}A8)")
        print(f"Group sizes to search : {self.group_sizes}")
        print(f"Formats to search     : {[f.value for f in self.formats]}")
        print(f"Skip patterns         : {skip}")

        def _should_skip(name: str) -> bool:
            return any(p in name for p in skip)

        candidates = [
            (name, mod)
            for name, mod in self.pipe.transformer.named_modules()
            if isinstance(mod, nn.Linear) and not _should_skip(name)
        ]
        clip_grid = CLIP_RATIO_SEARCH if self.clip_search else [1.0]
        mp_active = self.mixed_precision and self.weight_bits == 4
        print(f"Layers to quantise    : {len(candidates)}")
        print(f"Selection criterion   : "
              f"{'input-aware (output error)' if (self.input_aware_weights or mp_active) else 'weight-only SNR'}")
        print(f"Clip search           : {'on ('+str(clip_grid)+')' if self.clip_search else 'off'}")
        print(f"Mixed precision       : "
              f"{'on (4→%d-bit, avg target %.2f)' % (self.mp_promote_to, self.mp_target_bits) if mp_active else 'off (uniform %d-bit)' % self.weight_bits}")
        print(f"{'='*60}\n")

        # Importance is needed for input-aware selection AND mixed-precision sensitivity.
        importance: Dict[str, Optional[torch.Tensor]] = {}
        if self.input_aware_weights or mp_active:
            importance = self._collect_input_importance(
                candidates, num_prompts=self.input_aware_samples
            )

        def _imp(nm):
            return importance.get(nm) if (self.input_aware_weights or mp_active) else None

        if mp_active:
            # Search each layer at BOTH 4-bit and the promote bit-width, then allocate.
            base_formats = FORMAT_SEARCH_BY_BITS[4]
            hi_formats   = FORMAT_SEARCH_BY_BITS[self.mp_promote_to]
            per_layer: Dict[str, dict] = {}
            for name, module in tqdm(candidates, desc="MP search (4-bit + promote)"):
                w = module.weight.data
                base = search_best_format_granularity(
                    w, self.group_sizes, base_formats, verbose, _imp(name), clip_grid)
                hi   = search_best_format_granularity(
                    w, self.group_sizes, hi_formats,   verbose, _imp(name), clip_grid)
                per_layer[name] = {'base': base, 'hi': hi, 'params': int(w.numel()),
                                   'module': module}

            promoted = allocate_mixed_precision(
                {n: {'params': d['params'], 'err_base': d['base'][4], 'err_hi': d['hi'][4]}
                 for n, d in per_layer.items()},
                target_bits=self.mp_target_bits, base_bits=4, promote_bits=self.mp_promote_to,
            )
            total_p  = sum(d['params'] for d in per_layer.values())
            avg_bits = sum((self.mp_promote_to if n in promoted else 4) * per_layer[n]['params']
                           for n in per_layer) / max(total_p, 1)
            self.mp_promoted = promoted
            self.mp_avg_bits = avg_bits
            print(f"\nMixed precision: promoted {len(promoted)}/{len(per_layer)} layers "
                  f"to {self.mp_promote_to}-bit → avg {avg_bits:.2f} bits "
                  f"(target {self.mp_target_bits}).\n")

            for name, d in per_layer.items():
                g, f, c, snr, _err = d['hi'] if name in promoted else d['base']
                self._build_and_replace_layer(name, d['module'], g, f, c, snr)
        else:
            for name, module in tqdm(candidates, desc="Searching (g,f[,clip]) + injecting"):
                g, f, c, snr, _err = search_best_format_granularity(
                    module.weight.data, self.group_sizes, self.formats,
                    verbose, _imp(name), clip_grid)
                if verbose:
                    print(f"  → Best: g={g}, f={f.value}, clip={c:.2f}, SNR={snr:.2f} dB")
                self._build_and_replace_layer(name, module, g, f, c, snr)

        self.weight_quantization_complete = True
        self._install_timestep_hook()
        self._print_weight_summary()

    def _build_and_replace_layer(self, name, module, group_size, fmt, clip_ratio, snr):
        """Record the per-layer choice and swap in a quantised layer."""
        self.layer_search_results[name] = (group_size, fmt, clip_ratio, snr)
        ql = ChameleonDiTQuantizedLinear(
            original_layer=module,
            group_size=group_size,
            weight_format=fmt,
            clip_ratio=clip_ratio,
            num_buckets=self.num_buckets,
            act_group_size=self.act_group_size,
            act_mode=self.act_mode,
        )
        self._replace_module(self.pipe.transformer, name, ql)
        self.quant_layers.append(ql)
        self.layer_names.append(name)

    # ── Step 2: activation calibration ────────────────────────────────────────

    def calibrate_activations(
        self,
        num_calibration_samples: int   = 64,
        num_inference_steps:     int   = 20,
        guidance_scale:          float = 4.5,
        verbose:                 bool  = False,
    ):
        """
        Calibrate per-bucket macro activation format LUT via kurtosis/SNR routing.

        Micro-scale is NOT stored — it is computed dynamically at inference.
        """
        if not self.weight_quantization_complete:
            raise RuntimeError(
                "Call inject_weight_quantization (or load_weight_quantization) first."
            )

        print(f"\n{'='*60}")
        print("CHAMELEON-DiT STEP 2: ACTIVATION CALIBRATION")
        print(f"{'='*60}")
        print(f"Calibration samples : {num_calibration_samples}")
        print(f"Inference steps     : {num_inference_steps}")
        print(f"Num buckets         : {self.num_buckets}")
        print(f"Routing thresholds  : κ_high={KURTOSIS_HIGH}, κ_low={KURTOSIS_LOW}")
        print(f"                      SNR_low={DIFF_SNR_LOW}, SNR_high={DIFF_SNR_HIGH}")
        print(f"{'='*60}\n")

        # Initialise stats + set calibrating mode
        for layer in self.quant_layers:
            layer.activation_stats = PerBucketActivationStats(self.num_buckets)
            layer.calibrating      = True
            layer.quantized        = False

        # Collect stats over calibration prompts
        prompts = self._get_calibration_prompts(num_calibration_samples)
        print(f"Running {len(prompts)} calibration samples...")

        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
                try:
                    self.pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(i),
                    )
                    if (i + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\nError at sample {i}: {e}")

        # Build format LUT from collected stats
        alphas_cumprod = self.pipe.scheduler.alphas_cumprod
        bucket_size    = 1000 // self.num_buckets

        print("\nBuilding macro activation format LUT per layer...")
        for layer, name in zip(self.quant_layers, self.layer_names):
            if layer.activation_stats is None:
                continue

            for b in range(self.num_buckets):
                if not layer.activation_stats.has_data(b):
                    # No data for this bucket — default to FP8_E4M3
                    layer.fake_quant.calibrate_bucket(b, FORMAT_FP8_E4M3)
                    continue

                kurtosis    = layer.activation_stats.compute_kurtosis(b)
                t_mid       = min(b * bucket_size + bucket_size // 2,
                                  len(alphas_cumprod) - 1)
                diff_snr    = compute_diffusion_snr(t_mid, alphas_cumprod)
                fmt_code    = route_format_by_kurtosis_snr(kurtosis, diff_snr)
                layer.fake_quant.calibrate_bucket(b, fmt_code)

            layer.fake_quant.enabled = True
            layer.calibrating        = False
            layer.quantized          = True

            if verbose:
                layer.fake_quant.lut_summary(name)

        self.activation_calibration_complete = True
        self._print_activation_summary()

    # ── Image generation ───────────────────────────────────────────────────────

    def generate_images(
        self,
        prompts:             List[str],
        output_dir:          str,
        num_inference_steps: int   = 20,
        guidance_scale:      float = 4.5,
        batch_size:          int   = 1,
        start_idx:           int   = 0,
    ):
        """Generate images with the Chameleon-DiT quantised PixArt-alpha model."""
        if not self.activation_calibration_complete:
            print("Warning: activation calibration not done — weights-only quantisation.")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating {len(prompts)} images (start_idx={start_idx})...")

        tracker = PerfTracker(
            label=f"chameleon_dit_w{self.weight_bits}a8",
            device=self.device,
        )
        tracker.reset()

        with torch.no_grad():
            for bs in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
                be    = min(bs + batch_size, len(prompts))
                batch = prompts[bs:be]
                try:
                    generators = [
                        torch.Generator(device=self.device).manual_seed(start_idx + bs + j)
                        for j in range(len(batch))
                    ]
                    tracker.batch_start()
                    result = self.pipe(
                        prompt=batch,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generators,
                    )
                    tracker.batch_end(len(batch))
                    for j, image in enumerate(result.images):
                        image.save(f"{output_dir}/{start_idx + bs + j:05d}.png")

                except Exception as e:
                    print(f"\nError at batch {start_idx + bs}: {e}")
                    for j, prompt in enumerate(batch):
                        try:
                            idx = start_idx + bs + j
                            res = self.pipe(
                                prompt=prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                generator=torch.Generator(device=self.device).manual_seed(idx),
                            )
                            res.images[0].save(f"{output_dir}/{idx:05d}.png")
                        except Exception as e2:
                            print(f"  Error at image {start_idx + bs + j}: {e2}")

        tracker.finish()
        tracker.dump_json(os.path.join(output_dir, "perf.json"))
        print(f"Saved {len(prompts)} images to {output_dir}")

    # ── Save / load ────────────────────────────────────────────────────────────

    def save_weight_quantization(self, save_path: str):
        """Save per-layer (g, f) decisions and activation format LUT to JSON."""
        if not self.weight_quantization_complete:
            raise RuntimeError("Weight quantisation not complete.")

        # Format distribution summary
        fmt_dist: Dict[str, int] = {}
        g_dist:   Dict[str, int] = {}
        for name, (g, f, _clip, _snr) in self.layer_search_results.items():
            fmt_dist[f.value] = fmt_dist.get(f.value, 0) + 1
            g_dist[str(g)]    = g_dist.get(str(g), 0) + 1

        config = {
            'model_id':       self.model_id,
            'num_buckets':    self.num_buckets,
            'weight_bits':    self.weight_bits,
            'group_sizes':    self.group_sizes,
            'formats':        [f.value for f in self.formats],
            'act_group_size': self.act_group_size,
            'act_mode':       self.act_mode,
            'input_aware_weights': self.input_aware_weights,
            'clip_search':    self.clip_search,
            'mixed_precision': self.mixed_precision,
            'mp_target_bits': self.mp_target_bits,
            'mp_promote_to':  self.mp_promote_to,
            'mp_avg_bits':    self.mp_avg_bits,
            'mp_num_promoted': len(self.mp_promoted),
            'num_layers':     len(self.quant_layers),
            'format_distribution': fmt_dist,
            'group_distribution':  g_dist,
            'layers': {},
        }

        for name, layer in zip(self.layer_names, self.quant_layers):
            g, f, clip, snr = self.layer_search_results.get(
                name, (layer.group_size, layer.weight_format,
                       getattr(layer, 'clip_ratio', 1.0), 0.0))
            config['layers'][name] = {
                'group_size':    g,
                'weight_format': f.value,
                'clip_ratio':    clip,
                'best_snr_db':   snr,
                'promoted':      name in self.mp_promoted,
            }

        # Embed activation LUT if calibration is complete
        if self.activation_calibration_complete:
            config['activation_lut'] = {}
            for name, layer in zip(self.layer_names, self.quant_layers):
                config['activation_lut'][name] = {
                    'format_lut': layer.fake_quant.format_lut.tolist(),
                }

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {save_path}")

    def load_weight_quantization(self, load_path: str):
        """Load per-layer (g, f) decisions and (optionally) activation LUT from JSON."""
        if self.pipe is None:
            self.load_model()

        with open(load_path, 'r') as f:
            config = json.load(f)

        print(f"\n{'='*60}")
        print("LOADING CHAMELEON-DiT CONFIG FROM FILE")
        print(f"{'='*60}")
        print(f"Config      : {load_path}")
        print(f"Model ID    : {config['model_id']}")
        print(f"Weight bits : {config.get('weight_bits', 4)} (W{config.get('weight_bits', 4)}A8)")
        print(f"Layers      : {config['num_layers']}")
        print(f"{'='*60}\n")

        num_buckets = config.get('num_buckets', self.num_buckets)
        # Activation granularity/mode: prefer the saved config, else current settings.
        act_group_size = config.get('act_group_size', self.act_group_size)
        act_mode       = config.get('act_mode', self.act_mode)
        self.act_group_size = act_group_size
        self.act_mode       = act_mode
        # Provenance (weights themselves are fully determined by per-layer g/f/clip).
        self.input_aware_weights = config.get('input_aware_weights', self.input_aware_weights)
        self.clip_search         = config.get('clip_search', self.clip_search)
        self.mixed_precision     = config.get('mixed_precision', self.mixed_precision)
        self.mp_target_bits      = config.get('mp_target_bits', self.mp_target_bits)
        self.mp_promote_to       = config.get('mp_promote_to', self.mp_promote_to)
        self.mp_avg_bits         = config.get('mp_avg_bits', self.mp_avg_bits)

        for name, layer_cfg in tqdm(config['layers'].items(), desc="Loading weights"):
            module = self._get_module(self.pipe.transformer, name)
            if module is None:
                print(f"Warning: {name} not found in model")
                continue
            if not isinstance(module, nn.Linear):
                print(f"Warning: {name} is not nn.Linear — skipping")
                continue

            g    = layer_cfg['group_size']
            fmt  = WeightFormat(layer_cfg['weight_format'])
            clip = layer_cfg.get('clip_ratio', 1.0)
            snr  = layer_cfg.get('best_snr_db', 0.0)
            if layer_cfg.get('promoted', False):
                self.mp_promoted.add(name)

            self.layer_search_results[name] = (g, fmt, clip, snr)

            ql = ChameleonDiTQuantizedLinear(
                original_layer=module,
                group_size=g,
                weight_format=fmt,
                num_buckets=num_buckets,
                act_group_size=act_group_size,
                act_mode=act_mode,
                clip_ratio=clip,
            )
            self._replace_module(self.pipe.transformer, name, ql)
            self.quant_layers.append(ql)
            self.layer_names.append(name)

        self.weight_quantization_complete = True
        self._install_timestep_hook()

        # Restore activation LUT if present
        if 'activation_lut' in config:
            print("Restoring activation format LUT...")
            for name, layer in zip(self.layer_names, self.quant_layers):
                if name not in config['activation_lut']:
                    continue
                lut_entry = config['activation_lut'][name]
                fmt_lut   = lut_entry['format_lut']
                for b, fcode in enumerate(fmt_lut[:num_buckets]):
                    layer.fake_quant.calibrate_bucket(b, fcode)
                layer.fake_quant.enabled = True
            self.activation_calibration_complete = True
            print("Activation format LUT restored.")

        print(f"\nLoaded {len(self.quant_layers)} layers")
        if 'format_distribution' in config:
            print("Weight format distribution:")
            for fmt_str, count in config['format_distribution'].items():
                print(f"  {fmt_str}: {count}")
        if 'group_distribution' in config:
            print("Group size distribution:")
            for g_str, count in sorted(config['group_distribution'].items(), key=lambda x: int(x[0])):
                print(f"  g={g_str}: {count}")

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a serialisable statistics dict."""
        fmt_dist: Dict[str, int] = {}
        g_dist:   Dict[str, int] = {}
        snr_values = []
        for _name, (g, f, _clip, snr) in self.layer_search_results.items():
            fmt_dist[f.value] = fmt_dist.get(f.value, 0) + 1
            g_dist[str(g)]    = g_dist.get(str(g), 0) + 1
            if snr != float('inf') and snr > -float('inf'):
                snr_values.append(snr)

        return {
            'model_id':                    self.model_id,
            'num_buckets':                 self.num_buckets,
            'weight_bits':                 self.weight_bits,
            'act_group_size':              self.act_group_size,
            'act_mode':                    self.act_mode,
            'input_aware_weights':         self.input_aware_weights,
            'clip_search':                 self.clip_search,
            'mixed_precision':             self.mixed_precision,
            'mp_target_bits':              self.mp_target_bits,
            'mp_promote_to':               self.mp_promote_to,
            'mp_avg_bits':                 self.mp_avg_bits,
            'mp_num_promoted':             len(self.mp_promoted),
            'num_quant_layers':            len(self.quant_layers),
            'weight_format_distribution':  fmt_dist,
            'group_size_distribution':     g_dist,
            'mean_weight_snr_db':          (sum(snr_values) / len(snr_values)) if snr_values else 0.0,
            'activation_calibrated':       self.activation_calibration_complete,
        }

    def cleanup(self):
        """Free GPU memory."""
        if self.pipe is not None:
            self._remove_timestep_hook()
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _get_module(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        parts = name.split('.')
        try:
            m = model
            for part in parts:
                m = getattr(m, part)
            return m
        except AttributeError:
            return None

    def _get_calibration_prompts(self, n: int) -> List[str]:
        base = [
            "A photo of a cat sitting on a windowsill",
            "A beautiful sunset over the ocean",
            "A futuristic city skyline at night with neon lights",
            "A serene mountain landscape with snow-capped peaks",
            "An astronaut riding a horse on Mars",
            "A cozy coffee shop interior with warm lighting",
            "A colorful butterfly resting on a flower",
            "A majestic lion surveying the African savanna",
            "A Japanese garden with cherry blossoms in full bloom",
            "A steampunk mechanical robot in a workshop",
            "A tropical beach with crystal clear turquoise water",
            "A medieval castle perched on a cliff above the sea",
            "A portrait of a wise wizard with a long silver beard",
            "A bowl of fresh tropical fruits on a wooden table",
            "A golden retriever running on a sunny beach",
        ]
        prompts = []
        while len(prompts) < n:
            prompts.extend(base)
        return prompts[:n]

    def _print_weight_summary(self):
        fmt_dist: Dict[str, int] = {}
        g_dist:   Dict[str, int] = {}
        snr_values = []
        for _, (g, f, _clip, snr) in self.layer_search_results.items():
            fmt_dist[f.value] = fmt_dist.get(f.value, 0) + 1
            g_dist[str(g)]    = g_dist.get(str(g), 0) + 1
            if snr != float('inf') and snr > -float('inf'):
                snr_values.append(snr)

        print(f"\n{'='*60}")
        print("CHAMELEON-DiT WEIGHT QUANTISATION COMPLETE")
        print(f"{'='*60}")
        print(f"Quantised layers    : {len(self.quant_layers)}")
        if snr_values:
            print(f"Mean SNR            : {sum(snr_values)/len(snr_values):.2f} dB")
        print(f"Weight format dist.:")
        for fmt_str, count in sorted(fmt_dist.items()):
            print(f"  {fmt_str}: {count}")
        print(f"Group size dist.:")
        for g_str, count in sorted(g_dist.items(), key=lambda x: int(x[0])):
            print(f"  g={g_str}: {count}")
        print(f"{'='*60}\n")

    def _print_activation_summary(self):
        fmt_dist: Dict[str, int] = {}
        for layer in self.quant_layers:
            for b in range(self.num_buckets):
                fcode = int(layer.fake_quant.format_lut[b].item())
                fname = FORMAT_NAMES.get(fcode, str(fcode))
                fmt_dist[fname] = fmt_dist.get(fname, 0) + 1

        print(f"\n{'='*60}")
        print("ACTIVATION CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Layers calibrated   : {len(self.quant_layers)}")
        if self.act_mode == ACT_MODE_PERTENSOR:
            print(f"Micro-scaling       : per-tensor (canonical), mode={self.act_mode}")
        else:
            print(f"Micro-scaling       : group-wise ablation, group={self.act_group_size}, "
                  f"mode={self.act_mode}")
        print(f"Macro format dist.  :")
        for fname, count in sorted(fmt_dist.items()):
            print(f"  {fname}: {count} bucket-slots")
        print(f"{'='*60}\n")


# ── Factory function ───────────────────────────────────────────────────────────

def create_chameleon_dit_quantizer(
    model_id:    str              = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device:      str              = "cuda",
    dtype:       torch.dtype      = torch.float16,
    group_sizes: Optional[List[int]]         = None,
    formats:     Optional[List[WeightFormat]] = None,
    num_buckets: int              = 10,
    weight_bits: int              = 4,
    act_group_size: int           = DEFAULT_ACT_GROUP_SIZE,
    act_mode:    str              = DEFAULT_ACT_MODE,
    input_aware_weights: bool     = False,
    input_aware_samples: int      = 16,
    clip_search: bool             = False,
    mixed_precision: bool         = False,
    mp_target_bits: float         = 4.5,
    mp_promote_to: int            = 8,
) -> ChameleonDiTQuantizer:
    """Create a ChameleonDiTQuantizer with the specified configuration."""
    return ChameleonDiTQuantizer(
        model_id=model_id,
        device=device,
        dtype=dtype,
        group_sizes=group_sizes,
        formats=formats,
        num_buckets=num_buckets,
        weight_bits=weight_bits,
        act_group_size=act_group_size,
        act_mode=act_mode,
        input_aware_weights=input_aware_weights,
        input_aware_samples=input_aware_samples,
        clip_search=clip_search,
        mixed_precision=mixed_precision,
        mp_target_bits=mp_target_bits,
        mp_promote_to=mp_promote_to,
    )
