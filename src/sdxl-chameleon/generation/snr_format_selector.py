"""
SNR-based Format Selection for Adaptive Weight Quantization
+
Kurtosis/DiffusionSNR-based Dynamic Activation Routing

Weight quantization (offline, per output-channel):
  Computes SNR for each channel quantized to different 8-bit formats,
  selects the format with the highest SNR.

Activation quantization (runtime, per timestep-bucket):
  Routes activations to a format based on the empirical kurtosis of the
  distribution and the diffusion SNR  SNR(t) = ᾱ_t / (1 - ᾱ_t).

Supported weight formats:
  8-bit (W8A8): int8_symmetric, int8_asymmetric, mxint8
  4-bit (W4A8): int4_symmetric, int4_asymmetric, nf4, fp4_e2m1, mxint4, mxfp4_e2m1

Supported activation formats (all 8-bit, selected at calibration):
  float8_e4m3fn, float8_e5m2, int8_asymmetric, mxfp8_e4m3, mxint8
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field


# ── Quantization format enum ───────────────────────────────────────────────────

class QuantFormat(Enum):
    """Supported quantization formats (8-bit and 4-bit)"""
    FLOAT8_E4M3 = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    INT8_SYM    = "int8_symmetric"
    INT8_ASYM   = "int8_asymmetric"
    MXFP8_E4M3  = "mxfp8_e4m3"
    MXFP8_E5M2  = "mxfp8_e5m2"
    MXINT8      = "mxint8"
    # 4-bit weight formats (W4A8)
    INT4_SYM    = "int4_symmetric"
    INT4_ASYM   = "int4_asymmetric"
    NF4         = "nf4"
    FP4_E2M1    = "fp4_e2m1"
    MXINT4      = "mxint4"
    MXFP4_E2M1  = "mxfp4_e2m1"
    FP16        = "fp16"


# Mapping from QuantFormat to integer codes used by DynamicTimestepFakeQuantize
QUANT_FORMAT_TO_CODE: Dict[QuantFormat, int] = {
    QuantFormat.INT8_ASYM:  0,
    QuantFormat.FLOAT8_E4M3: 1,
    QuantFormat.FLOAT8_E5M2: 2,
    QuantFormat.MXFP8_E4M3: 3,
    QuantFormat.MXINT8:     4,
}

# Weight format groups
WEIGHT_FORMATS_8BIT = [QuantFormat.INT8_SYM, QuantFormat.INT8_ASYM, QuantFormat.MXINT8]
WEIGHT_FORMATS_4BIT = [QuantFormat.INT4_SYM, QuantFormat.INT4_ASYM, QuantFormat.NF4,
                       QuantFormat.FP4_E2M1, QuantFormat.MXINT4, QuantFormat.MXFP4_E2M1]

# NF4 quantile levels (16 values from QLoRA paper, optimal for N(0,1))
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634, 0.33791524171829224,
     0.44070982933044434,  0.5626170039176941,   0.7229568362236023,  1.0,
])

# FP4 E2M1 representable values (1s 2e 1m, bias=1)
FP4_E2M1_LEVELS = torch.tensor([
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
])


# ── Routing constants (§3.2) ───────────────────────────────────────────────────

KURTOSIS_HIGH       = 5.0   # above → FP8 E5M2 (heavy tails / pure noise)
KURTOSIS_LOW        = 3.0   # below → INT8 ASYM when SNR is also high
SNR_DIFFUSION_LOW   = 0.2   # below → FP8 E5M2 (high-noise phase)
SNR_DIFFUSION_HIGH  = 2.0   # above → INT8 ASYM for clean-signal bucket


# ── Channel / layer info dataclasses (weights) ────────────────────────────────

@dataclass
class ChannelFormatInfo:
    """Format selection result for one output channel."""
    channel_idx:     int
    selected_format: QuantFormat
    snr_db:          float
    all_snrs:        Dict[QuantFormat, float] = field(default_factory=dict)


@dataclass
class LayerFormatInfo:
    """Format selection result for an entire layer (per-channel)."""
    layer_name:           str
    channel_formats:      List[ChannelFormatInfo]
    format_distribution:  Dict[QuantFormat, int]


# ── Low-level quantization primitives ─────────────────────────────────────────

def compute_snr_db(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """SNR in dB: 10·log10(E[x²] / E[(x-x̂)²])."""
    signal_power = torch.mean(original ** 2)
    noise_power  = torch.mean((original - quantized) ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return (10 * torch.log10(signal_power / noise_power + 1e-10)).item()


def quantize_to_int8_symmetric(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8.  Returns (dequantized, scale)."""
    max_abs = torch.max(torch.abs(tensor))
    scale   = torch.clamp(max_abs / 127.0, min=1e-8)
    q       = torch.clamp(torch.round(tensor / scale), -128, 127)
    return q * scale, scale


def quantize_to_int8_asymmetric(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric INT8.  Returns (dequantized, scale, zero_point)."""
    t_min = tensor.min()
    t_max = tensor.max()
    scale  = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
    zp     = torch.clamp(torch.round(-t_min / scale), 0, 255)
    q      = torch.clamp(torch.round(tensor / scale + zp), 0, 255)
    return (q - zp) * scale, scale, zp


def quantize_to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 quantization."""
    if hasattr(torch, 'float8_e4m3fn'):
        orig = tensor.dtype
        return tensor.to(torch.float8_e4m3fn).to(orig)
    return _simulate_fp8_e4m3(tensor)


def quantize_to_fp8_e5m2(tensor: torch.Tensor) -> torch.Tensor:
    """FP8 E5M2 quantization."""
    if hasattr(torch, 'float8_e5m2'):
        orig = tensor.dtype
        return tensor.to(torch.float8_e5m2).to(orig)
    return _simulate_fp8_e5m2(tensor)


def _simulate_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Software FP8 E4M3 (1s 4e 3m, max ≈ 448)."""
    max_val = 448.0
    min_sub = 2.0 ** -9
    clamped = tensor.clamp(-max_val, max_val)
    sign    = torch.sign(clamped)
    abs_val = clamped.abs()
    mask_z  = abs_val < min_sub
    safe    = abs_val.clamp(min=min_sub)
    exp     = torch.floor(torch.log2(safe))
    m       = safe / (2.0 ** exp)
    qm      = torch.round(m * 8.0) / 8.0
    result  = sign * qm * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


def _simulate_fp8_e5m2(tensor: torch.Tensor) -> torch.Tensor:
    """Software FP8 E5M2 (1s 5e 2m, max ≈ 57344)."""
    max_val = 57344.0
    min_sub = 2.0 ** -16
    clamped = tensor.clamp(-max_val, max_val)
    sign    = torch.sign(clamped)
    abs_val = clamped.abs()
    mask_z  = abs_val < min_sub
    safe    = abs_val.clamp(min=min_sub)
    exp     = torch.floor(torch.log2(safe))
    m       = safe / (2.0 ** exp)
    qm      = torch.round(m * 4.0) / 4.0
    result  = sign * qm * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


def quantize_to_mxfp8_e4m3(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP8 E4M3 (per-block max scaling + FP8 E4M3)."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    pad  = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    q      = quantize_to_fp8_e4m3(blocks / scales)
    result = (q * scales).flatten()[:tensor.numel()]
    return result.view(original_shape)


def quantize_to_mxfp8_e5m2(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP8 E5M2 (per-block max scaling + FP8 E5M2)."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    pad  = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    q      = quantize_to_fp8_e5m2(blocks / scales)
    result = (q * scales).flatten()[:tensor.numel()]
    return result.view(original_shape)


def quantize_to_mxint8(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling INT8 (per-block max/127 scaling + INT8 sym)."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    pad  = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scales = (blocks.abs().max(dim=1, keepdim=True)[0] / 127.0).clamp(min=1e-8)
    q_int  = torch.clamp(torch.round(blocks / scales), -128, 127)
    result = (q_int * scales).flatten()[:tensor.numel()]
    return result.view(original_shape)


# ── 4-bit quantization primitives ─────────────────────────────────────────────

def quantize_to_int4_symmetric(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4.  Returns (dequantized, scale).  q ∈ [-8, 7]."""
    max_abs = torch.max(torch.abs(tensor))
    scale   = torch.clamp(max_abs / 7.0, min=1e-8)
    q       = torch.clamp(torch.round(tensor / scale), -8, 7)
    return q * scale, scale


def quantize_to_int4_asymmetric(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric INT4.  Returns (dequantized, scale, zero_point).  q ∈ [0, 15]."""
    t_min = tensor.min()
    t_max = tensor.max()
    scale  = torch.clamp((t_max - t_min) / 15.0, min=1e-8)
    zp     = torch.clamp(torch.round(-t_min / scale), 0, 15)
    q      = torch.clamp(torch.round(tensor / scale + zp), 0, 15)
    return (q - zp) * scale, scale, zp


def quantize_to_nf4(tensor: torch.Tensor) -> torch.Tensor:
    """NormalFloat4 quantization (16 quantile-optimal levels for N(0,1)).
    Normalizes by max_abs, snaps to nearest NF4 level, scales back."""
    max_abs = torch.max(torch.abs(tensor)).clamp(min=1e-8)
    normalized = tensor / max_abs  # map into [-1, 1]

    # NF4_LEVELS is sorted; use midpoints as bucket boundaries for nearest-level
    levels = NF4_LEVELS.to(tensor.device, tensor.dtype)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    indices = torch.bucketize(normalized, midpoints)
    quantized_norm = levels[indices]
    return quantized_norm * max_abs


def quantize_to_fp4_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 quantization (1s 2e 1m, bias=1).
    Positive representable: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Scales tensor into [-6, 6], snaps to nearest level, scales back."""
    max_abs = torch.max(torch.abs(tensor)).clamp(min=1e-8)
    scale   = max_abs / 6.0
    normalized = tensor / scale  # map into [-6, 6]

    levels = FP4_E2M1_LEVELS.to(tensor.device, tensor.dtype)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    indices = torch.bucketize(normalized, midpoints)
    quantized_norm = levels[indices]
    return quantized_norm * scale


def quantize_to_mxint4(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling INT4 (per-block max/7 scaling + INT4 sym, q ∈ [-8, 7])."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    pad  = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scales = (blocks.abs().max(dim=1, keepdim=True)[0] / 7.0).clamp(min=1e-8)
    q_int  = torch.clamp(torch.round(blocks / scales), -8, 7)
    result = (q_int * scales).flatten()[:tensor.numel()]
    return result.view(original_shape)


def quantize_to_mxfp4_e2m1(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP4 E2M1 (per-block shared exponent + FP4 E2M1)."""
    original_shape = tensor.shape
    flat = tensor.flatten()
    pad  = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scales = (blocks.abs().max(dim=1, keepdim=True)[0] / 6.0).clamp(min=1e-8)
    normalized = blocks / scales  # map each block into [-6, 6]

    levels = FP4_E2M1_LEVELS.to(tensor.device, tensor.dtype)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    indices = torch.bucketize(normalized, midpoints)
    quantized_norm = levels[indices]

    result = (quantized_norm * scales).flatten()[:tensor.numel()]
    return result.view(original_shape)


# ── SNR computation across formats ────────────────────────────────────────────

def compute_snr_for_all_formats(
    tensor:        torch.Tensor,
    formats:       Optional[List[QuantFormat]] = None,
    mx_block_size: int = 32,
) -> Dict[QuantFormat, float]:
    """
    Compute quantization SNR for a 1-D tensor across all specified formats.

    Default formats are restricted to INT8 / MX-INT8 for weight quantization
    (per §3.1: "aggressively INT8 or MX-INT8").
    """
    if formats is None:
        formats = [
            QuantFormat.INT8_SYM,
            QuantFormat.INT8_ASYM,
            QuantFormat.MXINT8,
        ]

    snrs     = {}
    original = tensor.float()

    for fmt in formats:
        if fmt == QuantFormat.FLOAT8_E4M3:
            q = quantize_to_fp8_e4m3(original)
        elif fmt == QuantFormat.FLOAT8_E5M2:
            q = quantize_to_fp8_e5m2(original)
        elif fmt == QuantFormat.INT8_SYM:
            q, _ = quantize_to_int8_symmetric(original)
        elif fmt == QuantFormat.INT8_ASYM:
            q, _, _ = quantize_to_int8_asymmetric(original)
        elif fmt == QuantFormat.MXFP8_E4M3:
            q = quantize_to_mxfp8_e4m3(original, mx_block_size)
        elif fmt == QuantFormat.MXFP8_E5M2:
            q = quantize_to_mxfp8_e5m2(original, mx_block_size)
        elif fmt == QuantFormat.MXINT8:
            q = quantize_to_mxint8(original, mx_block_size)
        elif fmt == QuantFormat.INT4_SYM:
            q, _ = quantize_to_int4_symmetric(original)
        elif fmt == QuantFormat.INT4_ASYM:
            q, _, _ = quantize_to_int4_asymmetric(original)
        elif fmt == QuantFormat.NF4:
            q = quantize_to_nf4(original)
        elif fmt == QuantFormat.FP4_E2M1:
            q = quantize_to_fp4_e2m1(original)
        elif fmt == QuantFormat.MXINT4:
            q = quantize_to_mxint4(original, mx_block_size)
        elif fmt == QuantFormat.MXFP4_E2M1:
            q = quantize_to_mxfp4_e2m1(original, mx_block_size)
        elif fmt == QuantFormat.FP16:
            q = original
        else:
            raise ValueError(f"Unknown format: {fmt}")
        snrs[fmt] = compute_snr_db(original, q)

    return snrs


def select_best_format_for_channel(
    channel_data:      torch.Tensor,
    formats:           Optional[List[QuantFormat]] = None,
    min_snr_threshold: float = 20.0,
    mx_block_size:     int   = 32,
) -> ChannelFormatInfo:
    """Select the best quantization format for one channel based on SNR."""
    snrs        = compute_snr_for_all_formats(channel_data, formats, mx_block_size)
    best_format = max(snrs, key=snrs.get)
    best_snr    = snrs[best_format]

    if best_snr < min_snr_threshold:
        best_format = QuantFormat.FP16
        snrs[QuantFormat.FP16] = float('inf')

    return ChannelFormatInfo(
        channel_idx=-1,
        selected_format=best_format,
        snr_db=best_snr,
        all_snrs=snrs,
    )


def select_formats_for_layer(
    weight_tensor:     torch.Tensor,
    layer_name:        str,
    is_conv:           bool = False,
    formats:           Optional[List[QuantFormat]] = None,
    min_snr_threshold: float = 20.0,
    mx_block_size:     int   = 32,
    verbose:           bool  = False,
) -> LayerFormatInfo:
    """Select optimal quantization format per output channel of a layer."""
    num_channels  = weight_tensor.size(0)
    flat_weights  = weight_tensor.view(num_channels, -1)

    channel_infos = []
    format_counts: Dict[QuantFormat, int] = {}

    for ch_idx in range(num_channels):
        info            = select_best_format_for_channel(
            flat_weights[ch_idx], formats, min_snr_threshold, mx_block_size
        )
        info.channel_idx = ch_idx
        channel_infos.append(info)
        format_counts[info.selected_format] = format_counts.get(info.selected_format, 0) + 1

    if verbose:
        print(f"\n{layer_name}: format distribution ({num_channels} channels):")
        for fmt, count in format_counts.items():
            print(f"  {fmt.value}: {count} ({100*count/num_channels:.1f}%)")

    return LayerFormatInfo(
        layer_name=layer_name,
        channel_formats=channel_infos,
        format_distribution={k: v for k, v in format_counts.items() if v > 0},
    )


# ── Kurtosis & diffusion SNR helpers (§3.2) ───────────────────────────────────

def compute_kurtosis(tensor: torch.Tensor) -> float:
    """
    Compute excess-free kurtosis  κ = E[(X-μ)⁴] / σ⁴  from a tensor.

    Uses closed-form expansion in terms of raw moments so no sample storage
    is required (O(1) memory if called on a running accumulator's snapshot).
    Returns 3.0 (normal distribution) when σ is negligible.
    """
    x    = tensor.float().flatten()
    n    = x.numel()
    if n < 2:
        return 3.0
    m1   = x.mean().item()
    m2   = (x ** 2).mean().item()
    m3   = (x ** 3).mean().item()
    m4   = (x ** 4).mean().item()

    sigma2 = m2 - m1 ** 2
    if sigma2 < 1e-10:
        return 3.0

    # E[(X-μ)⁴] = E[X⁴] - 4μE[X³] + 6μ²E[X²] - 3μ⁴
    central_m4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    return central_m4 / (sigma2 ** 2)


def compute_diffusion_snr(
    timestep:        int,
    alphas_cumprod:  torch.Tensor,
) -> float:
    """
    Compute diffusion SNR at timestep t:
        SNR(t) = ᾱ_t / (1 - ᾱ_t)

    Args:
        timestep:       Integer in [0, T-1].
        alphas_cumprod: 1-D tensor of length T (from scheduler.alphas_cumprod).

    Returns:
        SNR as a float.  Returns 0.0 if ᾱ_t ≥ 1.
    """
    t      = min(timestep, len(alphas_cumprod) - 1)
    alpha  = alphas_cumprod[t].float().item()
    denom  = 1.0 - alpha
    if denom < 1e-10:
        return 0.0
    return alpha / denom


def route_format_by_kurtosis_snr(
    kurtosis:     float,
    diffusion_snr: float,
    is_shortcut:  bool = False,
) -> QuantFormat:
    """
    3-phase activation format routing (§3.2 + §3.3).

    Phase 1 – Pure Noise:   κ > 5.0  OR  SNR(t) < 0.2  →  FP8 E5M2
    Phase 2 – Transition:   3.0 < κ ≤ 5.0               →  FP8 E4M3
    Phase 3 – Clean Signal: κ ≤ 3.0  AND  SNR(t) > 2.0  →  INT8 ASYM
    Shortcut layers: bypass kurtosis check                →  MXFP8 E4M3

    Returns:
        QuantFormat enum value.
    """
    if is_shortcut:
        return QuantFormat.MXFP8_E4M3

    # Phase 1
    if kurtosis > KURTOSIS_HIGH or diffusion_snr < SNR_DIFFUSION_LOW:
        return QuantFormat.FLOAT8_E5M2

    # Phase 3
    if kurtosis <= KURTOSIS_LOW and diffusion_snr > SNR_DIFFUSION_HIGH:
        return QuantFormat.INT8_ASYM

    # Phase 2 (default / transition)
    return QuantFormat.FLOAT8_E4M3


# ── Per-bucket activation statistics ──────────────────────────────────────────

class PerBucketActivationStats:
    """
    Tracks activation statistics per timestep bucket using running moments.

    Collects n, Σx, Σx², Σx³, Σx⁴, min, max per bucket.
    All O(1) memory — no sample tensors stored.
    """

    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets
        self.n         = [0]              * num_buckets
        self.sum       = [0.0]            * num_buckets
        self.sum_sq    = [0.0]            * num_buckets
        self.sum_cub   = [0.0]            * num_buckets
        self.sum_quad  = [0.0]            * num_buckets
        self.min_val   = [float('inf')]   * num_buckets
        self.max_val   = [float('-inf')]  * num_buckets

    def update(self, activations: torch.Tensor, bucket_idx: int):
        """Accumulate statistics for one forward-pass output."""
        with torch.no_grad():
            flat = activations.float().flatten()
            nn   = flat.numel()
            if nn == 0 or bucket_idx < 0 or bucket_idx >= self.num_buckets:
                return
            b = bucket_idx
            self.n[b]        += nn
            self.sum[b]      += flat.sum().item()
            self.sum_sq[b]   += (flat ** 2).sum().item()
            self.sum_cub[b]  += (flat ** 3).sum().item()
            self.sum_quad[b] += (flat ** 4).sum().item()
            bmin = flat.min().item()
            bmax = flat.max().item()
            if bmin < self.min_val[b]:
                self.min_val[b] = bmin
            if bmax > self.max_val[b]:
                self.max_val[b] = bmax

    def has_data(self, bucket_idx: int) -> bool:
        return self.n[bucket_idx] > 0

    def compute_kurtosis(self, bucket_idx: int) -> float:
        """
        Kurtosis  κ = E[(X-μ)⁴] / σ⁴  computed from running moments.
        Returns 3.0 when σ is negligible or data is insufficient.
        """
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

        central_m4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
        return central_m4 / (sigma2 ** 2)

    def compute_scale_zp(
        self,
        bucket_idx: int,
        fmt:        QuantFormat,
    ) -> Tuple[float, float]:
        """
        Compute (scale, zero_point) for a given format from bucket statistics.

        FP8 E4M3:    scale = max_abs / 448.0
        FP8 E5M2:    scale = max_abs / 57344.0
        INT8 ASYM:   scale = (max-min)/255,  zp = round(-min/scale)
        MX formats:  scale = 1.0  (block-wise Eblock handles scaling)
        """
        b       = bucket_idx
        min_v   = self.min_val[b]
        max_v   = self.max_val[b]
        max_abs = max(abs(min_v), abs(max_v), 1e-8)

        if fmt == QuantFormat.FLOAT8_E4M3:
            return max_abs / 448.0, 0.0

        elif fmt == QuantFormat.FLOAT8_E5M2:
            return max_abs / 57344.0, 0.0

        elif fmt == QuantFormat.INT8_ASYM:
            range_v = max_v - min_v
            scale   = range_v / 255.0 if range_v > 1e-8 else 1.0
            zp      = round(-min_v / max(scale, 1e-8))
            return scale, float(zp)

        else:
            # MXFP8_E4M3, MXINT8 — block-wise Eblock is self-contained
            return 1.0, 0.0


# ── SNRFormatSelector (weights only) ──────────────────────────────────────────

class SNRFormatSelector:
    """
    SNR-based per-channel weight format selector.

    Restricted to INT8 / MX-INT8 formats per §3.1.
    Activation analysis has been removed; use PerBucketActivationStats +
    route_format_by_kurtosis_snr for activations.
    """

    def __init__(
        self,
        formats:           Optional[List[QuantFormat]] = None,
        min_snr_threshold: float = 20.0,
        mx_block_size:     int   = 32,
        weight_bits:       int   = 8,
    ):
        self.weight_bits = weight_bits
        if formats is not None:
            self.formats = formats
        elif weight_bits == 4:
            self.formats = list(WEIGHT_FORMATS_4BIT)
        else:
            self.formats = list(WEIGHT_FORMATS_8BIT)
        self.min_snr_threshold = min_snr_threshold
        self.mx_block_size     = mx_block_size

        self.weight_layer_infos: Dict[str, LayerFormatInfo] = {}

    def analyze_weights(
        self,
        weight_tensor: torch.Tensor,
        layer_name:    str,
        is_conv:       bool = False,
        verbose:       bool = False,
    ) -> LayerFormatInfo:
        """Analyze and select formats for a layer's weight channels."""
        info = select_formats_for_layer(
            weight_tensor=weight_tensor,
            layer_name=layer_name,
            is_conv=is_conv,
            formats=self.formats,
            min_snr_threshold=self.min_snr_threshold,
            mx_block_size=self.mx_block_size,
            verbose=verbose,
        )
        self.weight_layer_infos[layer_name] = info
        return info

    def get_global_weight_format_distribution(self) -> Dict[QuantFormat, int]:
        """Format distribution across all weight channels."""
        dist: Dict[QuantFormat, int] = {}
        for layer_info in self.weight_layer_infos.values():
            for fmt, count in layer_info.format_distribution.items():
                dist[fmt] = dist.get(fmt, 0) + count
        return {k: v for k, v in dist.items() if v > 0}

    def print_summary(self):
        """Print weight format selection summary."""
        print(f"\n{'='*60}")
        print("SNR-BASED WEIGHT FORMAT SELECTION SUMMARY")
        print(f"{'='*60}")

        if self.weight_layer_infos:
            total_ch  = sum(len(i.channel_formats) for i in self.weight_layer_infos.values())
            weight_dist = self.get_global_weight_format_distribution()

            print(f"\nWEIGHTS (per output-channel, SNR-based):")
            print(f"  Layers analyzed    : {len(self.weight_layer_infos)}")
            print(f"  Total out-channels : {total_ch}")
            print(f"  Format distribution:")
            for fmt, count in sorted(weight_dist.items(), key=lambda x: -x[1]):
                pct = 100 * count / total_ch
                print(f"    {fmt.value}: {count} ({pct:.1f}%)")

        print(f"{'='*60}\n")
