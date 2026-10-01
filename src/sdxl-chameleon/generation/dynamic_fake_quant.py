"""
Dynamic Fake Quantization with Timestep-Bucket LUT

Implements Algorithm 1 from the Chameleon Dynamic Format Adapter.
Supports per-bucket format routing based on kurtosis and diffusion SNR.

Format codes stored in the LUT:
  0 = INT8_ASYM
  1 = FP8_E4M3
  2 = FP8_E5M2
  3 = MXFP8_E4M3
  4 = MXINT8
"""

import torch
import torch.nn as nn

# ── Format codes ───────────────────────────────────────────────────────────────
FORMAT_INT8_ASYM   = 0
FORMAT_FP8_E4M3    = 1
FORMAT_FP8_E5M2    = 2
FORMAT_MXFP8_E4M3  = 3
FORMAT_MXINT8      = 4

FORMAT_NAMES = {
    FORMAT_INT8_ASYM:  "INT8_ASYM",
    FORMAT_FP8_E4M3:   "FP8_E4M3",
    FORMAT_FP8_E5M2:   "FP8_E5M2",
    FORMAT_MXFP8_E4M3: "MXFP8_E4M3",
    FORMAT_MXINT8:     "MXINT8",
}

# FP8 max representable values
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


# ── Low-level FP8 helpers ──────────────────────────────────────────────────────

def _fp8_e4m3_fake_quant(tensor: torch.Tensor) -> torch.Tensor:
    """Round-trip through FP8 E4M3 (or a software simulation).

    Clamps to [-448, 448] before the native cast.  Without this, values
    above 448 map to NaN bit-patterns in E4M3 (it has no ±Inf), which
    propagate through norm layers and produce black images.
    """
    if hasattr(torch, 'float8_e4m3fn'):
        orig = tensor.dtype
        return tensor.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(orig)
    # Software fallback: 1-sign, 4-exp, 3-mantissa
    max_val = FP8_E4M3_MAX
    clamped = tensor.clamp(-max_val, max_val)
    sign     = torch.sign(clamped)
    abs_val  = clamped.abs()
    min_sub  = 2.0 ** -9
    mask_z   = abs_val < min_sub
    safe     = abs_val.clamp(min=min_sub)
    exp      = torch.floor(torch.log2(safe))
    mantissa = safe / (2.0 ** exp)
    q_m      = torch.round(mantissa * 8.0) / 8.0
    result   = sign * q_m * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


def _fp8_e5m2_fake_quant(tensor: torch.Tensor) -> torch.Tensor:
    """Round-trip through FP8 E5M2 (or a software simulation).

    Clamps to [-57344, 57344] before the native cast.  Without this, overflow
    produces ±Inf in E5M2, which then becomes NaN through (Inf - Inf) in norm
    layers and produces black images.
    """
    if hasattr(torch, 'float8_e5m2'):
        orig = tensor.dtype
        return tensor.clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX).to(torch.float8_e5m2).to(orig)
    # Software fallback: 1-sign, 5-exp, 2-mantissa
    max_val = FP8_E5M2_MAX
    clamped = tensor.clamp(-max_val, max_val)
    sign     = torch.sign(clamped)
    abs_val  = clamped.abs()
    min_sub  = 2.0 ** -16
    mask_z   = abs_val < min_sub
    safe     = abs_val.clamp(min=min_sub)
    exp      = torch.floor(torch.log2(safe))
    mantissa = safe / (2.0 ** exp)
    q_m      = torch.round(mantissa * 4.0) / 4.0
    result   = sign * q_m * (2.0 ** exp)
    result[mask_z] = 0.0
    return result


# ── Main module ────────────────────────────────────────────────────────────────

class DynamicTimestepFakeQuantize(nn.Module):
    """
    Per-bucket fake quantization module with a Look-Up Table (LUT).

    During calibration (enabled=False): transparent pass-through.
    At inference (enabled=True): routes each activation to the pre-computed
    format for the current diffusion timestep bucket.

    Buffers (AOT-serializable):
        format_lut  [num_buckets]  int32   – format code per bucket
        scale_lut   [num_buckets]  float32 – scale per bucket
        zp_lut      [num_buckets]  float32 – zero-point per bucket

    Bucket convention:
        bucket_idx = t // bucket_size
        Bucket 0  → t ≈ 0    (cleanest signal)
        Bucket B-1 → t ≈ T-1 (pure noise)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_buckets:   int = 10,
        mx_block_size: int = 32,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_buckets   = num_buckets
        self.mx_block_size = mx_block_size
        self.bucket_size   = num_timesteps // num_buckets

        # Default LUT: FP8_E4M3 (code=1), scale=1.0, zp=0.0
        self.register_buffer(
            'format_lut',
            torch.ones(num_buckets, dtype=torch.int32) * FORMAT_FP8_E4M3,
        )
        self.register_buffer(
            'scale_lut',
            torch.ones(num_buckets, dtype=torch.float32),
        )
        self.register_buffer(
            'zp_lut',
            torch.zeros(num_buckets, dtype=torch.float32),
        )

        self.enabled: bool = False  # toggled to True after calibration

    def resize_buckets(self, num_buckets: int):
        """Re-shape the LUT buffers to `num_buckets`, resetting them to defaults.

        Needed when a loaded activation LUT has a different bucket count than the
        one baked into a weights file. The bucket-count ablation loads shared
        10-bucket W4A8 weights but a B-bucket LUT; without this, every layer stays
        pinned to 10 buckets (so B=1/5 silently mis-bucket and B=20 indexes out of
        bounds). Call this before calibrate_bucket when loading a B-bucket LUT."""
        if num_buckets == self.num_buckets:
            return
        self.num_buckets = num_buckets
        self.bucket_size = self.num_timesteps // num_buckets
        dev = self.format_lut.device
        self.format_lut = torch.ones(num_buckets, dtype=torch.int32, device=dev) * FORMAT_FP8_E4M3
        self.scale_lut  = torch.ones(num_buckets, dtype=torch.float32, device=dev)
        self.zp_lut     = torch.zeros(num_buckets, dtype=torch.float32, device=dev)

    # ── Calibration API ────────────────────────────────────────────────────────

    def calibrate_bucket(
        self,
        bucket_idx:  int,
        format_code: int,
        scale:       float,
        zero_point:  float,
    ):
        """Set a single LUT entry. Called offline during the AOT calibration pass."""
        self.format_lut[bucket_idx] = format_code
        self.scale_lut[bucket_idx]  = scale
        self.zp_lut[bucket_idx]     = zero_point

    # ── Inference ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        """Apply format fake-quant for timestep t."""
        if not self.enabled:
            return x

        t_val      = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        bucket_idx = min(t_val // self.bucket_size, self.num_buckets - 1)

        fmt_code = int(self.format_lut[bucket_idx].item())
        scale    = float(self.scale_lut[bucket_idx].item())
        zp       = float(self.zp_lut[bucket_idx].item())

        return self._apply_format(x, fmt_code, scale, zp)

    # ── Format dispatch ────────────────────────────────────────────────────────

    def _apply_format(
        self,
        x:        torch.Tensor,
        fmt_code: int,
        scale:    float,
        zp:       float,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()

        if fmt_code == FORMAT_INT8_ASYM:
            s = max(scale, 1e-8)
            q = torch.clamp(torch.round(x_f / s + zp), 0.0, 255.0)
            result = (q - zp) * s

        elif fmt_code == FORMAT_FP8_E4M3:
            s          = max(scale, 1e-8)
            normalized = x_f / s
            q          = _fp8_e4m3_fake_quant(normalized)
            result     = q * s

        elif fmt_code == FORMAT_FP8_E5M2:
            s          = max(scale, 1e-8)
            normalized = x_f / s
            q          = _fp8_e5m2_fake_quant(normalized)
            result     = q * s

        elif fmt_code == FORMAT_MXFP8_E4M3:
            result = self._mx_block_quantize(x_f, use_fp8=True)

        elif fmt_code == FORMAT_MXINT8:
            result = self._mx_block_quantize(x_f, use_fp8=False)

        else:
            result = x_f

        return result.to(orig_dtype)

    # ── MX block quantizer (§3.3) ──────────────────────────────────────────────

    def _mx_block_quantize(self, x: torch.Tensor, use_fp8: bool) -> torch.Tensor:
        """
        MX block quantization using shared exponent Eblock = floor(log2(max_i |x_i|)).

        MXFP8_E4M3: normalize by 2^Eblock → fp8_e4m3fn → scale back.
        MXINT8:     normalize by 2^Eblock → INT8 sym (scale=2/127) → scale back.
        """
        original_shape = x.shape
        flat = x.flatten()
        n    = flat.numel()

        # Pad to a multiple of block_size
        pad = (self.mx_block_size - n % self.mx_block_size) % self.mx_block_size
        if pad > 0:
            flat = torch.cat([flat, flat.new_zeros(pad)])

        blocks = flat.view(-1, self.mx_block_size)  # [num_blocks, block_size]

        # Per-block shared exponent
        max_abs     = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-10)
        eblock      = torch.floor(torch.log2(max_abs))          # [num_blocks, 1]
        block_scale = torch.pow(2.0, eblock)                    # [num_blocks, 1]

        # Normalize: values land in (-2, 2)
        normalized = blocks / block_scale

        if use_fp8:
            # MXFP8_E4M3: FP8 handles the full range, no further clipping needed
            q = _fp8_e4m3_fake_quant(normalized)
        else:
            # MXINT8: INT8 symmetric with scale = 2/127
            int8_scale = 2.0 / 127.0
            q_int = torch.clamp(torch.round(normalized / int8_scale), -128.0, 127.0)
            q     = q_int * int8_scale

        dequant = q * block_scale
        return dequant.flatten()[:n].view(original_shape)

    # ── Diagnostic ────────────────────────────────────────────────────────────

    def lut_summary(self, layer_name: str = ""):
        """Print the per-bucket format table."""
        prefix = f"[{layer_name}] " if layer_name else ""
        print(f"\n{prefix}Activation LUT:")
        print(f"  {'Bucket':>6} | {'t_range':>9} | {'Format':<12} | {'Scale':>10} | {'ZP':>8}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
        for b in range(self.num_buckets):
            t_lo     = b * self.bucket_size
            t_hi     = (b + 1) * self.bucket_size - 1
            fmt_code = int(self.format_lut[b].item())
            fmt_name = FORMAT_NAMES.get(fmt_code, f"UNKNOWN({fmt_code})")
            scale    = float(self.scale_lut[b].item())
            zp       = float(self.zp_lut[b].item())
            print(f"  {b:6d} | {t_lo:4d}-{t_hi:4d}  | {fmt_name:<12} | {scale:10.6f} | {zp:8.3f}")
