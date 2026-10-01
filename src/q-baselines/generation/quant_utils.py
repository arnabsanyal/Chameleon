"""
Quantization Utilities for Diffusion Models
Shared utilities for Q-Diffusion and PTQ4DM implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from enum import Enum


class QuantizationScheme(Enum):
    """Supported quantization schemes"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer with fake quantization support.
    Supports both weight and activation quantization.

    For 4-bit weights, uses per-channel quantization (separate scale per output channel)
    which is critical for maintaining quality.
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        weight_bits: int = 8,
        act_bits: int = 8,
        weight_quant: bool = True,
        act_quant: bool = True,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    ):
        super().__init__()
        self.layer = original_layer
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.scheme = scheme

        # Use per-channel quantization for 4-bit weights (critical for quality)
        self.per_channel_weights = (weight_bits <= 4)

        # Quantization range
        self.w_qmin = -(2 ** (weight_bits - 1))
        self.w_qmax = 2 ** (weight_bits - 1) - 1
        self.a_qmin = -(2 ** (act_bits - 1))
        self.a_qmax = 2 ** (act_bits - 1) - 1

        # Weight scale/zero-point: per-channel for 4-bit, per-tensor for 8-bit
        out_features = original_layer.out_features
        if self.per_channel_weights:
            self.register_buffer('w_scale', torch.ones(out_features))
            self.register_buffer('w_zero_point', torch.zeros(out_features))
        else:
            self.register_buffer('w_scale', torch.tensor(1.0))
            self.register_buffer('w_zero_point', torch.tensor(0.0))

        # Activation scale/zero-point (always per-tensor)
        self.register_buffer('a_scale', torch.tensor(1.0))
        self.register_buffer('a_zero_point', torch.tensor(0.0))

        # Running statistics for activation calibration
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))

        # Mode flags
        self.calibrating = False
        self.quantized = False

    def __getattr__(self, name: str):
        """Forward attribute access to underlying layer (e.g., in_features, out_features)"""
        # First, let nn.Module handle its own attributes (_modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # If not found by nn.Module, forward to the underlying layer
        try:
            _modules = object.__getattribute__(self, '_modules')
            layer = _modules.get('layer')
            if layer is not None:
                return getattr(layer, name)
        except (KeyError, AttributeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _compute_scale_zp(self, min_val: torch.Tensor, max_val: torch.Tensor,
                          qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero point from min/max values"""
        if self.scheme == QuantizationScheme.SYMMETRIC:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / ((qmax - qmin) / 2)
            zero_point = torch.zeros_like(scale)
        else:  # ASYMMETRIC
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        scale = torch.clamp(scale, min=1e-8)
        return scale, zero_point

    def _fake_quantize(self, x: torch.Tensor, scale: torch.Tensor,
                       zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Apply fake quantization (quantize then dequantize)"""
        # Ensure scale is on same device and dtype as x
        scale = scale.to(x.device, x.dtype)
        zero_point = zero_point.to(x.device, x.dtype)
        # Clamp scale to avoid division by zero
        scale = torch.clamp(scale, min=1e-8)
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    def calibrate_weights(self):
        """Calibrate weight quantization parameters"""
        w = self.layer.weight.data  # Shape: [out_features, in_features]

        if self.per_channel_weights:
            # Per-channel quantization: compute scale per output channel
            # This is critical for 4-bit quantization quality
            w_min = w.min(dim=1)[0]  # Min per output channel
            w_max = w.max(dim=1)[0]  # Max per output channel
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w_min, w_max, self.w_qmin, self.w_qmax
            )
        else:
            # Per-tensor quantization (fine for 8-bit)
            w_min = w.min()
            w_max = w.max()
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w_min, w_max, self.w_qmin, self.w_qmax
            )

    def update_activation_stats(self, x: torch.Tensor):
        """Update running statistics for activation calibration"""
        with torch.no_grad():
            self.running_min = torch.min(self.running_min, x.min())
            self.running_max = torch.max(self.running_max, x.max())

    def calibrate_activations(self):
        """Calibrate activation quantization parameters from running stats"""
        self.a_scale, self.a_zero_point = self._compute_scale_zp(
            self.running_min, self.running_max, self.a_qmin, self.a_qmax
        )

    def _fake_quantize_per_channel(self, w: torch.Tensor, scale: torch.Tensor,
                                    zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Apply per-channel fake quantization to weights"""
        # scale shape: [out_features], w shape: [out_features, in_features]
        scale = scale.to(w.device, w.dtype)
        zero_point = zero_point.to(w.device, w.dtype)
        scale = torch.clamp(scale, min=1e-8)
        # Reshape scale for broadcasting: [out_features, 1]
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        w_q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
        w_dq = (w_q - zero_point) * scale
        return w_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calibration mode: collect statistics
        if self.calibrating and self.act_quant:
            self.update_activation_stats(x)

        # Quantized inference mode
        if self.quantized:
            # Quantize activations (always per-tensor)
            if self.act_quant:
                x = self._fake_quantize(x, self.a_scale, self.a_zero_point,
                                       self.a_qmin, self.a_qmax)

            # Quantize weights and compute output
            if self.weight_quant:
                if self.per_channel_weights:
                    w_q = self._fake_quantize_per_channel(
                        self.layer.weight, self.w_scale, self.w_zero_point,
                        self.w_qmin, self.w_qmax
                    )
                else:
                    w_q = self._fake_quantize(self.layer.weight, self.w_scale,
                                             self.w_zero_point, self.w_qmin, self.w_qmax)
                return F.linear(x, w_q, self.layer.bias)
            else:
                # Only activation quantization, use original weights
                return self.layer(x)

        # Default: standard forward pass (calibration or no quantization)
        return self.layer(x)


class QuantizedConv2d(nn.Module):
    """
    Quantized Conv2d layer with fake quantization support.

    For 4-bit weights, uses per-channel quantization (separate scale per output channel)
    which is critical for maintaining quality.
    """
    def __init__(
        self,
        original_layer: nn.Conv2d,
        weight_bits: int = 8,
        act_bits: int = 8,
        weight_quant: bool = True,
        act_quant: bool = True,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    ):
        super().__init__()
        self.layer = original_layer
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.scheme = scheme

        # Use per-channel quantization for 4-bit weights (critical for quality)
        self.per_channel_weights = (weight_bits <= 4)

        # Quantization range
        self.w_qmin = -(2 ** (weight_bits - 1))
        self.w_qmax = 2 ** (weight_bits - 1) - 1
        self.a_qmin = -(2 ** (act_bits - 1))
        self.a_qmax = 2 ** (act_bits - 1) - 1

        # Weight scale/zero-point: per-channel for 4-bit, per-tensor for 8-bit
        out_channels = original_layer.out_channels
        if self.per_channel_weights:
            self.register_buffer('w_scale', torch.ones(out_channels))
            self.register_buffer('w_zero_point', torch.zeros(out_channels))
        else:
            self.register_buffer('w_scale', torch.tensor(1.0))
            self.register_buffer('w_zero_point', torch.tensor(0.0))

        # Activation scale/zero-point (always per-tensor)
        self.register_buffer('a_scale', torch.tensor(1.0))
        self.register_buffer('a_zero_point', torch.tensor(0.0))

        # Running statistics
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))

        # Mode flags
        self.calibrating = False
        self.quantized = False

    def __getattr__(self, name: str):
        """Forward attribute access to underlying layer (e.g., in_channels, out_channels)"""
        # First, let nn.Module handle its own attributes (_modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # If not found by nn.Module, forward to the underlying layer
        try:
            _modules = object.__getattribute__(self, '_modules')
            layer = _modules.get('layer')
            if layer is not None:
                return getattr(layer, name)
        except (KeyError, AttributeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _compute_scale_zp(self, min_val: torch.Tensor, max_val: torch.Tensor,
                          qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero point from min/max values"""
        if self.scheme == QuantizationScheme.SYMMETRIC:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / ((qmax - qmin) / 2)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        scale = torch.clamp(scale, min=1e-8)
        return scale, zero_point

    def _fake_quantize(self, x: torch.Tensor, scale: torch.Tensor,
                       zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Apply fake quantization"""
        # Ensure scale is on same device and dtype as x
        scale = scale.to(x.device, x.dtype)
        zero_point = zero_point.to(x.device, x.dtype)
        scale = torch.clamp(scale, min=1e-8)
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    def calibrate_weights(self):
        """Calibrate weight quantization parameters"""
        w = self.layer.weight.data  # Shape: [out_channels, in_channels, kH, kW]

        if self.per_channel_weights:
            # Per-channel quantization: compute scale per output channel
            # Flatten spatial and input channel dims, compute min/max per output channel
            w_flat = w.view(w.size(0), -1)  # [out_channels, in_channels * kH * kW]
            w_min = w_flat.min(dim=1)[0]
            w_max = w_flat.max(dim=1)[0]
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w_min, w_max, self.w_qmin, self.w_qmax
            )
        else:
            # Per-tensor quantization
            w_min = w.min()
            w_max = w.max()
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w_min, w_max, self.w_qmin, self.w_qmax
            )

    def update_activation_stats(self, x: torch.Tensor):
        """Update running statistics"""
        with torch.no_grad():
            self.running_min = torch.min(self.running_min, x.min())
            self.running_max = torch.max(self.running_max, x.max())

    def calibrate_activations(self):
        """Calibrate activation quantization parameters"""
        self.a_scale, self.a_zero_point = self._compute_scale_zp(
            self.running_min, self.running_max, self.a_qmin, self.a_qmax
        )

    def _fake_quantize_per_channel_conv(self, w: torch.Tensor, scale: torch.Tensor,
                                         zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Apply per-channel fake quantization to conv weights"""
        # scale shape: [out_channels], w shape: [out_channels, in_channels, kH, kW]
        scale = scale.to(w.device, w.dtype)
        zero_point = zero_point.to(w.device, w.dtype)
        scale = torch.clamp(scale, min=1e-8)
        # Reshape scale for broadcasting: [out_channels, 1, 1, 1]
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
        w_q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
        w_dq = (w_q - zero_point) * scale
        return w_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating and self.act_quant:
            self.update_activation_stats(x)

        if self.quantized:
            if self.act_quant:
                x = self._fake_quantize(x, self.a_scale, self.a_zero_point,
                                       self.a_qmin, self.a_qmax)

            if self.weight_quant:
                if self.per_channel_weights:
                    w_q = self._fake_quantize_per_channel_conv(
                        self.layer.weight, self.w_scale, self.w_zero_point,
                        self.w_qmin, self.w_qmax
                    )
                else:
                    w_q = self._fake_quantize(self.layer.weight, self.w_scale,
                                             self.w_zero_point, self.w_qmin, self.w_qmax)
                return F.conv2d(x, w_q, self.layer.bias, self.layer.stride,
                               self.layer.padding, self.layer.dilation, self.layer.groups)
            else:
                return self.layer(x)

        return self.layer(x)


class TimestepAwareQuantizedLayer(nn.Module):
    """
    Timestep-aware quantized layer for diffusion models.
    Maintains separate calibration statistics per timestep bucket.
    Used by PTQ4DM for handling timestep-dependent activation distributions.

    For 4-bit weights, uses per-channel quantization (separate scale per output channel)
    which is critical for maintaining quality.
    """
    def __init__(
        self,
        original_layer: Union[nn.Linear, nn.Conv2d],
        weight_bits: int = 8,
        act_bits: int = 8,
        num_timestep_buckets: int = 10,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    ):
        super().__init__()
        self.layer = original_layer
        self.is_conv = isinstance(original_layer, nn.Conv2d)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.num_buckets = num_timestep_buckets
        self.scheme = scheme

        # Use per-channel quantization for 4-bit weights (critical for quality)
        self.per_channel_weights = (weight_bits <= 4)

        # Quantization ranges
        self.w_qmin = -(2 ** (weight_bits - 1))
        self.w_qmax = 2 ** (weight_bits - 1) - 1
        self.a_qmin = -(2 ** (act_bits - 1))
        self.a_qmax = 2 ** (act_bits - 1) - 1

        # Weight quantization params (timestep-independent)
        # Per-channel for 4-bit, per-tensor for 8-bit
        if self.is_conv:
            out_features = original_layer.out_channels
        else:
            out_features = original_layer.out_features

        if self.per_channel_weights:
            self.register_buffer('w_scale', torch.ones(out_features))
            self.register_buffer('w_zero_point', torch.zeros(out_features))
        else:
            self.register_buffer('w_scale', torch.tensor(1.0))
            self.register_buffer('w_zero_point', torch.tensor(0.0))

        # Per-bucket activation stats
        self.register_buffer('bucket_mins', torch.full((num_timestep_buckets,), float('inf')))
        self.register_buffer('bucket_maxs', torch.full((num_timestep_buckets,), float('-inf')))
        self.register_buffer('a_scales', torch.ones(num_timestep_buckets))
        self.register_buffer('a_zero_points', torch.zeros(num_timestep_buckets))

        self.calibrating = False
        self.quantized = False
        self.current_bucket = 0

    def __getattr__(self, name: str):
        """Forward attribute access to underlying layer (e.g., in_features, out_features)"""
        # First, let nn.Module handle its own attributes (_modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # If not found by nn.Module, forward to the underlying layer
        try:
            _modules = object.__getattribute__(self, '_modules')
            layer = _modules.get('layer')
            if layer is not None:
                return getattr(layer, name)
        except (KeyError, AttributeError):
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_timestep_bucket(self, timestep: int, max_timestep: int = 1000) -> int:
        """Map timestep to bucket index"""
        bucket = int(timestep * self.num_buckets / max_timestep)
        return min(bucket, self.num_buckets - 1)

    def set_current_timestep(self, timestep: int, max_timestep: int = 1000):
        """Set the current timestep for bucket selection"""
        self.current_bucket = self.get_timestep_bucket(timestep, max_timestep)

    def _compute_scale_zp(self, min_val, max_val, qmin, qmax):
        if self.scheme == QuantizationScheme.SYMMETRIC:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / ((qmax - qmin) / 2)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale
        scale = torch.clamp(scale, min=1e-8)
        return scale, zero_point

    def _fake_quantize(self, x, scale, zero_point, qmin, qmax):
        # Ensure scale is on same device and dtype as x
        scale = scale.to(x.device, x.dtype)
        zero_point = zero_point.to(x.device, x.dtype)
        scale = torch.clamp(scale, min=1e-8)
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        return (x_q - zero_point) * scale

    def calibrate_weights(self):
        w = self.layer.weight.data

        if self.per_channel_weights:
            # Per-channel quantization
            if self.is_conv:
                # Conv: [out_channels, in_channels, kH, kW]
                w_flat = w.view(w.size(0), -1)
                w_min = w_flat.min(dim=1)[0]
                w_max = w_flat.max(dim=1)[0]
            else:
                # Linear: [out_features, in_features]
                w_min = w.min(dim=1)[0]
                w_max = w.max(dim=1)[0]
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w_min, w_max, self.w_qmin, self.w_qmax
            )
        else:
            # Per-tensor quantization
            self.w_scale, self.w_zero_point = self._compute_scale_zp(
                w.min(), w.max(), self.w_qmin, self.w_qmax
            )

    def update_activation_stats(self, x: torch.Tensor):
        with torch.no_grad():
            self.bucket_mins[self.current_bucket] = torch.min(
                self.bucket_mins[self.current_bucket], x.min()
            )
            self.bucket_maxs[self.current_bucket] = torch.max(
                self.bucket_maxs[self.current_bucket], x.max()
            )

    def calibrate_activations(self):
        """Calibrate all timestep buckets"""
        for i in range(self.num_buckets):
            if self.bucket_mins[i] < float('inf'):
                scale, zp = self._compute_scale_zp(
                    self.bucket_mins[i], self.bucket_maxs[i],
                    self.a_qmin, self.a_qmax
                )
                self.a_scales[i] = scale
                self.a_zero_points[i] = zp

    def _fake_quantize_per_channel(self, w: torch.Tensor, scale: torch.Tensor,
                                    zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Apply per-channel fake quantization to weights"""
        scale = scale.to(w.device, w.dtype)
        zero_point = zero_point.to(w.device, w.dtype)
        scale = torch.clamp(scale, min=1e-8)

        if self.is_conv:
            # Conv: reshape to [out_channels, 1, 1, 1]
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        else:
            # Linear: reshape to [out_features, 1]
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)

        w_q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
        return (w_q - zero_point) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.update_activation_stats(x)

        if self.quantized:
            # Use timestep-specific activation quantization
            a_scale = self.a_scales[self.current_bucket]
            a_zp = self.a_zero_points[self.current_bucket]
            x = self._fake_quantize(x, a_scale, a_zp, self.a_qmin, self.a_qmax)

            # Quantize weights (per-channel for 4-bit, per-tensor for 8-bit)
            if self.per_channel_weights:
                w_q = self._fake_quantize_per_channel(
                    self.layer.weight, self.w_scale, self.w_zero_point,
                    self.w_qmin, self.w_qmax
                )
            else:
                w_q = self._fake_quantize(self.layer.weight, self.w_scale,
                                         self.w_zero_point, self.w_qmin, self.w_qmax)

            if self.is_conv:
                return F.conv2d(x, w_q, self.layer.bias, self.layer.stride,
                               self.layer.padding, self.layer.dilation, self.layer.groups)
            else:
                return F.linear(x, w_q, self.layer.bias)

        return self.layer(x)


def replace_layers_with_quantized(
    module: nn.Module,
    weight_bits: int = 8,
    act_bits: int = 8,
    timestep_aware: bool = False,
    num_timestep_buckets: int = 10,
    skip_layers: Optional[List[str]] = None,
    skip_attention_qkv: bool = True,
    skip_attention_out: bool = False,
    skip_first_last_conv: bool = True,
) -> List:
    """
    Recursively replace Linear and Conv2d layers with quantized versions.

    Args:
        module: Root module to modify
        weight_bits: Bit-width for weights
        act_bits: Bit-width for activations
        timestep_aware: Use timestep-aware quantization (PTQ4DM style)
        num_timestep_buckets: Number of timestep buckets for PTQ4DM
        skip_layers: List of layer name patterns to skip (e.g., ['to_q', 'to_k'])
        skip_attention_qkv: If True, skip attention Q/K/V projections (highly recommended)
        skip_attention_out: If True, also skip attention output projection (conservative)
        skip_first_last_conv: If True, skip first and last conv layers

    Returns:
        List of all quantized layers for easy access

    Note on attention layers:
        - to_q, to_k: MUST skip - quantization errors multiply in QK^T
        - to_v: Should skip - affects attention output values
        - to_out: Can quantize - it's after softmax, less sensitive

    Note on FFN layers:
        - ff.net.0.proj, ff.net.2: These ARE quantized (feed-forward networks)
        - proj_in, proj_out: These ARE quantized (not attention projections)
    """
    # Attention Q/K/V patterns - HIGHLY sensitive to quantization
    # These affect the attention scores before softmax
    attention_qkv_patterns = [
        'to_q', 'to_k', 'to_v',  # Attention Q/K/V projections
        'query', 'key', 'value',  # Alternative naming
    ]

    # Attention output projection - less sensitive (after softmax)
    attention_out_patterns = [
        'to_out',  # Attention output projection
    ]

    # FFN layers we explicitly DO NOT skip (these should be quantized):
    # - ff.net (feed-forward networks)
    # - proj_in, proj_out (projection layers in transformers)
    # - ff.net.0.proj, ff.net.2 (GEGLU and output projection in FFN)

    # Patterns for first/last conv layers (often sensitive)
    first_last_patterns = [
        'conv_in',  # Input conv
        'conv_out',  # Output conv
        'conv_norm_out',  # Output normalization conv
    ]

    skip_patterns = set(skip_layers or [])
    if skip_attention_qkv:
        skip_patterns.update(attention_qkv_patterns)
    if skip_attention_out:
        skip_patterns.update(attention_out_patterns)
    if skip_first_last_conv:
        skip_patterns.update(first_last_patterns)

    # Determine if per-channel quantization will be used
    per_channel = (weight_bits <= 4)

    print(f"\nQuantization config:")
    print(f"  Weight bits: {weight_bits}, Activation bits: {act_bits}")
    print(f"  Per-channel weight quantization: {per_channel} {'(critical for 4-bit!)' if per_channel else ''}")
    print(f"  Skip attention Q/K/V: {skip_attention_qkv}")
    print(f"  Skip attention output: {skip_attention_out}")
    print(f"  Skip conv in/out: {skip_first_last_conv}")

    quant_layers = []
    skipped_layers = []
    all_conv_layers = []  # Track all conv layers for first/last detection

    def should_skip(full_name: str) -> bool:
        """Check if layer should be skipped based on name patterns"""
        name_lower = full_name.lower()
        for pattern in skip_patterns:
            if pattern.lower() in name_lower:
                return True
        return False

    def collect_layers(parent_module, prefix=""):
        """First pass: collect all layers and their full names"""
        layers_info = []
        for name, child in parent_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                all_conv_layers.append(full_name)
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                layers_info.append((parent_module, name, child, full_name))
            else:
                layers_info.extend(collect_layers(child, full_name))
        return layers_info

    def replace_fn(parent_module, prefix=""):
        for name, child in list(parent_module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (nn.Linear, nn.Conv2d)):
                # Check if this layer should be skipped
                if should_skip(full_name):
                    skipped_layers.append(full_name)
                    continue

                # Skip first and last conv if enabled
                if skip_first_last_conv and isinstance(child, nn.Conv2d):
                    if all_conv_layers and (full_name == all_conv_layers[0] or full_name == all_conv_layers[-1]):
                        skipped_layers.append(f"{full_name} (first/last conv)")
                        continue

                if timestep_aware:
                    q_layer = TimestepAwareQuantizedLayer(
                        child, weight_bits, act_bits, num_timestep_buckets
                    )
                else:
                    if isinstance(child, nn.Linear):
                        q_layer = QuantizedLinear(child, weight_bits, act_bits)
                    else:
                        q_layer = QuantizedConv2d(child, weight_bits, act_bits)

                setattr(parent_module, name, q_layer)
                quant_layers.append(q_layer)
            else:
                replace_fn(child, full_name)

    # First pass: collect all conv layers for first/last detection
    collect_layers(module)

    # Second pass: replace layers
    replace_fn(module)

    # Report skipped layers
    if skipped_layers:
        print(f"\nSkipped {len(skipped_layers)} sensitive layers (kept in FP16):")
        # Group by type
        attn_skipped = [l for l in skipped_layers if any(p in l.lower() for p in ['to_q', 'to_k', 'to_v', 'to_out', 'query', 'key', 'value'])]
        conv_skipped = [l for l in skipped_layers if 'conv' in l.lower()]
        other_skipped = [l for l in skipped_layers if l not in attn_skipped and l not in conv_skipped]

        print(f"  Attention Q/K/V layers: {len(attn_skipped)}")
        print(f"  Conv in/out layers: {len(conv_skipped)}")
        if other_skipped:
            print(f"  Other layers: {len(other_skipped)}")
            for layer_name in other_skipped[:5]:
                print(f"    - {layer_name}")

        # Show sample of skipped attention layers
        if attn_skipped:
            print(f"  Sample skipped attention layers:")
            for layer_name in attn_skipped[:3]:
                print(f"    - {layer_name}")

    # Report quantized layers
    print(f"\nQuantized {len(quant_layers)} layers (INT8/FP8):")
    # Count FFN layers to verify they're being quantized
    ffn_quantized = [str(l) for l in quant_layers if hasattr(l, 'layer') and 'ff' in str(type(l.layer)).lower()]
    linear_count = sum(1 for l in quant_layers if 'Linear' in type(l).__name__)
    conv_count = sum(1 for l in quant_layers if 'Conv' in type(l).__name__)
    print(f"  Linear layers: {linear_count}")
    print(f"  Conv layers: {conv_count}")

    return quant_layers


def set_calibration_mode(quant_layers: List, calibrating: bool = True):
    """Set calibration mode for all quantized layers"""
    for layer in quant_layers:
        layer.calibrating = calibrating
        layer.quantized = False


def set_quantized_mode(quant_layers: List, quantized: bool = True):
    """Set quantized inference mode for all quantized layers"""
    for layer in quant_layers:
        layer.calibrating = False
        layer.quantized = quantized


def calibrate_all_weights(quant_layers: List):
    """Calibrate weight quantization for all layers"""
    for layer in quant_layers:
        layer.calibrate_weights()


def calibrate_all_activations(quant_layers: List):
    """Calibrate activation quantization for all layers"""
    for layer in quant_layers:
        layer.calibrate_activations()


def set_timestep_for_all(quant_layers: List, timestep: int, max_timestep: int = 1000):
    """Set current timestep for all timestep-aware layers"""
    for layer in quant_layers:
        if hasattr(layer, 'set_current_timestep'):
            layer.set_current_timestep(timestep, max_timestep)


def disable_quantization(quant_layers: List):
    """
    Completely disable quantization for all layers.
    Useful for testing if the pipeline works without quantization.
    """
    for layer in quant_layers:
        layer.calibrating = False
        layer.quantized = False


def enable_quantization(quant_layers: List):
    """Enable quantization for all layers (assumes calibration is complete)"""
    for layer in quant_layers:
        layer.calibrating = False
        layer.quantized = True


def print_quantization_stats(quant_layers: List, max_layers: int = 10):
    """Print statistics about quantized layers for debugging"""
    print(f"\nQuantization Statistics ({len(quant_layers)} layers):")
    for i, layer in enumerate(quant_layers[:max_layers]):
        layer_type = type(layer).__name__

        # Handle weight scale (per-tensor or per-channel)
        if hasattr(layer, 'w_scale'):
            if layer.w_scale.numel() == 1:
                w_scale = f"{layer.w_scale.item():.6f}"
            else:
                # Per-channel: show range
                w_scale = f"[{layer.w_scale.min().item():.4f}, {layer.w_scale.max().item():.4f}]"
        else:
            w_scale = 'N/A'

        # Handle activation scale (per-tensor or per-bucket)
        if hasattr(layer, 'a_scale'):
            if layer.a_scale.numel() == 1:
                a_scale = f"{layer.a_scale.item():.6f}"
            else:
                a_scale = f"[{layer.a_scale.min().item():.4f}, {layer.a_scale.max().item():.4f}]"
        elif hasattr(layer, 'a_scales'):
            a_scale = f"[{layer.a_scales.min().item():.4f}, {layer.a_scales.max().item():.4f}]"
        else:
            a_scale = 'N/A'

        print(f"  Layer {i}: {layer_type}, w_scale={w_scale}, a_scale={a_scale}")
    if len(quant_layers) > max_layers:
        print(f"  ... and {len(quant_layers) - max_layers} more layers")


# ── Auto batch-size helper (shared by qdiffusion + ptq4dm) ────────────────────

def auto_batch_size_for_sdxl(
    pipe,
    num_inference_steps: int,
    guidance_scale: float,
    device: str = "cuda",
    extra_pipe_kwargs: Optional[dict] = None,
    safety: float = 0.60,
    calib_prompt: str = "a cat sitting on a chair",
) -> int:
    """Two-point VRAM calibration → batch size that fits in `safety` × total VRAM.

    Same approach as LCM's _auto_batch_size_for_images: measure peak_mem at
    N=1 and N=2 to separate fixed model overhead from per-image marginal cost,
    then solve `model + N * latent ≤ safety * total` for N.

    Returns 1 on CPU, on calibration error, or whenever the two-point estimate
    can't be trusted.  The caller is still expected to handle OOM at the chosen
    batch by halving and retrying (some configs allocate more during the
    *second* batch than the first due to autograd graph reuse, kernel cache
    growth, etc.).

    ``extra_pipe_kwargs`` is forwarded verbatim to ``pipe(...)`` so callers can
    pass e.g. PTQ4DM's ``callback_on_step_end`` if they want the calibration
    to exercise the same code path as real generation.
    """
    if not torch.cuda.is_available():
        return 1

    extra = extra_pipe_kwargs or {}

    def _measure(n: int) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        pipe(
            prompt=[calib_prompt] * n,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=[torch.Generator(device=device).manual_seed(i) for i in range(n)],
            **extra,
        )
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    print("\nAuto-sizing batch (two-point VRAM calibration)...")
    try:
        total = torch.cuda.mem_get_info()[1]
        peak1 = _measure(1)

        try:
            peak2 = _measure(2)
            latent_per_image = peak2 - peak1
            model_overhead   = 2 * peak1 - peak2

            if latent_per_image <= 0:
                # peak(2) <= peak(1) → cache effect dominating; single-point estimate.
                batch = max(1, int(total * safety / peak1))
            else:
                budget = total * safety - model_overhead
                batch  = max(1, int(budget / latent_per_image))

            print(f"  Fixed overhead  : {model_overhead   // 1024**2} MB")
            print(f"  Per-image cost  : {latent_per_image // 1024**2} MB")
        except Exception as exc2:
            print(f"  Batch-2 calibration failed ({exc2}); single-point fallback")
            batch = max(1, int(total * safety / peak1))

        print(f"  Total VRAM      : {total // 1024**2} MB")
        print(f"  Auto batch size : {batch}  "
              f"(budget {int(total * safety) // 1024**2} MB @ {int(safety*100)}% safety)")
        return batch

    except Exception as exc:
        print(f"  Calibration failed ({exc}); falling back to batch_size=1")
        return 1
