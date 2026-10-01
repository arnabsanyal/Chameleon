"""
Chameleon Quantization: Adaptive Mixed-Format Weight and Activation Quantization

Algorithm:
  Weights   — offline per output-channel (SNR-based).
              W8A8: INT8/MXINT8 formats.  W4A8: INT4/NF4/FP4/MX4 formats.
  Activations — dynamic per-bucket routing via kurtosis + diffusion SNR LUT.
  UNet skip-connection layers (up_blocks.*.resnets.*.conv1) — always MXFP8 E4M3.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import os
import json
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402

from snr_format_selector import (
    QuantFormat,
    SNRFormatSelector,
    LayerFormatInfo,
    ChannelFormatInfo,
    PerBucketActivationStats,
    compute_diffusion_snr,
    route_format_by_kurtosis_snr,
    QUANT_FORMAT_TO_CODE,
    WEIGHT_FORMATS_4BIT,
    WEIGHT_FORMATS_8BIT,
    quantize_to_int8_symmetric,
    quantize_to_int8_asymmetric,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
    quantize_to_mxfp8_e4m3,
    quantize_to_mxfp8_e5m2,
    quantize_to_mxint8,
    quantize_to_int4_symmetric,
    quantize_to_int4_asymmetric,
    quantize_to_nf4,
    quantize_to_fp4_e2m1,
    quantize_to_mxint4,
    quantize_to_mxfp4_e2m1,
)
from dynamic_fake_quant import DynamicTimestepFakeQuantize, FORMAT_NAMES


# ── Helper: identify UNet skip-connection (shortcut) layers ───────────────────

def is_unet_shortcut_layer(name: str) -> bool:
    """
    Returns True for up_blocks.*.resnets.*.conv1 layers.
    These receive skip-concatenated inputs and are routed to MX format.
    """
    parts = name.split('.')
    if 'up_blocks' not in parts:
        return False
    if 'resnets' not in parts:
        return False
    return parts[-1] == 'conv1'


# ── Weight-quantization helper ─────────────────────────────────────────────────

def apply_format_quantization(
    tensor:        torch.Tensor,
    fmt:           QuantFormat,
    scale:         Optional[float] = None,
    zero_point:    Optional[float] = None,
    mx_block_size: int = 32,
) -> torch.Tensor:
    """Apply fake quantization for a given QuantFormat."""
    if fmt == QuantFormat.FP16:
        return tensor
    elif fmt == QuantFormat.FLOAT8_E4M3:
        return quantize_to_fp8_e4m3(tensor)
    elif fmt == QuantFormat.FLOAT8_E5M2:
        return quantize_to_fp8_e5m2(tensor)
    elif fmt == QuantFormat.INT8_SYM:
        if scale is not None:
            q = torch.clamp(torch.round(tensor / scale), -128, 127)
            return q * scale
        dq, _ = quantize_to_int8_symmetric(tensor)
        return dq
    elif fmt == QuantFormat.INT8_ASYM:
        if scale is not None and zero_point is not None:
            q = torch.clamp(torch.round(tensor / scale + zero_point), 0, 255)
            return (q - zero_point) * scale
        dq, _, _ = quantize_to_int8_asymmetric(tensor)
        return dq
    elif fmt == QuantFormat.MXFP8_E4M3:
        return quantize_to_mxfp8_e4m3(tensor, mx_block_size)
    elif fmt == QuantFormat.MXFP8_E5M2:
        return quantize_to_mxfp8_e5m2(tensor, mx_block_size)
    elif fmt == QuantFormat.MXINT8:
        return quantize_to_mxint8(tensor, mx_block_size)
    elif fmt == QuantFormat.INT4_SYM:
        if scale is not None:
            q = torch.clamp(torch.round(tensor / scale), -8, 7)
            return q * scale
        dq, _ = quantize_to_int4_symmetric(tensor)
        return dq
    elif fmt == QuantFormat.INT4_ASYM:
        if scale is not None and zero_point is not None:
            q = torch.clamp(torch.round(tensor / scale + zero_point), 0, 15)
            return (q - zero_point) * scale
        dq, _, _ = quantize_to_int4_asymmetric(tensor)
        return dq
    elif fmt == QuantFormat.NF4:
        return quantize_to_nf4(tensor)
    elif fmt == QuantFormat.FP4_E2M1:
        return quantize_to_fp4_e2m1(tensor)
    elif fmt == QuantFormat.MXINT4:
        return quantize_to_mxint4(tensor, mx_block_size)
    elif fmt == QuantFormat.MXFP4_E2M1:
        return quantize_to_mxfp4_e2m1(tensor, mx_block_size)
    raise ValueError(f"Unknown format: {fmt}")


# ── Quantized layer classes ────────────────────────────────────────────────────

class ChameleonQuantizedLinear(nn.Module):
    """
    Chameleon-quantized nn.Linear.

    Weights:      pre-quantized per output-channel via SNR.
    Activations:  runtime fake-quantized via DynamicTimestepFakeQuantize LUT.
    """

    def __init__(
        self,
        original_layer:    nn.Linear,
        weight_format_info: LayerFormatInfo,
        mx_block_size:     int  = 32,
        is_shortcut:       bool = False,
        num_buckets:       int  = 10,
    ):
        super().__init__()
        self.layer              = original_layer
        self.weight_format_info = weight_format_info
        self.mx_block_size      = mx_block_size
        self.is_shortcut        = is_shortcut

        self.out_features = original_layer.out_features
        self.in_features  = original_layer.in_features

        self.weight_channel_formats = [
            info.selected_format for info in weight_format_info.channel_formats
        ]
        self._quantize_and_store_weights()

        self.fake_quant = DynamicTimestepFakeQuantize(
            num_buckets=num_buckets,
            mx_block_size=mx_block_size,
        )

        self.activation_stats: Optional[PerBucketActivationStats] = None
        self.calibrating  = False
        self.quantized    = True
        self.current_timestep: int = 0

    # ── Attribute forwarding (for diffusers compatibility) ─────────────────────

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

    # ── Weight quantization ────────────────────────────────────────────────────

    def _quantize_and_store_weights(self):
        w = self.layer.weight.data
        qw = torch.zeros_like(w)

        format_groups: Dict[QuantFormat, List[int]] = {}
        for ch, fmt in enumerate(self.weight_channel_formats):
            format_groups.setdefault(fmt, []).append(ch)

        for fmt, channels in format_groups.items():
            ch_weights = w[channels]
            if fmt == QuantFormat.FP16:
                qw[channels] = ch_weights
            elif fmt == QuantFormat.FLOAT8_E4M3:
                qw[channels] = quantize_to_fp8_e4m3(ch_weights)
            elif fmt == QuantFormat.FLOAT8_E5M2:
                qw[channels] = quantize_to_fp8_e5m2(ch_weights)
            elif fmt == QuantFormat.INT8_SYM:
                for i, ch in enumerate(channels):
                    qw[ch], _ = quantize_to_int8_symmetric(ch_weights[i])
            elif fmt == QuantFormat.INT8_ASYM:
                for i, ch in enumerate(channels):
                    qw[ch], _, _ = quantize_to_int8_asymmetric(ch_weights[i])
            elif fmt == QuantFormat.MXFP8_E4M3:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_mxfp8_e4m3(ch_weights[i], self.mx_block_size)
            elif fmt == QuantFormat.MXFP8_E5M2:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_mxfp8_e5m2(ch_weights[i], self.mx_block_size)
            elif fmt == QuantFormat.MXINT8:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_mxint8(ch_weights[i], self.mx_block_size)
            elif fmt == QuantFormat.INT4_SYM:
                for i, ch in enumerate(channels):
                    qw[ch], _ = quantize_to_int4_symmetric(ch_weights[i])
            elif fmt == QuantFormat.INT4_ASYM:
                for i, ch in enumerate(channels):
                    qw[ch], _, _ = quantize_to_int4_asymmetric(ch_weights[i])
            elif fmt == QuantFormat.NF4:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_nf4(ch_weights[i])
            elif fmt == QuantFormat.FP4_E2M1:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_fp4_e2m1(ch_weights[i])
            elif fmt == QuantFormat.MXINT4:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_mxint4(ch_weights[i], self.mx_block_size)
            elif fmt == QuantFormat.MXFP4_E2M1:
                for i, ch in enumerate(channels):
                    qw[ch] = quantize_to_mxfp4_e2m1(ch_weights[i], self.mx_block_size)

        self.register_buffer('quantized_weight', qw)

    # ── Runtime API ───────────────────────────────────────────────────────────

    def set_timestep(self, timestep: int):
        self.current_timestep = timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            output = self.layer(x)
            if self.activation_stats is not None:
                bsz = self.fake_quant.bucket_size
                nb  = self.fake_quant.num_buckets
                b   = min(self.current_timestep // bsz, nb - 1)
                self.activation_stats.update(output, b)
            return output

        if self.quantized:
            output = F.linear(x, self.quantized_weight, self.layer.bias)
            output = self.fake_quant(output, self.current_timestep)
            return output

        return self.layer(x)


class ChameleonQuantizedConv2d(nn.Module):
    """
    Chameleon-quantized nn.Conv2d.

    Weights:      pre-quantized per output-channel via SNR.
    Activations:  runtime fake-quantized via DynamicTimestepFakeQuantize LUT.
    """

    def __init__(
        self,
        original_layer:    nn.Conv2d,
        weight_format_info: LayerFormatInfo,
        mx_block_size:     int  = 32,
        is_shortcut:       bool = False,
        num_buckets:       int  = 10,
    ):
        super().__init__()
        self.layer              = original_layer
        self.weight_format_info = weight_format_info
        self.mx_block_size      = mx_block_size
        self.is_shortcut        = is_shortcut

        self.out_channels = original_layer.out_channels
        self.in_channels  = original_layer.in_channels

        self.weight_channel_formats = [
            info.selected_format for info in weight_format_info.channel_formats
        ]
        self._quantize_and_store_weights()

        self.fake_quant = DynamicTimestepFakeQuantize(
            num_buckets=num_buckets,
            mx_block_size=mx_block_size,
        )

        self.activation_stats: Optional[PerBucketActivationStats] = None
        self.calibrating  = False
        self.quantized    = True
        self.current_timestep: int = 0

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
        w  = self.layer.weight.data
        qw = torch.zeros_like(w)

        format_groups: Dict[QuantFormat, List[int]] = {}
        for ch, fmt in enumerate(self.weight_channel_formats):
            format_groups.setdefault(fmt, []).append(ch)

        for fmt, channels in format_groups.items():
            ch_weights = w[channels]
            if fmt == QuantFormat.FP16:
                qw[channels] = ch_weights
            elif fmt == QuantFormat.FLOAT8_E4M3:
                qw[channels] = quantize_to_fp8_e4m3(ch_weights)
            elif fmt == QuantFormat.FLOAT8_E5M2:
                qw[channels] = quantize_to_fp8_e5m2(ch_weights)
            elif fmt == QuantFormat.INT8_SYM:
                for i, ch in enumerate(channels):
                    flat        = ch_weights[i].flatten()
                    q_flat, _   = quantize_to_int8_symmetric(flat)
                    qw[ch]      = q_flat.view_as(w[ch])
            elif fmt == QuantFormat.INT8_ASYM:
                for i, ch in enumerate(channels):
                    flat           = ch_weights[i].flatten()
                    q_flat, _, _   = quantize_to_int8_asymmetric(flat)
                    qw[ch]         = q_flat.view_as(w[ch])
            elif fmt == QuantFormat.MXFP8_E4M3:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_mxfp8_e4m3(flat, self.mx_block_size).view_as(w[ch])
            elif fmt == QuantFormat.MXFP8_E5M2:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_mxfp8_e5m2(flat, self.mx_block_size).view_as(w[ch])
            elif fmt == QuantFormat.MXINT8:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_mxint8(flat, self.mx_block_size).view_as(w[ch])
            elif fmt == QuantFormat.INT4_SYM:
                for i, ch in enumerate(channels):
                    flat        = ch_weights[i].flatten()
                    q_flat, _   = quantize_to_int4_symmetric(flat)
                    qw[ch]      = q_flat.view_as(w[ch])
            elif fmt == QuantFormat.INT4_ASYM:
                for i, ch in enumerate(channels):
                    flat           = ch_weights[i].flatten()
                    q_flat, _, _   = quantize_to_int4_asymmetric(flat)
                    qw[ch]         = q_flat.view_as(w[ch])
            elif fmt == QuantFormat.NF4:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_nf4(flat).view_as(w[ch])
            elif fmt == QuantFormat.FP4_E2M1:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_fp4_e2m1(flat).view_as(w[ch])
            elif fmt == QuantFormat.MXINT4:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_mxint4(flat, self.mx_block_size).view_as(w[ch])
            elif fmt == QuantFormat.MXFP4_E2M1:
                for i, ch in enumerate(channels):
                    flat   = ch_weights[i].flatten()
                    qw[ch] = quantize_to_mxfp4_e2m1(flat, self.mx_block_size).view_as(w[ch])

        self.register_buffer('quantized_weight', qw)

    def set_timestep(self, timestep: int):
        self.current_timestep = timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            output = self.layer(x)
            if self.activation_stats is not None:
                bsz = self.fake_quant.bucket_size
                nb  = self.fake_quant.num_buckets
                b   = min(self.current_timestep // bsz, nb - 1)
                self.activation_stats.update(output, b)
            return output

        if self.quantized:
            output = F.conv2d(
                x, self.quantized_weight, self.layer.bias,
                self.layer.stride, self.layer.padding,
                self.layer.dilation, self.layer.groups,
            )
            output = self.fake_quant(output, self.current_timestep)
            return output

        return self.layer(x)


# ── Main quantizer ─────────────────────────────────────────────────────────────

class ChameleonQuantizer:
    """
    Chameleon quantizer for SDXL UNet.

    Step 1 — Weight quantization:
        SNR-based per-channel format selection (INT8/MXINT8).
        Results saved/loaded as JSON.

    Step 2 — Activation calibration:
        Per-bucket kurtosis + diffusion-SNR routing → LUT.
        UNet skip layers always route to MXFP8 E4M3.
    """

    def __init__(
        self,
        model_id:          str              = "stabilityai/stable-diffusion-xl-base-1.0",
        device:            str              = "cuda",
        dtype:             torch.dtype      = torch.float16,
        formats:           Optional[List[QuantFormat]] = None,
        min_snr_threshold: float            = 20.0,
        mx_block_size:     int              = 32,
        num_buckets:       int              = 10,
        weight_bits:       int              = 8,
    ):
        self.device            = device
        self.dtype             = dtype
        self.model_id          = model_id
        self.mx_block_size     = mx_block_size
        self.num_buckets       = num_buckets
        self.weight_bits       = weight_bits

        self.format_selector = SNRFormatSelector(
            formats=formats,
            min_snr_threshold=min_snr_threshold,
            mx_block_size=mx_block_size,
            weight_bits=weight_bits,
        )

        self.pipe:     Optional[StableDiffusionXLPipeline] = None
        self.quant_layers: List = []
        self.layer_names:  List[str] = []
        self.weight_quantization_complete    = False
        self.activation_calibration_complete = False

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self):
        """Load SDXL pipeline."""
        print(f"Loading SDXL from {self.model_id}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)
        self.pipe.unet.eval()

        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            print(f"Pipeline loaded on {torch.cuda.get_device_name(dev_idx)}")
        return self.pipe

    # ── Step 1: weight quantization ───────────────────────────────────────────

    def inject_weight_quantization(
        self,
        skip_attention_qkv:    bool = True,
        skip_attention_out:    bool = False,
        skip_first_last_conv:  bool = True,
        verbose:               bool = False,
    ):
        """Analyze and inject SNR-based per-channel weight quantization."""
        if self.pipe is None:
            self.load_model()

        print(f"\n{'='*60}")
        print("CHAMELEON STEP 1: WEIGHT QUANTIZATION")
        print(f"{'='*60}")
        print(f"Formats: {[f.value for f in self.format_selector.formats]}")
        print(f"Min SNR threshold: {self.format_selector.min_snr_threshold} dB")
        print(f"MX block size: {self.mx_block_size}")

        skip_patterns = []
        if skip_attention_qkv:
            skip_patterns.extend(['to_q', 'to_k', 'to_v', 'query', 'key', 'value'])
        if skip_attention_out:
            skip_patterns.extend(['to_out'])
        if skip_first_last_conv:
            skip_patterns.extend(['conv_in', 'conv_out', 'conv_norm_out'])

        def should_skip(name: str) -> bool:
            nl = name.lower()
            return any(p.lower() in nl for p in skip_patterns)

        layers_to_quantize = [
            (name, mod)
            for name, mod in self.pipe.unet.named_modules()
            if isinstance(mod, (nn.Linear, nn.Conv2d)) and not should_skip(name)
        ]
        print(f"Layers to quantize: {len(layers_to_quantize)}")
        print(f"{'='*60}\n")

        for name, module in tqdm(layers_to_quantize, desc="Analyzing weights"):
            is_conv    = isinstance(module, nn.Conv2d)
            is_shortcut = is_unet_shortcut_layer(name)

            weight_format_info = self.format_selector.analyze_weights(
                weight_tensor=module.weight.data,
                layer_name=name,
                is_conv=is_conv,
                verbose=verbose,
            )

            if is_conv:
                ql = ChameleonQuantizedConv2d(
                    module, weight_format_info,
                    self.mx_block_size, is_shortcut, self.num_buckets,
                )
            else:
                ql = ChameleonQuantizedLinear(
                    module, weight_format_info,
                    self.mx_block_size, is_shortcut, self.num_buckets,
                )

            self._replace_module(self.pipe.unet, name, ql)
            self.quant_layers.append(ql)
            self.layer_names.append(name)

        self.weight_quantization_complete = True

    # ── Step 2: activation calibration ────────────────────────────────────────

    def calibrate_activations(
        self,
        num_calibration_samples: int   = 128,
        num_inference_steps:     int   = 20,
        guidance_scale:          float = 7.5,
        verbose:                 bool  = False,
    ):
        """
        Calibrate per-bucket activation format LUT using kurtosis + diffusion SNR.

        Algorithm:
          1. Collect per-bucket activation stats over a full denoising loop.
          2. For each bucket: compute κ, SNR(t_mid), route to format.
          3. Compute scale/zp and populate fake_quant LUT.
          4. Enable fake_quant on all layers.
        """
        if not self.weight_quantization_complete:
            raise RuntimeError("Call inject_weight_quantization (or load_weight_quantization) first.")

        print(f"\n{'='*60}")
        print("CHAMELEON STEP 2: DYNAMIC ACTIVATION CALIBRATION")
        print(f"{'='*60}")
        print(f"Calibration samples : {num_calibration_samples}")
        print(f"Inference steps     : {num_inference_steps}")
        print(f"Num buckets         : {self.num_buckets}")
        print(f"Kurtosis thresholds : low={3.0}, high={5.0}")
        print(f"Diffusion SNR thr.  : low={0.2}, high={2.0}")
        print(f"{'='*60}\n")

        # Step 1: initialise stats + set calibration mode
        for layer in self.quant_layers:
            layer.activation_stats = PerBucketActivationStats(self.num_buckets)
            layer.calibrating      = True
            layer.quantized        = False

        # Step 2: collect stats
        prompts = self._get_calibration_prompts(num_calibration_samples)
        print(f"Running {len(prompts)} calibration samples...")
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
                try:
                    self._run_calibration_sample(prompt, i, num_inference_steps, guidance_scale)
                    if (i + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"\nError at sample {i}: {e}")
                    import traceback
                    traceback.print_exc()

        # Step 3: build LUT from stats
        alphas_cumprod = self.pipe.scheduler.alphas_cumprod
        bucket_size    = 1000 // self.num_buckets

        print("\nBuilding activation LUT per layer...")
        for layer, name in zip(self.quant_layers, self.layer_names):
            if layer.activation_stats is None:
                continue

            for b in range(self.num_buckets):
                if not layer.activation_stats.has_data(b):
                    continue

                kurtosis    = layer.activation_stats.compute_kurtosis(b)
                t_mid       = min(b * bucket_size + bucket_size // 2, len(alphas_cumprod) - 1)
                diff_snr    = compute_diffusion_snr(t_mid, alphas_cumprod)
                fmt         = route_format_by_kurtosis_snr(kurtosis, diff_snr, layer.is_shortcut)
                scale, zp   = layer.activation_stats.compute_scale_zp(b, fmt)
                fmt_code    = QUANT_FORMAT_TO_CODE.get(fmt, 1)  # default FP8_E4M3
                layer.fake_quant.calibrate_bucket(b, fmt_code, scale, zp)

            layer.fake_quant.enabled = True
            layer.calibrating        = False
            layer.quantized          = True

            if verbose:
                layer.fake_quant.lut_summary(name)

        self.activation_calibration_complete = True
        self._print_activation_summary()

    # ── Save / load weight quantization ───────────────────────────────────────

    def save_weight_quantization(self, save_path: str):
        """Save weight (and optionally activation) quantization config to JSON."""
        if not self.weight_quantization_complete:
            raise RuntimeError("Weight quantization not complete.")

        config = {
            'model_id':         self.model_id,
            'mx_block_size':    self.mx_block_size,
            'num_buckets':      self.num_buckets,
            'weight_bits':      self.weight_bits,
            'min_snr_threshold': self.format_selector.min_snr_threshold,
            'num_layers':       len(self.quant_layers),
            'layers':           {},
            'weight_format_distribution': {
                fmt.value: count for fmt, count in
                self.format_selector.get_global_weight_format_distribution().items()
            },
        }

        for name, layer in zip(self.layer_names, self.quant_layers):
            config['layers'][name] = {
                'formats':      [fmt.value for fmt in layer.weight_channel_formats],
                'num_channels': len(layer.weight_channel_formats),
                'is_shortcut':  layer.is_shortcut,
            }

        # Embed activation LUT if calibration is complete
        if self.activation_calibration_complete:
            config['activation_lut'] = {}
            for name, layer in zip(self.layer_names, self.quant_layers):
                config['activation_lut'][name] = {
                    'format_lut': layer.fake_quant.format_lut.tolist(),
                    'scale_lut':  layer.fake_quant.scale_lut.tolist(),
                    'zp_lut':     layer.fake_quant.zp_lut.tolist(),
                }

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Weight quantization config saved to {save_path}")

    def load_weight_quantization(self, load_path: str):
        """Load weight (and optionally activation) quantization config from JSON."""
        if self.pipe is None:
            self.load_model()

        with open(load_path, 'r') as f:
            config = json.load(f)

        weight_bits = config.get('weight_bits', 8)

        print(f"\n{'='*60}")
        print("LOADING WEIGHT QUANTIZATION FROM FILE")
        print(f"{'='*60}")
        print(f"Config      : {load_path}")
        print(f"Model ID    : {config['model_id']}")
        print(f"Weight bits : {weight_bits}")
        print(f"Layers      : {config['num_layers']}")
        print(f"{'='*60}\n")

        # The quantizer's configured bucket count (from --num-buckets, stored in
        # self.num_buckets) is authoritative for the activation fake-quant — NOT
        # the count baked into the weights file. The bucket-count ablation reuses
        # one shared 10-bucket W4A8 weights file at B ∈ {1,5,20}; taking the file's
        # value here silently pinned every layer to 10 (B=20 then indexed out of
        # bounds; B=1/5 mis-bucketed). Build layers at self.num_buckets instead.
        file_num_buckets = config.get('num_buckets', self.num_buckets)
        num_buckets      = self.num_buckets

        for name, layer_cfg in tqdm(config['layers'].items(), desc="Loading weights"):
            module = self._get_module(self.pipe.unet, name)
            if module is None:
                print(f"Warning: layer {name} not found in model")
                continue

            is_conv     = isinstance(module, nn.Conv2d)
            is_shortcut = layer_cfg.get('is_shortcut', is_unet_shortcut_layer(name))
            formats     = [QuantFormat(fv) for fv in layer_cfg['formats']]

            channel_formats = [
                ChannelFormatInfo(channel_idx=i, selected_format=fmt, snr_db=0.0)
                for i, fmt in enumerate(formats)
            ]
            weight_format_info = LayerFormatInfo(
                layer_name=name,
                channel_formats=channel_formats,
                format_distribution={fmt: formats.count(fmt) for fmt in set(formats)},
            )
            self.format_selector.weight_layer_infos[name] = weight_format_info

            mx = config.get('mx_block_size', 32)
            if is_conv:
                ql = ChameleonQuantizedConv2d(
                    module, weight_format_info, mx, is_shortcut, num_buckets
                )
            else:
                ql = ChameleonQuantizedLinear(
                    module, weight_format_info, mx, is_shortcut, num_buckets
                )

            self._replace_module(self.pipe.unet, name, ql)
            self.quant_layers.append(ql)
            self.layer_names.append(name)

        self.weight_quantization_complete = True

        # Restore activation LUT if present — but only when its bucket count
        # matches the layers we just built. When they differ (bucket-count
        # ablation: 10-bucket embedded LUT, B-bucket layers), skip it; the caller
        # supplies the correct LUT via --load-activation-lut, or recalibrates.
        if 'activation_lut' in config and file_num_buckets != num_buckets:
            print(f"Skipping embedded {file_num_buckets}-bucket LUT restore "
                  f"(layers built at {num_buckets} buckets); "
                  f"provide --load-activation-lut or calibrate.")
        elif 'activation_lut' in config:
            print("Restoring activation LUT from config...")
            for name, layer in zip(self.layer_names, self.quant_layers):
                if name not in config['activation_lut']:
                    continue
                lut = config['activation_lut'][name]
                for b in range(num_buckets):
                    layer.fake_quant.calibrate_bucket(
                        b,
                        lut['format_lut'][b],
                        lut['scale_lut'][b],
                        lut['zp_lut'][b],
                    )
                layer.fake_quant.enabled = True
            self.activation_calibration_complete = True
            print("Activation LUT restored.")

        print(f"\nLoaded {len(self.quant_layers)} layers")
        print("Weight format distribution:")
        for fmt_str, count in config.get('weight_format_distribution', {}).items():
            print(f"  {fmt_str}: {count}")

    # ── Separate activation LUT save / load ───────────────────────────────────

    def save_activation_lut(self, path: str):
        """Save per-layer activation LUT to a separate JSON file."""
        if not self.activation_calibration_complete:
            raise RuntimeError("Activation calibration not complete.")

        lut_data = {
            'num_buckets': self.num_buckets,
            'layers':      {},
        }
        for name, layer in zip(self.layer_names, self.quant_layers):
            lut_data['layers'][name] = {
                'format_lut': layer.fake_quant.format_lut.tolist(),
                'scale_lut':  layer.fake_quant.scale_lut.tolist(),
                'zp_lut':     layer.fake_quant.zp_lut.tolist(),
            }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(lut_data, f, indent=2)
        print(f"Activation LUT saved to {path}")

    def load_activation_lut(self, path: str):
        """Load per-layer activation LUT and enable fake quantization."""
        if not self.weight_quantization_complete:
            raise RuntimeError("Must load/inject weight quantization first.")

        with open(path, 'r') as f:
            lut_data = json.load(f)

        nb = lut_data.get('num_buckets', self.num_buckets)
        # The loaded LUT is authoritative for bucket count. Weights are timestep-
        # independent, but load_weight_quantization pins each layer's fake-quant to
        # the num_buckets baked into the weights file (default 10). Re-shape here so
        # the bucket-count ablation (shared 10-bucket weights + a B-bucket LUT)
        # actually runs at B buckets instead of silently staying at 10.
        self.num_buckets = nb
        for name, layer in zip(self.layer_names, self.quant_layers):
            if name not in lut_data['layers']:
                print(f"Warning: {name} not in LUT file")
                continue
            lut = lut_data['layers'][name]
            layer.fake_quant.resize_buckets(nb)
            for b in range(nb):
                layer.fake_quant.calibrate_bucket(
                    b,
                    lut['format_lut'][b],
                    lut['scale_lut'][b],
                    lut['zp_lut'][b],
                )
            layer.fake_quant.enabled = True

        self.activation_calibration_complete = True
        print(f"Activation LUT loaded from {path}")

    # ── Calibration loop ───────────────────────────────────────────────────────

    def _run_calibration_sample(
        self,
        prompt:             str,
        sample_idx:         int,
        num_inference_steps: int,
        guidance_scale:     float,
    ):
        """Run one full denoising pass, collecting stats at every timestep."""
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = \
            self.pipe.encode_prompt(prompt)

        latents = torch.randn(
            (1, 4, 128, 128),
            device=self.device,
            dtype=self.dtype,
            generator=torch.Generator(device=self.device).manual_seed(sample_idx),
        )

        self.pipe.scheduler.set_timesteps(num_inference_steps)

        for t in self.pipe.scheduler.timesteps:
            t_val = int(t.item())

            # Update timestep for every layer (bucket computed in forward)
            for layer in self.quant_layers:
                layer.set_timestep(t_val)

            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            add_time_ids = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]],
                device=self.device, dtype=self.dtype,
            )
            added_cond_kwargs = {
                "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
                "time_ids":    torch.cat([add_time_ids] * 2),
            }

            noise_pred = self.pipe.unet(
                latent_input, t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents    = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # ── Image generation ───────────────────────────────────────────────────────

    def generate_images(
        self,
        prompts:             List[str],
        output_dir:          str,
        num_inference_steps: int   = 50,
        guidance_scale:      float = 7.5,
        batch_size:          int   = 1,
        start_idx:           int   = 0,
    ):
        """Generate images using the Chameleon-quantized model."""
        if not self.activation_calibration_complete:
            print("Warning: Activation calibration not complete — weights-only quantization.")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating {len(prompts)} images...")

        # Pre-compute scheduler timesteps once.  The pipe will call set_timesteps
        # internally with the same count so these values are identical.
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        sched_timesteps = self.pipe.scheduler.timesteps.clone()

        def _pre_init_layers():
            """Set all layers to the first scheduled timestep.
            This fixes the off-by-one on step 0: without this, current_timestep=0
            (bucket 0 / clean-signal parameters) would be used for the first UNet
            forward at t≈999 (pure-noise), causing GroupNorm σ≈0 → NaN images."""
            if sched_timesteps.numel() > 0 and self.quant_layers:
                t_first = int(sched_timesteps[0].item())
                for layer in self.quant_layers:
                    layer.set_timestep(t_first)

        def _generation_callback(step: int, timestep: int, latents: torch.Tensor):
            """Look-ahead callback: set layers to the NEXT step's timestep so
            the next UNet forward immediately sees the correct LUT bucket."""
            next_step = step + 1
            if next_step < len(sched_timesteps):
                next_t = int(sched_timesteps[next_step].item())
            else:
                next_t = int(timestep)  # final step — hold current
            for layer in self.quant_layers:
                layer.set_timestep(next_t)

        tracker = PerfTracker(
            label=f"sdxl_chameleon_w{self.weight_bits}a8",
            device=self.device,
        )
        tracker.reset()

        with torch.no_grad():
            for bs in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
                be    = min(bs + batch_size, len(prompts))
                batch = prompts[bs:be]
                try:
                    _pre_init_layers()
                    gens = [
                        torch.Generator(device=self.device).manual_seed(start_idx + bs + j)
                        for j in range(len(batch))
                    ]
                    tracker.batch_start()
                    result = self.pipe(
                        prompt=batch,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=gens,
                        callback=_generation_callback,
                        callback_steps=1,
                    )
                    tracker.batch_end(len(batch))
                    for j, image in enumerate(result.images):
                        image.save(f"{output_dir}/{start_idx + bs + j:05d}.png")
                except Exception as e:
                    print(f"Error at batch {bs}: {e}")
                    for j, prompt in enumerate(batch):
                        try:
                            idx = start_idx + bs + j
                            _pre_init_layers()
                            res = self.pipe(
                                prompt=prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                generator=torch.Generator(device=self.device).manual_seed(idx),
                                callback=_generation_callback,
                                callback_steps=1,
                            )
                            res.images[0].save(f"{output_dir}/{idx:05d}.png")
                        except Exception as e2:
                            print(f"  Error at image {start_idx + bs + j}: {e2}")

        tracker.finish()
        tracker.dump_json(os.path.join(output_dir, "perf.json"))
        print(f"Generated {len(prompts)} images in {output_dir}")

    # ── Statistics / printing ─────────────────────────────────────────────────

    def print_weight_stats(self):
        """Print weight quantization statistics."""
        if not self.weight_quantization_complete:
            print("Weight quantization not complete.")
            return

        print(f"\n{'='*60}")
        print("WEIGHT QUANTIZATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total layers: {len(self.quant_layers)}")

        weight_dist = self.format_selector.get_global_weight_format_distribution()
        total = sum(weight_dist.values())
        print(f"Total output channels: {total}")
        for fmt, count in sorted(weight_dist.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"  {fmt.value}: {count} ({pct:.1f}%)")
        print(f"{'='*60}\n")

    def _print_activation_summary(self):
        """Print per-bucket format distribution across all layers."""
        bucket_size = 1000 // self.num_buckets
        print(f"\n{'='*60}")
        print("ACTIVATION CALIBRATION COMPLETE — LUT SUMMARY")
        print(f"{'='*60}")
        print(f"{'Bucket':>6} | {'t_range':>9} | Format distribution (across all layers)")
        print(f"{'-'*6}-+-{'-'*9}-+-{'-'*40}")
        for b in range(self.num_buckets):
            t_lo, t_hi = b * bucket_size, (b + 1) * bucket_size - 1
            fmt_counts: Dict[str, int] = {}
            for layer in self.quant_layers:
                code = int(layer.fake_quant.format_lut[b].item())
                nm   = FORMAT_NAMES.get(code, f"?({code})")
                fmt_counts[nm] = fmt_counts.get(nm, 0) + 1
            dist_str = ", ".join(f"{nm}:{c}" for nm, c in sorted(fmt_counts.items()))
            print(f"{b:6d} | {t_lo:4d}-{t_hi:4d}  | {dist_str}")
        print(f"{'='*60}\n")

    def get_quantization_stats(self) -> dict:
        """Return statistics about the quantized model."""
        stats = {
            "num_quantized_layers":          len(self.quant_layers),
            "weight_quantization_complete":  self.weight_quantization_complete,
            "activation_calibration_complete": self.activation_calibration_complete,
            "weight_format_distribution":    self.format_selector.get_global_weight_format_distribution(),
            "num_buckets":                   self.num_buckets,
        }

        if self.activation_calibration_complete:
            bucket_size = 1000 // self.num_buckets
            bucket_dist: Dict[int, Dict[str, int]] = {}
            for b in range(self.num_buckets):
                fmt_counts: Dict[str, int] = {}
                for layer in self.quant_layers:
                    code = int(layer.fake_quant.format_lut[b].item())
                    nm   = FORMAT_NAMES.get(code, f"?({code})")
                    fmt_counts[nm] = fmt_counts.get(nm, 0) + 1
                bucket_dist[b] = fmt_counts
            stats["activation_bucket_distribution"] = bucket_dist

        return stats

    # ── Module helpers ─────────────────────────────────────────────────────────

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        parts, parent = name.split('.'), model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _get_module(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        try:
            m = model
            for part in name.split('.'):
                m = getattr(m, part)
            return m
        except AttributeError:
            return None

    # ── Calibration prompts ────────────────────────────────────────────────────

    def _get_calibration_prompts(self, num_samples: int) -> List[str]:
        base = [
            "A photo of a cat sitting on a windowsill",
            "A portrait of a wise old wizard with a long beard",
            "A futuristic city skyline at night with neon lights",
            "A serene mountain landscape with snow-capped peaks",
            "A tropical beach with crystal clear water",
            "An astronaut riding a horse on Mars",
            "A cozy coffee shop interior",
            "A majestic lion in the African savanna",
            "A Japanese garden with cherry blossoms",
            "A steampunk mechanical robot",
            "A colorful butterfly on a flower",
            "A medieval castle on a cliff",
            "A golden retriever running on a beach",
            "A bowl of fresh fruits on a wooden table",
            "A modern glass skyscraper reflecting clouds",
            "An abstract painting with vibrant colors",
        ]
        prompts = []
        while len(prompts) < num_samples:
            prompts.extend(base)
        return prompts[:num_samples]

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()


# ── Factory function ───────────────────────────────────────────────────────────

def create_chameleon_quantizer(
    formats:           Optional[List[QuantFormat]] = None,
    min_snr_threshold: float = 20.0,
    mx_block_size:     int   = 32,
    device:            str   = "cuda",
    num_buckets:       int   = 10,
    weight_bits:       int   = 8,
) -> ChameleonQuantizer:
    """Factory function to create a ChameleonQuantizer."""
    return ChameleonQuantizer(
        formats=formats,
        min_snr_threshold=min_snr_threshold,
        mx_block_size=mx_block_size,
        device=device,
        num_buckets=num_buckets,
        weight_bits=weight_bits,
    )
