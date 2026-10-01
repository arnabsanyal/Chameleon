"""
Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers
Applied to PixArt-alpha (DiT-based T2I model).

Reference: Chen et al., "Q-DiT: Accurate Post-Training Quantization for Diffusion
           Transformers", arXiv:2406.17343

Two core components (§5):

  1. Group Weight Quantization (offline, §4 + §5.1):
       Weights are grouped along the input-channel dimension.
       Each group gets its own (scale, zero_point) pair.
       Asymmetric uniform quantization: s = (max-min)/(2^b-1),  Z = -⌊min/s⌋
       Group size per layer is configurable (paper default: 128).

  2. Sample-wise Dynamic Activation Quantization (online, §5.2):
       For each forward pass (each sample i at timestep t), activation
       min-max is computed on-the-fly:
           s_{i,t} = (max(x_{i,t}) - min(x_{i,t})) / (2^b - 1)
           Z_{i,t} = -⌊min(x_{i,t}) / s_{i,t}⌋
       Applied in the same group structure as the corresponding weight layer.

PixArt-alpha layer coverage (per block, 28 blocks × 10 linears = 280 total):
  - Self-attention:  attn1.{to_q, to_k, to_v, to_out.0}  (in=1152, out=1152)
  - Cross-attention: attn2.{to_q, to_k, to_v, to_out.0}  (in=1152, out=1152)
  - Feed-forward:    ff.net.0.proj (in=1152→4608),  ff.net.2 (in=4608→1152)

Skipped layers: proj_out, adaln_single.* (embedders, output head — sensitive)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from diffusers import PixArtAlphaPipeline
from tqdm import tqdm
import os
import json
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402


# ── Paper constants ────────────────────────────────────────────────────────────

# Group size search space from the paper (§5.1 / Algorithm 1)
GROUP_SIZE_SEARCH_SPACE = [32, 64, 128, 192, 288]

# Default group size: 128 (paper's best for DiT-XL/2 256×256, §6.1 Table 1)
DEFAULT_GROUP_SIZE = 128


# ── Low-level quantization primitives ─────────────────────────────────────────

def _asym_group_quant(grouped: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Core asymmetric uniform quantization for a (N, group_size) tensor.
    Each row is one group; returns dequantized tensor of the same shape.

    Q(x; b) = s · (clip(⌊x/s⌋ + Z, 0, 2^b-1) - Z)
    where  s = (max(x) - min(x)) / (2^b - 1),  Z = -⌊min(x) / s⌋
    """
    qmax  = 2 ** bits - 1
    g_min = grouped.min(dim=1, keepdim=True)[0]
    g_max = grouped.max(dim=1, keepdim=True)[0]
    scale = (g_max - g_min).clamp(min=1e-8) / qmax
    zp    = torch.clamp(torch.round(-g_min / scale), 0, qmax)
    q     = torch.clamp(torch.round(grouped / scale + zp), 0, qmax)
    return (q - zp) * scale


def group_quantize_weight(
    weight:     torch.Tensor,
    bits:       int,
    group_size: int,
) -> torch.Tensor:
    """
    Static group quantization of a weight matrix along the input-channel dim.

    W ∈ R^{d_out × d_in} is reshaped so that d_in is chunked into groups of
    `group_size`. Each (d_out, group) sub-tensor is quantized independently.
    Returns a dequantized float tensor of the same shape (fake-quant).

    If group_size ≥ d_in, falls back to tensor-wise quantization.
    """
    orig_shape = weight.shape
    d_out = orig_shape[0]
    flat  = weight.reshape(d_out, -1)          # (d_out, d_in)
    d_in  = flat.shape[1]

    if group_size >= d_in:
        # Tensor-wise fallback
        return _asym_group_quant(flat.unsqueeze(0), bits).squeeze(0).reshape(orig_shape)

    pad = (group_size - d_in % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))

    num_groups = flat.shape[1] // group_size
    # Reshape to (d_out * num_groups, group_size) — one row per group
    grouped = flat.reshape(d_out * num_groups, group_size)
    dq      = _asym_group_quant(grouped.float(), bits).to(weight.dtype)

    dq_flat = dq.reshape(d_out, -1)[:, :d_in]
    return dq_flat.reshape(orig_shape)


def group_quantize_activation(
    x:          torch.Tensor,
    bits:       int,
    group_size: int,
) -> torch.Tensor:
    """
    Dynamic (sample-wise) group quantization of an activation tensor.

    Activation x of shape (..., d_in) is grouped along d_in; (scale, zp)
    are computed on-the-fly from the current min/max — this is the
    sample-wise dynamic quantization from §5.2 (Eqs. 7–8).
    """
    orig_shape = x.shape
    d_in  = orig_shape[-1]
    flat  = x.reshape(-1, d_in)               # (N, d_in)
    N     = flat.shape[0]

    if group_size >= d_in:
        dq = _asym_group_quant(flat.float(), bits).to(x.dtype)
        return dq.reshape(orig_shape)

    pad = (group_size - d_in % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))

    num_groups = flat.shape[1] // group_size
    grouped = flat.reshape(N * num_groups, group_size)
    dq      = _asym_group_quant(grouped.float(), bits).to(x.dtype)

    dq_flat = dq.reshape(N, -1)[:, :d_in]
    return dq_flat.reshape(orig_shape)


# ── Quantized linear layer ─────────────────────────────────────────────────────

class QDiTQuantizedLinear(nn.Module):
    """
    Q-DiT quantized nn.Linear.

    Weight:     group-quantized offline (static fake-quant, stored as buffer).
    Activation: group-quantized online per forward pass (dynamic, sample-wise).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        weight_bits:    int = 8,
        act_bits:       int = 8,
        group_size:     int = DEFAULT_GROUP_SIZE,
    ):
        super().__init__()
        self.layer       = original_layer
        self.weight_bits = weight_bits
        self.act_bits    = act_bits
        self.group_size  = group_size
        self.quantized   = True

        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features

        # Pre-quantize weights offline
        with torch.no_grad():
            qw = group_quantize_weight(
                original_layer.weight.data,
                weight_bits,
                group_size,
            )
        self.register_buffer('quantized_weight', qw)

    # ── Attribute forwarding (diffusers compatibility) ─────────────────────────

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return self.layer(x)

        # Dynamic sample-wise activation quantization (§5.2, Eqs. 7–8)
        x_q = group_quantize_activation(x, self.act_bits, self.group_size)
        return F.linear(x_q, self.quantized_weight, self.layer.bias)


# ── Main quantizer ─────────────────────────────────────────────────────────────

class QDiTQuantizer:
    """
    Q-DiT post-training quantizer for PixArt-alpha.

    Applies group quantization to all nn.Linear layers inside the DiT
    transformer_blocks (self-attention, cross-attention, FFN).
    Skips: proj_out, adaln_single.* (embedding/output layers).
    """

    # Patterns to skip (sensitive layers — embedders and output head)
    DEFAULT_SKIP = ['proj_out', 'adaln_single']

    def __init__(
        self,
        model_id:    str          = "PixArt-alpha/PixArt-XL-2-1024-MS",
        weight_bits: int          = 8,
        act_bits:    int          = 8,
        group_size:  int          = DEFAULT_GROUP_SIZE,
        device:      str          = "cuda",
        dtype:       torch.dtype  = torch.float16,
    ):
        self.model_id    = model_id
        self.weight_bits = weight_bits
        self.act_bits    = act_bits
        self.group_size  = group_size
        self.device      = device
        self.dtype       = dtype

        self.pipe:          Optional[PixArtAlphaPipeline] = None
        self.quant_layers:  List[QDiTQuantizedLinear]     = []
        self.layer_names:   List[str]                     = []
        self.quant_complete = False

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self):
        print(f"Loading PixArt-alpha from {self.model_id}...")
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            print(f"Pipeline on {torch.cuda.get_device_name(dev_idx)} (cuda:{dev_idx})")

        return self.pipe

    # ── Quantization injection ─────────────────────────────────────────────────

    def inject_quantization(
        self,
        skip_patterns: Optional[List[str]] = None,
        verbose:       bool = False,
    ):
        """
        Replace all eligible nn.Linear layers in transformer_blocks with
        QDiTQuantizedLinear (group weight quant + dynamic activation quant).
        """
        if self.pipe is None:
            self.load_model()

        skip = skip_patterns if skip_patterns is not None else self.DEFAULT_SKIP

        print(f"\n{'='*60}")
        print("Q-DiT QUANTIZATION INJECTION")
        print(f"{'='*60}")
        print(f"Weight bits : {self.weight_bits}")
        print(f"Act bits    : {self.act_bits}")
        print(f"Group size  : {self.group_size}")
        print(f"Skip        : {skip}")

        def _should_skip(name: str) -> bool:
            return any(p in name for p in skip)

        candidates = [
            (name, mod)
            for name, mod in self.pipe.transformer.named_modules()
            if isinstance(mod, nn.Linear) and not _should_skip(name)
        ]
        print(f"Layers to quantize: {len(candidates)}")
        print(f"{'='*60}\n")

        for name, module in tqdm(candidates, desc="Quantizing layers"):
            ql = QDiTQuantizedLinear(
                original_layer=module,
                weight_bits=self.weight_bits,
                act_bits=self.act_bits,
                group_size=self.group_size,
            )
            self._replace_module(self.pipe.transformer, name, ql)
            self.quant_layers.append(ql)
            self.layer_names.append(name)

            if verbose:
                print(f"  Quantized: {name}  ({module.in_features}→{module.out_features})")

        self.quant_complete = True
        self._print_stats()

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
        """Generate images with the Q-DiT quantized PixArt-alpha pipeline."""
        if not self.quant_complete:
            print("Warning: quantization not injected — generating with full-precision model.")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating {len(prompts)} images (start_idx={start_idx})...")

        tracker = PerfTracker(
            label=f"q_dit_w{self.weight_bits}a{self.act_bits}",
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

    # ── Statistics ─────────────────────────────────────────────────────────────

    def _print_stats(self):
        if not self.quant_layers:
            return
        print(f"\n{'='*60}")
        print("Q-DiT QUANTIZATION STATISTICS")
        print(f"{'='*60}")
        print(f"Quantized layers : {len(self.quant_layers)}")
        print(f"Weight bits      : {self.weight_bits}")
        print(f"Activation bits  : {self.act_bits}")
        print(f"Group size       : {self.group_size}")

        # Count layer types
        type_counts: Dict[str, int] = {}
        for name in self.layer_names:
            parts = name.split('.')
            # e.g. transformer_blocks.0.attn1.to_q -> "attn1.to_q"
            key = '.'.join(parts[-2:]) if len(parts) >= 2 else name
            type_counts[key] = type_counts.get(key, 0) + 1
        print(f"\nLayer type distribution:")
        for ltype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {ltype}: {cnt}")
        print(f"{'='*60}\n")

    def get_stats(self) -> dict:
        return {
            "model_id":          self.model_id,
            "weight_bits":       self.weight_bits,
            "act_bits":          self.act_bits,
            "group_size":        self.group_size,
            "num_quant_layers":  len(self.quant_layers),
            "quant_complete":    self.quant_complete,
        }

    # ── Module helpers ─────────────────────────────────────────────────────────

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        parts  = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()


# ── Factory function ───────────────────────────────────────────────────────────

def create_q_dit_quantizer(
    model_id:    str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    weight_bits: int = 8,
    act_bits:    int = 8,
    group_size:  int = DEFAULT_GROUP_SIZE,
    device:      str = "cuda",
) -> QDiTQuantizer:
    """Factory function to create a QDiTQuantizer."""
    return QDiTQuantizer(
        model_id=model_id,
        weight_bits=weight_bits,
        act_bits=act_bits,
        group_size=group_size,
        device=device,
    )
