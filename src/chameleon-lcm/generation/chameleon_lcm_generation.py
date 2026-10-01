"""
Chameleon-LCM: Few-Step Quantized Diffusion Generation (Chameleon-on-MixDQ)

The few-step Chameleon-LCM model is built ON TOP of MixDQ: it keeps MixDQ's
calibrated static activation quantisation (act_scales + BOS + mixed precision)
and replaces only the *weight* fold with Chameleon's adaptive palette
(per-output-channel SNR selection @ W8; per-layer (group_size, format) search
@ W4).  Pass --chameleon-weights to enable the Chameleon weight palette; without
it this same driver reproduces the plain MixDQ baseline.

  - MixDQ baseline : python chameleon_lcm_generation.py ...                (no flag)
  - Chameleon-LCM  : python chameleon_lcm_generation.py --chameleon-weights ...

Why this and not a standalone Chameleon-LCM: the earlier standalone re-derived
activation scales with dynamic per-sample micro-scaling, which over-saturated in
the 1-step Turbo regime and lost ~15 FID to MixDQ.  MixDQ's calibrated static
activation scales are robust there, so we keep them and contribute only weights.

MixDQ applies mixed-precision post-training quantization to few-step diffusion
models, achieving near-lossless image quality with significantly reduced memory
and latency.  The officially released pre-quantized model targets SDXL Turbo
(1–4 inference steps, guidance-free) as the few-step backbone.

Quantization techniques (from the paper):
  1. BOS-Aware Quantization — pre-computes cross-attention outputs for the
     "Beginning of Sentence" text token, avoiding quantization of the most
     sensitivity-sensitive embeddings.
  2. Metric-Decoupled Sensitivity Analysis — evaluates per-layer sensitivity
     for image quality and text alignment independently, allowing different
     bit-widths per layer.
  3. Integer-Programming Bit-Width Allocation — solves an optimisation problem
     to find the Pareto-optimal mixed-precision config under a memory budget.

Reference:
  Zhao et al., "MixDQ: Memory-Efficient Few-Step Diffusion Models for Image
  Generation", ECCV 2024.  arXiv:2405.17873
  Code  : https://github.com/A-suozhang/MixDQ
  Model : https://huggingface.co/nics-efc/MixDQ

Backend model:
  stabilityai/sdxl-turbo  —  1-step, CFG-free (guidance_scale=0.0), 512×512
  Quantization applied to the UNet via pipe.quantize_unet(w_bit, a_bit, bos).

Supported quantization configs:
  W8A8  (default)  — ~2× weight compression, near-lossless quality
  W4A8             — ~4× weight compression, minor quality trade-off

Requirements:
  pip install -i https://pypi.org/simple/ mixdq-extension
  Note: mixdq-extension officially supports Python 3.8–3.10.
  For Python ≥ 3.11 you must build from source:
    https://github.com/A-suozhang/MixDQ

Usage:
    # Quick test with default prompts (W8A8)
    python chameleon_lcm_generation.py --num-samples 100

    # COCO evaluation — W8A8
    python chameleon_lcm_generation.py \\
        --coco-captions $CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json \\
        --num-samples 5000

    # W4A8 (more aggressive compression)
    python chameleon_lcm_generation.py \\
        --coco-captions /path/to/captions_val2014.json \\
        --w-bit 4 --num-samples 5000

    # Resume from index 1000 (already-generated files are skipped automatically)
    python chameleon_lcm_generation.py \\
        --coco-captions /path/to/captions_val2014.json \\
        --num-samples 5000 --start-idx 1000
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import os as _os
# Enable expandable memory segments before any CUDA initialisation.
# Reduces allocator fragmentation that causes spurious OOMs at large batch sizes.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import DiffusionPipeline
import os
import json
import random
import argparse
import sys as _sys
from tqdm import tqdm
from typing import List, Optional, Union

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

DEFAULT_BASE_MODEL      = "stabilityai/sdxl-turbo"
DEFAULT_CUSTOM_PIPELINE = "nics-efc/MixDQ"
# Chameleon-LCM (--chameleon-weights) writes here; the plain MixDQ baseline
# (no flag) is routed by run_chameleon_lcm.sh to the mixdq-lcm output dir.
DEFAULT_OUTPUT          = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "chameleon-lcm")

DEFAULT_STEPS    = 1
DEFAULT_GUIDANCE = 0.0
DEFAULT_HEIGHT   = 512
DEFAULT_WIDTH    = 512

DEFAULT_W_BIT = 8
DEFAULT_A_BIT = 8
DEFAULT_BOS   = True


# ── Auto batch-size helper ─────────────────────────────────────────────────────

def _auto_batch_size_for_images(pipe, height: int, width: int,
                                 num_inference_steps: int,
                                 guidance_scale: float,
                                 bos: bool = False,
                                 w_bit: int = 8) -> int:
    """
    Two-point VRAM calibration to find the largest image batch that fits in
    60% of available GPU memory.

    Measures peak VRAM at N=1 and N=2, isolating the fixed model overhead from
    the per-image marginal cost.  empty_cache() is called between passes to
    flush PyTorch's reserved-but-unallocated pool, which would otherwise cause
    peak(2) to be measured on top of peak(1)'s residual memory and
    underestimate the per-image cost.

    Returns 1 on CPU or if calibration fails.  The generation loop will
    further halve the batch on OOM.

    BAQ batch constraint: when BAQ is *actually* applied in the forward pass
    (i.e. W8A8 + bos=True), only position 0 of the batch gets the pre-computed
    cross-attention output; positions 1+ decode into noise, so we force
    batch_size=1.  The W4A8 fake-quant path silently disables BAQ (the
    shipped bos_pre_computed was calibrated against W8 weights), so batching
    is safe regardless of bos.
    """
    if not torch.cuda.is_available():
        return 1

    if bos and w_bit == 8:
        print("\nBOS-Aware Quantization enabled → forcing batch_size=1 "
              "(BAQ only applies to position 0 of the batch; positions 1+ "
              "decode into noise). Pass --no-bos to enable batching.")
        return 1

    SAFETY       = 0.60
    CALIB_PROMPT = "a cat sitting on a chair"

    def _measure(n: int) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            pipe(
                prompt=[CALIB_PROMPT] * n,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=[torch.Generator(device=DEVICE).manual_seed(i) for i in range(n)],
            )
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    print("\nAuto-sizing batch (two-point calibration) ...")
    try:
        total = torch.cuda.mem_get_info()[1]

        peak1 = _measure(1)

        try:
            peak2 = _measure(2)
            latent_per_image = peak2 - peak1
            model_overhead   = 2 * peak1 - peak2

            if latent_per_image <= 0:
                batch = max(1, int(total * SAFETY / peak1))
            else:
                budget = total * SAFETY - model_overhead
                batch  = max(1, int(budget / latent_per_image))

            print(f"  Fixed overhead  : {model_overhead    // 1024**2} MB")
            print(f"  Per-image cost  : {latent_per_image  // 1024**2} MB")
        except Exception as exc2:
            # Two-point failed (e.g. BOS-aware quantization dtype issue with batch > 1)
            # Fall back to single-point conservative estimate
            print(f"  Batch-2 calibration failed ({exc2})")
            print(f"  Falling back to single-point estimate")
            batch = max(1, int(total * SAFETY / peak1))

        print(f"  Total VRAM      : {total  // 1024**2} MB")
        print(f"  Auto batch size : {batch}  "
              f"(budget {int(total * SAFETY) // 1024**2} MB @ {int(SAFETY*100)}% safety)")
        return batch

    except Exception as exc:
        print(f"  Calibration failed ({exc}), using batch_size=1")
        return 1


# ── Generator class ────────────────────────────────────────────────────────────

class ChameleonLCMGenerator:
    """
    Image generator using MixDQ-quantized SDXL Turbo.

    Loads stabilityai/sdxl-turbo with the MixDQ custom diffusers pipeline
    (nics-efc/MixDQ), then calls pipe.quantize_unet() to apply mixed-precision
    PTQ to the UNet in-place.

    Key differences from FP16 baselines:
      - W8A8: ~2× weight memory reduction, ~1.45× latency speedup (RTX 4080)
      - W4A8: ~4× weight memory reduction, minor quality trade-off
      - 1 inference step (vs 12–50 for non-distilled models)
      - guidance_scale=0.0 (CFG-free)
    """

    def __init__(
        self,
        base_model:      str         = DEFAULT_BASE_MODEL,
        custom_pipeline: str         = DEFAULT_CUSTOM_PIPELINE,
        device:          str         = DEVICE,
        dtype:           torch.dtype = DTYPE,
        w_bit:           int         = DEFAULT_W_BIT,
        a_bit:           int         = DEFAULT_A_BIT,
        bos:             bool        = DEFAULT_BOS,
        cuda_graph:      bool        = False,
        w_config:        str         = None,
        a_config:        str         = None,
        chameleon_weights: bool      = False,
    ):
        self.base_model      = base_model
        self.custom_pipeline = custom_pipeline
        self.device          = device
        self.dtype           = dtype
        self.w_bit           = w_bit
        self.a_bit           = a_bit
        self.bos             = bos
        self.cuda_graph      = cuda_graph
        self.w_config        = w_config
        self.a_config        = a_config
        self.chameleon_weights = chameleon_weights
        self.pipe            = None

    def setup_pipeline(self):
        """Load SDXL Turbo with MixDQ custom pipeline and apply UNet quantization."""
        print(f"Loading {self.base_model} with MixDQ pipeline ({self.custom_pipeline})...")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.base_model,
            custom_pipeline=self.custom_pipeline,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        label = f"W{self.w_bit}A{self.a_bit}"
        if self.chameleon_weights:
            label += " (Chameleon adaptive weight palette)"
        if self.w_config:
            label += f" (w_config={self.w_config})"
        if self.a_config:
            label += f" (a_config={self.a_config})"
        print(f"Quantizing UNet: {label}, BOS={self.bos}...")
        try:
            # W4A8 runs as fake-quant via the patched pipeline.py: symmetric
            # weight round-to-[-7,7] using delta_list[1] from the shipped
            # quant_para_wsym_fp16.pt, plus per-tensor A8 fake-quant on the
            # input at forward time.  The public checkpoint already contains
            # W4 scales — only the pipeline's hard-coded W8 gate had to be
            # relaxed.  W8A8 still takes the accelerated qlinear/qconv2d path.
            self.pipe.quantize_unet(
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                bos=self.bos,
                w_config_yaml=self.w_config,
                a_config_yaml=self.a_config,
            )
            print(f"  UNet quantized ({label})")
        except AttributeError:
            raise RuntimeError(
                "pipe.quantize_unet() not found.\n"
                "Install the MixDQ extension:\n"
                "  pip install -i https://pypi.org/simple/ mixdq-extension\n"
                "Note: mixdq-extension officially supports Python 3.8–3.10.\n"
                "For Python ≥ 3.11, build from source: "
                "https://github.com/A-suozhang/MixDQ"
            )
        except TypeError as e:
            if "unexpected keyword argument 'w_config_yaml'" in str(e):
                raise RuntimeError(
                    "The installed MixDQ pipeline.py is missing the W4A8 patch. "
                    "Expected keyword arg `w_config_yaml` on quantize_unet(). "
                    "Clear the HF modules cache and reload, or re-apply the "
                    "patch at:\n  ~/.cache/huggingface/modules/diffusers_modules/"
                    "local/nics-efc--MixDQ/*/pipeline.py"
                )
            raise RuntimeError(f"UNet quantization failed: {e}")
        except Exception as e:
            raise RuntimeError(f"UNet quantization failed: {e}")

        # SDXL VAE is numerically unstable in fp16 — keep parameters in fp32
        # and patch decode()/forward() to run outside any autocast context so
        # neither gets silently downcast to fp16 when the caller uses autocast.
        # Diffusers SDXL pipeline invokes vae.decode() directly, bypassing forward().
        self.pipe.vae.to(torch.float32)

        def _make_fp32_wrapper(orig_fn):
            def _wrapped(*args, **kwargs):
                args = tuple(a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                             for a in args)
                kwargs = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                          for k, v in kwargs.items()}
                with torch.autocast("cuda", enabled=False):
                    return orig_fn(*args, **kwargs)
            return _wrapped

        self.pipe.vae.forward = _make_fp32_wrapper(self.pipe.vae.forward)
        self.pipe.vae.decode  = _make_fp32_wrapper(self.pipe.vae.decode)
        self.pipe.vae.encode  = _make_fp32_wrapper(self.pipe.vae.encode)

        # Suppress the per-batch denoising progress bar; our outer tqdm tracks overall progress.
        self.pipe.set_progress_bar_config(disable=True)

        if self.cuda_graph:
            try:
                self.pipe.set_cuda_graph(run_pipeline=True)
                print("CUDA graph acceleration enabled")
            except Exception as e:
                print(f"CUDA graph not available: {e}")

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            print(f"Pipeline ready on {torch.cuda.get_device_name(dev_idx)} (cuda:{dev_idx})")
        else:
            print("Pipeline ready on CPU")

        return self.pipe

    def generate_images(
        self,
        prompts:             List[str],
        output_dir:          str,
        num_inference_steps: int              = DEFAULT_STEPS,
        guidance_scale:      float            = DEFAULT_GUIDANCE,
        height:              int              = DEFAULT_HEIGHT,
        width:               int              = DEFAULT_WIDTH,
        batch_size:          Union[int, str]  = "auto",
        start_idx:           int              = 0,
    ):
        """
        Generate images from text prompts and save as PNGs.

        batch_size: images per forward pass.
                    "auto" (default) runs a two-point VRAM calibration and picks
                    the largest batch that fits in 60% of free VRAM.  On OOM the
                    batch is halved and retried automatically.
        start_idx:  global index offset for file naming / resumption.
                    Already-present files are always skipped regardless of start_idx.
        """
        if self.pipe is None:
            self.setup_pipeline()

        os.makedirs(output_dir, exist_ok=True)

        # ── Resolve batch size ────────────────────────────────────────────────
        if batch_size == "auto":
            batch_size = _auto_batch_size_for_images(
                self.pipe, height, width, num_inference_steps, guidance_scale,
                bos=self.bos, w_bit=self.w_bit,
            )

        print(f"\nGenerating {len(prompts)} images (start_idx={start_idx}, "
              f"batch_size={batch_size})...")
        print(f"  Quantization : W{self.w_bit}A{self.a_bit}"
              f"  |  Steps: {num_inference_steps}"
              f"  |  CFG: {guidance_scale}"
              f"  |  Size: {width}×{height}")

        tracker = PerfTracker(
            label=(f"chameleon_lcm_w{self.w_bit}a{self.a_bit}"
                   if self.chameleon_weights
                   else f"mixdq_lcm_w{self.w_bit}a{self.a_bit}"),
            device=self.device,
        )
        tracker.reset()

        # ── Adaptive generation loop ──────────────────────────────────────────
        # Halves the batch on OOM and retries the same chunk rather than
        # falling through to single-image generation.
        current_bs = batch_size
        pos        = 0
        pbar       = tqdm(total=len(prompts), desc="Generating")

        while pos < len(prompts):
            end         = min(pos + current_bs, len(prompts))
            chunk_idx   = list(range(pos, end))
            chunk_prmpts = [prompts[i] for i in chunk_idx]
            chunk_paths  = [
                os.path.join(output_dir, f"{start_idx + i:05d}.png")
                for i in chunk_idx
            ]
            chunk_seeds  = [start_idx + i for i in chunk_idx]

            # Skip images that already exist (resumption support)
            missing = [j for j, p in enumerate(chunk_paths) if not os.path.exists(p)]
            if not missing:
                pbar.update(len(chunk_idx))
                pos = end
                continue

            m_prompts = [chunk_prmpts[j] for j in missing]
            m_paths   = [chunk_paths[j]  for j in missing]
            m_seeds   = [chunk_seeds[j]  for j in missing]

            try:
                generators = [
                    torch.Generator(device=self.device).manual_seed(s) for s in m_seeds
                ]
                tracker.batch_start()
                with torch.autocast("cuda", dtype=torch.float16):
                    result = self.pipe(
                        prompt=m_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        generator=generators,
                    )
                tracker.batch_end(len(m_prompts))
                for image, path in zip(result.images, m_paths):
                    image.save(path)
                pbar.update(len(chunk_idx))
                pos = end                          # advance only on success

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_bs > 1:
                    new_bs = max(1, current_bs // 2)
                    pbar.write(f"OOM at batch={current_bs} → retrying with {new_bs}")
                    current_bs = new_bs
                    # pos unchanged — retry same chunk with smaller batch
                else:
                    pbar.write(f"OOM at batch=1 — skipping {m_paths[0]}")
                    pbar.update(1)
                    pos += 1

            except Exception as exc:
                pbar.write(f"Error (batch={current_bs}): {exc} — skipping {len(missing)} images")
                pbar.update(len(chunk_idx))
                pos = end

        pbar.close()
        tracker.finish()
        tracker.dump_json(os.path.join(output_dir, "perf.json"))
        print(f"Saved images to {output_dir}")

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()


# ── Prompt utilities ───────────────────────────────────────────────────────────

def load_coco_captions(
    json_path:   str,
    num_samples: int,
    start_idx:   int  = 0,
    shuffle:     bool = True,
) -> List[str]:
    """Load captions from COCO annotations JSON (same shuffle seed as other baselines)."""
    print(f"Loading COCO captions from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    captions = [ann['caption'].strip() for ann in data['annotations']]
    print(f"Loaded {len(captions)} captions")

    if shuffle:
        random.seed(42)
        random.shuffle(captions)

    end_idx  = min(start_idx + num_samples, len(captions))
    selected = captions[start_idx:end_idx]
    print(f"Using captions {start_idx}–{end_idx} ({len(selected)} total)")
    return selected


def get_default_prompts(num_samples: int) -> List[str]:
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
    while len(prompts) < num_samples:
        prompts.extend(base)
    return prompts[:num_samples]


def load_prompts_from_file(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# ── Main generation function ───────────────────────────────────────────────────

def run_chameleon_lcm_generation(
    output_dir:          str,
    num_samples:         int              = 100,
    start_idx:           int              = 0,
    coco_captions:       Optional[str]    = None,
    prompt_file:         Optional[str]    = None,
    num_inference_steps: int              = DEFAULT_STEPS,
    guidance_scale:      float            = DEFAULT_GUIDANCE,
    height:              int              = DEFAULT_HEIGHT,
    width:               int              = DEFAULT_WIDTH,
    w_bit:               int              = DEFAULT_W_BIT,
    a_bit:               int              = DEFAULT_A_BIT,
    bos:                 bool             = DEFAULT_BOS,
    cuda_graph:          bool             = False,
    batch_size:          Union[int, str]  = "auto",
    base_model:          str              = DEFAULT_BASE_MODEL,
    device:              str              = DEVICE,
    w_config:            Optional[str]    = None,
    a_config:            Optional[str]    = None,
    chameleon_weights:   bool             = False,
) -> dict:
    """Run MixDQ generation. Returns statistics dictionary."""

    if coco_captions:
        prompts = load_coco_captions(coco_captions, num_samples, start_idx)
    elif prompt_file:
        all_prompts = load_prompts_from_file(prompt_file)
        end     = min(start_idx + num_samples, len(all_prompts))
        prompts = all_prompts[start_idx:end]
        print(f"Using {len(prompts)} prompts from file (indices {start_idx}–{end})")
    else:
        prompts = get_default_prompts(num_samples)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("MIXDQ GENERATION (Quantized Few-Step Diffusion)")
    print(f"{'='*60}")
    print(f"Base model          : {base_model}")
    print(f"MixDQ pipeline      : {DEFAULT_CUSTOM_PIPELINE}")
    print(f"Quantization        : W{w_bit}A{a_bit}, BOS={bos}"
          f"{'  | Chameleon adaptive weight palette' if chameleon_weights else ''}")
    print(f"Output              : {output_dir}")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Inference steps     : {num_inference_steps}  (SDXL Turbo: 1 step typical)")
    print(f"Guidance scale      : {guidance_scale}  (0.0 = CFG-free)")
    print(f"Image size          : {width}×{height}")
    print(f"Batch size          : {batch_size}")
    print(f"CUDA graph          : {cuda_graph}")
    print(f"Device              : {device}")
    print(f"{'='*60}\n")

    generator = ChameleonLCMGenerator(
        base_model=base_model,
        device=device,
        w_bit=w_bit,
        a_bit=a_bit,
        bos=bos,
        cuda_graph=cuda_graph,
        w_config=w_config,
        a_config=a_config,
        chameleon_weights=chameleon_weights,
    )
    generator.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        batch_size=batch_size,
        start_idx=start_idx,
    )

    stats = {
        "base_model":          base_model,
        "custom_pipeline":     DEFAULT_CUSTOM_PIPELINE,
        "quantization":        f"W{w_bit}A{a_bit}",
        "chameleon_weights":   chameleon_weights,
        "w_config":            w_config,
        "a_config":            a_config,
        "bos":                 bos,
        "cuda_graph":          cuda_graph,
        "num_samples":         len(prompts),
        "start_idx":           start_idx,
        "batch_size":          str(batch_size),
        "num_inference_steps": num_inference_steps,
        "guidance_scale":      guidance_scale,
        "image_height":        height,
        "image_width":         width,
        "output_dir":          output_dir,
    }

    stats_file = os.path.join(output_dir, "mixdq_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    generator.cleanup()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MixDQ: mixed-precision quantized SDXL Turbo generation"
    )

    parser.add_argument("--output-dir",   type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples",  type=int, default=100)
    parser.add_argument("--start-idx",    type=int, default=0,
                        help="Caption slice offset and seed base (for resumption). "
                             "Already-present PNG files are always skipped automatically.")

    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON (captions_val2014.json)")
    parser.add_argument("--prompt-file",   type=str, default=None)
    parser.add_argument("--base-model",    type=str, default=DEFAULT_BASE_MODEL)

    parser.add_argument("--w-bit", type=int, default=DEFAULT_W_BIT, choices=[4, 8],
                        help="Weight bit-width. 4 uses the W4A8 fake-quant path "
                             "(weights round-to-[-7,7] using shipped W4 scales, "
                             "A8 fake-quant on activations); 8 uses the "
                             "accelerated qlinear/qconv2d kernels.")
    parser.add_argument("--a-bit", type=int, default=DEFAULT_A_BIT, choices=[8],
                        help="Activation bit-width. Only A8 is supported — "
                             "A4 would require new pre-computed scales.")
    parser.add_argument("--w-config", type=str, default=None,
                        help="Path to a YAML file mapping module names to "
                             "per-layer weight bit-widths (e.g. {4,8}). "
                             "Overrides --w-bit for listed modules. Useful for "
                             "reproducing MixDQ's mixed-precision allocations.")
    parser.add_argument("--a-config", type=str, default=None,
                        help="Same as --w-config but for activations.")
    parser.add_argument("--chameleon-weights", action="store_true",
                        help="Replace MixDQ's symmetric weight fold with "
                             "Chameleon's adaptive palette (per-output-channel "
                             "SNR selection @ W8; per-layer (group_size, format) "
                             "search @ W4). Activations stay on MixDQ's calibrated "
                             "static-scale path. Sets MIXDQ_CHAMELEON_WEIGHTS=1.")
    parser.add_argument("--no-bos",    action="store_true",
                        help="Disable BOS-aware quantization (default: enabled)")
    parser.add_argument("--cuda-graph", action="store_true")

    parser.add_argument("--num-inference-steps", type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale",       type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--height",               type=int,   default=DEFAULT_HEIGHT)
    parser.add_argument("--width",                type=int,   default=DEFAULT_WIDTH)

    parser.add_argument(
        "--batch-size",
        type=lambda x: int(x) if x != "auto" else "auto",
        default="auto",
        help="Images per batch. 'auto' (default) runs a two-point VRAM calibration "
             "and picks the largest batch that fits in 60%% of free VRAM.",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Must be set before the MixDQ custom pipeline module is imported (its
    # module-level MIXDQ_CHAMELEON_WEIGHTS flag is read at import time, which
    # happens inside setup_pipeline's from_pretrained call).
    if args.chameleon_weights:
        os.environ["MIXDQ_CHAMELEON_WEIGHTS"] = "1"
        os.environ.setdefault(
            "MIXDQ_CHAMELEON_SRC",
            os.path.dirname(os.path.abspath(__file__)),
        )

    stats = run_chameleon_lcm_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        prompt_file=args.prompt_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        bos=not args.no_bos,
        cuda_graph=args.cuda_graph,
        batch_size=args.batch_size,
        base_model=args.base_model,
        device=args.device,
        w_config=args.w_config,
        a_config=args.a_config,
        chameleon_weights=args.chameleon_weights,
    )

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model              : {stats['base_model']}")
    print(f"Quantization       : {stats['quantization']}")
    print(f"Images generated   : {stats['num_samples']}")
    print(f"Batch size         : {stats['batch_size']}")
    print(f"Steps used         : {stats['num_inference_steps']}")
    print(f"Output directory   : {stats['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
