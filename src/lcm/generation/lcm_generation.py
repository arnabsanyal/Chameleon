"""
Few-Step FP16 Baseline Generation (SDXL-Turbo) for Text-to-Image Synthesis

The few-step FP16 reference for the LCM-family table.  We use SDXL-Turbo — the
same base model used by the MixDQ and Chameleon (few-step) entries — at a single,
CFG-free denoising step, so the baseline and the quantized methods are a true
apples-to-apples comparison (same base, 1 step, 512x512).

(Previously this baseline used LCM Dreamshaper v7 at 12 steps; it was migrated to
SDXL-Turbo so the whole few-step family shares one base model.  The "lcm" name is
retained for continuity with the paper's LCM-family table / perf labels.)

Architecture:
  - Backbone    : SDXL-Turbo UNet (adversarial-distilled, 1-step CFG-free)
  - Scheduler   : EulerAncestralDiscrete (Turbo schedule)
  - Text encoders: dual CLIP (SDXL)

Model:
  stabilityai/sdxl-turbo
  https://huggingface.co/stabilityai/sdxl-turbo

Key parameters vs other baselines:
  - num_inference_steps : 1   (vs 50 for SDXL, 20 for PixArt-alpha)
  - guidance_scale      : 0.0 (SDXL-Turbo is CFG-free)
  - Image size          : 512x512

Usage:
    # Quick test with default prompts
    python lcm_generation.py --num-samples 100

    # COCO captions evaluation (auto batch sizing)
    python lcm_generation.py --coco-captions /path/to/captions_val2014.json \
                             --num-samples 5000

    # Resume from a checkpoint (already-generated files are skipped automatically)
    python lcm_generation.py --coco-captions /path/to/captions.json \
                             --num-samples 5000 --start-idx 2000
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import os as _os
# Enable expandable memory segments before any CUDA initialisation.
# Reduces allocator fragmentation that causes spurious OOMs at large batch sizes.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import AutoPipelineForText2Image
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

# SDXL-Turbo — same base used by the MixDQ and Chameleon few-step entries.
DEFAULT_MODEL_ID = "stabilityai/sdxl-turbo"
DEFAULT_OUTPUT   = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "lcm")

# SDXL-Turbo native operating point: 1 step, CFG-free, 512x512.
DEFAULT_STEPS            = 1
DEFAULT_GUIDANCE_SCALE   = 0.0
DEFAULT_IMAGE_HEIGHT   = 512
DEFAULT_IMAGE_WIDTH    = 512


# ── Auto batch-size helper ─────────────────────────────────────────────────────

def _auto_batch_size_for_images(pipe, height: int, width: int,
                                 num_inference_steps: int,
                                 guidance_scale: float) -> int:
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
    """
    if not torch.cuda.is_available():
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

class LCMGenerator:
    """
    Few-step FP16 baseline generator using SDXL-Turbo.

    SDXL-Turbo (Adversarial Diffusion Distillation) produces high-quality images
    in a single, CFG-free denoising step — the few-step FP16 reference for the
    LCM-family table, sharing a base model with the MixDQ and Chameleon entries.

    Key parameters vs standard diffusion baselines:
      - 1 denoising step (vs 50 for SDXL, 20 for PixArt-alpha)
      - guidance_scale = 0.0 (CFG-free)
      - 512x512 native resolution
    """

    def __init__(
        self,
        model_id: str         = DEFAULT_MODEL_ID,
        device:   str         = DEVICE,
        dtype:    torch.dtype = DTYPE,
    ):
        self.model_id = model_id
        self.device   = device
        self.dtype    = dtype
        self.pipe     = None

    def setup_pipeline(self):
        """Load SDXL-Turbo pipeline."""
        print(f"Loading SDXL-Turbo from {self.model_id}...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to(self.device)

        # SDXL(-Turbo) VAE is numerically unstable in fp16 — keep its parameters
        # in fp32 and wrap decode/encode/forward to cast inputs and disable
        # autocast so they are not silently downcast under the outer autocast.
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

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            print(f"Pipeline loaded on {torch.cuda.get_device_name(dev_idx)} (cuda:{dev_idx})")
        else:
            print("Pipeline loaded on CPU")

        return self.pipe

    def generate_images(
        self,
        prompts:             List[str],
        output_dir:          str,
        num_inference_steps: int              = DEFAULT_STEPS,
        guidance_scale:      float            = DEFAULT_GUIDANCE_SCALE,
        height:              int              = DEFAULT_IMAGE_HEIGHT,
        width:               int              = DEFAULT_IMAGE_WIDTH,
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
            )

        print(f"\nGenerating {len(prompts)} images (start_idx={start_idx}, "
              f"batch_size={batch_size})...")
        print(f"  Steps: {num_inference_steps}  |  CFG: {guidance_scale}"
              f"  |  Size: {width}×{height}")

        tracker = PerfTracker(label="lcm_fp16", device=self.device)
        tracker.reset()

        # ── Adaptive generation loop ──────────────────────────────────────────
        # Halves the batch on OOM and retries the same chunk rather than
        # falling through to single-image generation.
        current_bs = batch_size
        pos        = 0
        pbar       = tqdm(total=len(prompts), desc="Generating")

        while pos < len(prompts):
            end          = min(pos + current_bs, len(prompts))
            chunk_idx    = list(range(pos, end))
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
                        output_type="pil",
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
    """Default prompts for quick testing."""
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

def run_lcm_generation(
    output_dir:          str,
    num_samples:         int              = 100,
    start_idx:           int              = 0,
    coco_captions:       Optional[str]    = None,
    prompt_file:         Optional[str]    = None,
    num_inference_steps: int              = DEFAULT_STEPS,
    guidance_scale:      float            = DEFAULT_GUIDANCE_SCALE,
    height:              int              = DEFAULT_IMAGE_HEIGHT,
    width:               int              = DEFAULT_IMAGE_WIDTH,
    batch_size:          Union[int, str]  = "auto",
    model_id:            str              = DEFAULT_MODEL_ID,
    device:              str              = DEVICE,
) -> dict:
    """Run LCM generation. Returns statistics dictionary."""

    # Load prompts
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
    print("FEW-STEP FP16 BASELINE (SDXL-Turbo)")
    print(f"{'='*60}")
    print(f"Model               : {model_id}")
    print(f"Output              : {output_dir}")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Inference steps     : {num_inference_steps}  (SDXL-Turbo: 1 typical)")
    print(f"Guidance scale      : {guidance_scale}  (0.0 = CFG-free)")
    print(f"Image size          : {width}×{height}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")
    print(f"{'='*60}\n")

    generator = LCMGenerator(model_id=model_id, device=device)
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
        "model_id":            model_id,
        "num_samples":         len(prompts),
        "start_idx":           start_idx,
        "batch_size":          str(batch_size),
        "num_inference_steps": num_inference_steps,
        "guidance_scale":      guidance_scale,
        "image_height":        height,
        "image_width":         width,
        "output_dir":          output_dir,
    }

    stats_file = os.path.join(output_dir, "lcm_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    generator.cleanup()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Few-step FP16 baseline (SDXL-Turbo, 1-step CFG-free) text-to-image generation"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Output directory for generated images",
    )
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of images to generate")
    parser.add_argument("--start-idx",  type=int, default=0,
                        help="Starting index for image numbering / caption slice. "
                             "Already-present PNG files are always skipped automatically.")

    # Prompt source
    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON (captions_val2014.json)")
    parser.add_argument("--prompt-file",   type=str, default=None,
                        help="Path to text file with prompts (one per line)")

    # Model
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: stabilityai/sdxl-turbo)",
    )

    # Generation parameters
    parser.add_argument(
        "--num-inference-steps", type=int, default=DEFAULT_STEPS,
        help=f"Denoising steps (default: {DEFAULT_STEPS}; SDXL-Turbo native is 1)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE_SCALE}; SDXL-Turbo is CFG-free)",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
        help=f"Output image height in pixels (default: {DEFAULT_IMAGE_HEIGHT})",
    )
    parser.add_argument(
        "--width",  type=int, default=DEFAULT_IMAGE_WIDTH,
        help=f"Output image width in pixels (default: {DEFAULT_IMAGE_WIDTH})",
    )
    parser.add_argument(
        "--batch-size",
        type=lambda x: int(x) if x != "auto" else "auto",
        default="auto",
        help="Images per batch. 'auto' (default) runs a two-point VRAM calibration "
             "and picks the largest batch that fits in 60%% of free VRAM.",
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    stats = run_lcm_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        prompt_file=args.prompt_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        model_id=args.model_id,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model              : {stats['model_id']}")
    print(f"Images generated   : {stats['num_samples']}")
    print(f"Batch size         : {stats['batch_size']}")
    print(f"Steps used         : {stats['num_inference_steps']}")
    print(f"Output directory   : {stats['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
