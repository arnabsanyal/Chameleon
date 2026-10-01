"""
PixArt-alpha Baseline Generation for Text-to-Image Synthesis

PixArt-alpha is a Diffusion Transformer (DiT) model for text-to-image generation.
Architecture: DiT backbone + T5 text encoder + cross-attention text conditioning.
Reference: Chen et al., "PixArt-alpha: Fast Training of Diffusion Transformer for
           Photorealistic Text-to-Image Synthesis", arXiv:2310.00426

Usage:
    python pixart_generation.py --num-samples 100 --output-dir $CHAMELEON_OUTPUT_ROOT/pixart-alpha-dit
    python pixart_generation.py --coco-captions /path/to/captions.json --num-samples 5000
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
from diffusers import PixArtAlphaPipeline
import os
import json
import random
import argparse
import sys as _sys
from tqdm import tqdm
from typing import List, Optional

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

# Available PixArt-alpha model variants:
#   PixArt-alpha/PixArt-XL-2-1024-MS  — 1024×1024, multi-scale (recommended)
#   PixArt-alpha/PixArt-XL-2-512x512  — 512×512
#   PixArt-alpha/PixArt-XL-2-256x256  — 256×256
DEFAULT_MODEL_ID  = "PixArt-alpha/PixArt-XL-2-1024-MS"
DEFAULT_OUTPUT    = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "pixart-alpha-dit")


# ── Generator class ────────────────────────────────────────────────────────────

class PixArtGenerator:
    """
    Image generator using PixArt-alpha (DiT-based T2I model).

    Key differences from UNet-based models (SDXL):
      - DiT backbone: patchified latent tokens processed by Transformer blocks
      - T5 text encoder (T5-XXL, max 300 tokens) instead of CLIP
      - Cross-attention injects text into every DiT block
      - adaLN-single: shared time-conditioning MLP across all blocks
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device:   str = DEVICE,
        dtype:    torch.dtype = DTYPE,
    ):
        self.model_id = model_id
        self.device   = device
        self.dtype    = dtype
        self.pipe     = None

    def setup_pipeline(self):
        """Load PixArt-alpha pipeline."""
        print(f"Loading PixArt-alpha from {self.model_id}...")
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        # xformers for memory efficiency
        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

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
        num_inference_steps: int   = 20,
        guidance_scale:      float = 4.5,
        batch_size:          int   = 1,
        start_idx:           int   = 0,
    ):
        """
        Generate images from text prompts and save them as PNGs.

        Args:
            prompts:             List of text prompts.
            output_dir:          Directory to save images.
            num_inference_steps: Denoising steps (PixArt-alpha is efficient at 20).
            guidance_scale:      CFG scale (paper uses 4.5 for 1024×1024).
            batch_size:          Images per forward pass.
            start_idx:           Global index offset for file naming.
        """
        if self.pipe is None:
            self.setup_pipeline()

        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating {len(prompts)} images (starting at index {start_idx})...")
        print(f"  Steps: {num_inference_steps}  |  CFG: {guidance_scale}  |  Batch: {batch_size}")

        tracker = PerfTracker(label="pixart_alpha_fp16", device=self.device)
        tracker.reset()

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
                # Fallback to single-image generation
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

def run_pixart_generation(
    output_dir:          str,
    num_samples:         int   = 100,
    start_idx:           int   = 0,
    coco_captions:       Optional[str] = None,
    prompt_file:         Optional[str] = None,
    num_inference_steps: int   = 20,
    guidance_scale:      float = 4.5,
    batch_size:          int   = 1,
    model_id:            str   = DEFAULT_MODEL_ID,
    device:              str   = DEVICE,
) -> dict:
    """Run PixArt-alpha generation. Returns basic statistics."""

    # Load prompts
    if coco_captions:
        prompts = load_coco_captions(coco_captions, num_samples, start_idx)
    elif prompt_file:
        all_prompts = load_prompts_from_file(prompt_file)
        end = min(start_idx + num_samples, len(all_prompts))
        prompts = all_prompts[start_idx:end]
        print(f"Using {len(prompts)} prompts from file (indices {start_idx}–{end})")
    else:
        prompts = get_default_prompts(num_samples)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PIXART-ALPHA GENERATION")
    print(f"{'='*60}")
    print(f"Model               : {model_id}")
    print(f"Output              : {output_dir}")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Inference steps     : {num_inference_steps}")
    print(f"Guidance scale      : {guidance_scale}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")
    print(f"{'='*60}\n")

    generator = PixArtGenerator(model_id=model_id, device=device)
    generator.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        start_idx=start_idx,
    )

    stats = {
        "model_id":            model_id,
        "num_samples":         len(prompts),
        "start_idx":           start_idx,
        "num_inference_steps": num_inference_steps,
        "guidance_scale":      guidance_scale,
        "output_dir":          output_dir,
    }

    stats_file = os.path.join(output_dir, "pixart_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    generator.cleanup()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PixArt-alpha: DiT-based text-to-image generation baseline"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Output directory for generated images",
    )
    parser.add_argument("--num-samples",  type=int, default=100,
                        help="Number of images to generate")
    parser.add_argument("--start-idx",    type=int, default=0,
                        help="Starting index for image numbering / caption slice")

    # Prompt source
    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON (captions_val2014.json)")
    parser.add_argument("--prompt-file",   type=str, default=None,
                        help="Path to text file with prompts (one per line)")

    # Model
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID for PixArt-alpha "
             "(default: PixArt-alpha/PixArt-XL-2-1024-MS)",
    )

    # Generation parameters
    parser.add_argument("--num-inference-steps", type=int,   default=20,
                        help="Number of denoising steps (default: 20)")
    parser.add_argument("--guidance-scale",       type=float, default=4.5,
                        help="CFG guidance scale (paper uses 4.5 for 1024px, default: 4.5)")
    parser.add_argument("--batch-size",           type=int,   default=1,
                        help="Images per batch (default: 1)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    stats = run_pixart_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        prompt_file=args.prompt_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        model_id=args.model_id,
        device=args.device,
    )

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Model              : {stats['model_id']}")
    print(f"Images generated   : {stats['num_samples']}")
    print(f"Output directory   : {stats['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
