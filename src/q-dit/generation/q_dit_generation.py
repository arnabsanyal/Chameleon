"""
Q-DiT Generation: Post-Training Quantized PixArt-alpha Image Generation

Q-DiT applies two PTQ techniques to PixArt-alpha's DiT transformer (§5):
  1. Group weight quantization  — offline, per-layer group size (default: 128)
  2. Sample-wise dynamic activation quantization — online, per forward pass

Reference: Chen et al., "Q-DiT: Accurate Post-Training Quantization for
           Diffusion Transformers", arXiv:2406.17343

Usage:
    # W8A8 (default, near-lossless per paper)
    python q_dit_generation.py --num-samples 100

    # W4A8
    python q_dit_generation.py --weight-bits 4 --act-bits 8 --num-samples 100

    # COCO evaluation
    python q_dit_generation.py --coco-captions /path/to/captions.json \\
                               --num-samples 5000 --weight-bits 8 --act-bits 8

    # Custom group size
    python q_dit_generation.py --group-size 64 --num-samples 100
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import os
import json
import random
import argparse
from typing import List, Optional

from q_dit_quant import create_q_dit_quantizer, DEFAULT_GROUP_SIZE


# ── Configuration ──────────────────────────────────────────────────────────────

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
DEFAULT_OUTPUT   = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "q-dit")


# ── Prompt utilities ───────────────────────────────────────────────────────────

def load_coco_captions(
    json_path:   str,
    num_samples: int,
    start_idx:   int  = 0,
    shuffle:     bool = True,
) -> List[str]:
    """Load COCO captions with the same shuffle seed used across all baselines."""
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

def run_q_dit_generation(
    output_dir:          str,
    num_samples:         int   = 100,
    start_idx:           int   = 0,
    coco_captions:       Optional[str] = None,
    prompt_file:         Optional[str] = None,
    num_inference_steps: int   = 20,
    guidance_scale:      float = 4.5,
    batch_size:          int   = 1,
    model_id:            str   = DEFAULT_MODEL_ID,
    weight_bits:         int   = 8,
    act_bits:            int   = 8,
    group_size:          int   = DEFAULT_GROUP_SIZE,
    device:              str   = DEVICE,
    verbose:             bool  = False,
) -> dict:
    """Run Q-DiT quantized generation. Returns statistics dictionary."""

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

    # Output subfolder encodes quantization config: w8a8_g128
    subdir     = f"w{weight_bits}a{act_bits}_g{group_size}"
    output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Q-DiT QUANTIZED GENERATION")
    print(f"{'='*60}")
    print(f"Model               : {model_id}")
    print(f"Quantization        : W{weight_bits}A{act_bits}, group size={group_size}")
    print(f"Output              : {output_dir}")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Inference steps     : {num_inference_steps}")
    print(f"Guidance scale      : {guidance_scale}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")
    print(f"{'='*60}\n")

    # Create quantizer, load model, inject quantization
    quantizer = create_q_dit_quantizer(
        model_id=model_id,
        weight_bits=weight_bits,
        act_bits=act_bits,
        group_size=group_size,
        device=device,
    )
    quantizer.load_model()
    quantizer.inject_quantization(verbose=verbose)

    # Generate images
    quantizer.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        start_idx=start_idx,
    )

    stats = quantizer.get_stats()
    stats.update({
        "num_samples":         len(prompts),
        "start_idx":           start_idx,
        "num_inference_steps": num_inference_steps,
        "guidance_scale":      guidance_scale,
        "output_dir":          output_dir,
    })

    stats_file = os.path.join(output_dir, "q_dit_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    quantizer.cleanup()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q-DiT: group-quantized PixArt-alpha image generation (PTQ)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Base output directory (a subdir w<W>a<A>_g<G> is appended)",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-idx",   type=int, default=0,
                        help="Starting index for image naming / caption slice")

    # Prompt source
    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON (captions_val2014.json)")
    parser.add_argument("--prompt-file",   type=str, default=None,
                        help="Text file with prompts (one per line)")

    # Model
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: PixArt-alpha/PixArt-XL-2-1024-MS)",
    )

    # Quantization — these are the key Q-DiT knobs
    parser.add_argument(
        "--weight-bits", type=int, default=8, choices=[4, 6, 8],
        help="Weight quantization bits (paper evaluates W6A8, W4A8; default: 8)",
    )
    parser.add_argument(
        "--act-bits", type=int, default=8, choices=[4, 6, 8],
        help="Activation quantization bits (default: 8)",
    )
    parser.add_argument(
        "--group-size", type=int, default=DEFAULT_GROUP_SIZE,
        help=f"Group size for input-channel quantization "
             f"(search space {{32,64,128,192,288}}, default: {DEFAULT_GROUP_SIZE})",
    )

    # Generation parameters
    parser.add_argument("--num-inference-steps", type=int,   default=20)
    parser.add_argument("--guidance-scale",       type=float, default=4.5)
    parser.add_argument("--batch-size",           type=int,   default=1)

    # Misc
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device",  type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    stats = run_q_dit_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        prompt_file=args.prompt_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        model_id=args.model_id,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        group_size=args.group_size,
        device=args.device,
        verbose=args.verbose,
    )

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Model         : {stats['model_id']}")
    print(f"Quantization  : W{stats['weight_bits']}A{stats['act_bits']}, group={stats['group_size']}")
    print(f"Quant layers  : {stats['num_quant_layers']}")
    print(f"Images        : {stats['num_samples']}")
    print(f"Output        : {stats['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
