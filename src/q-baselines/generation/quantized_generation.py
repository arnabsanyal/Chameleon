"""
Quantized Baseline Generation for Image Diffusion Models
Using Q-Diffusion and PTQ4DM quantization techniques for SDXL

This module provides a unified interface for generating images with
quantized diffusion models for comparison against FP16 baselines.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
import os
import argparse
import json
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm

from qdiffusion import QDiffusionQuantizer, create_qdiffusion_quantizer
from ptq4dm import PTQ4DMQuantizer, create_ptq4dm_quantizer


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
OUTPUT_DIR = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized")


def load_prompts_from_file(prompt_file: str) -> List[str]:
    """Load prompts from a text file (one prompt per line)"""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_coco_captions(annotations_file: str, num_samples: Optional[int] = None, shuffle: bool = True) -> List[str]:
    """Load captions from COCO annotations JSON file"""
    import random

    print(f"Loading COCO captions from {annotations_file}...")

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    captions = [ann['caption'].strip() for ann in data['annotations']]
    print(f"Loaded {len(captions)} captions")

    if shuffle:
        random.seed(42)
        random.shuffle(captions)

    if num_samples is not None:
        captions = captions[:num_samples]

    return captions


def generate_sample_prompts(num_samples: int = 100) -> List[str]:
    """Generate sample prompts for testing"""
    templates = [
        "A photo of {}",
        "An image of {}",
        "A high quality photograph of {}",
        "{} in natural lighting",
        "Professional photo of {}"
    ]

    subjects = [
        "a cat", "a dog", "a bird", "a flower", "a tree",
        "a mountain", "a beach", "a city", "a car", "a house",
        "a person", "a landscape", "a sunset", "a building", "a forest"
    ]

    prompts = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        subject = subjects[i % len(subjects)]
        prompts.append(template.format(subject))

    return prompts


def calculate_fid(real_images_dir: str, generated_images_dir: str) -> Optional[float]:
    """Calculate FID score using clean-fid library"""
    try:
        from cleanfid import fid

        print(f"Calculating FID between {real_images_dir} and {generated_images_dir}")
        score = fid.compute_fid(real_images_dir, generated_images_dir)
        print(f"FID Score: {score:.2f}")
        return score
    except ImportError:
        print("clean-fid not installed. Install with: pip install clean-fid")
        return None


def run_quantized_generation(
    method: str,
    prompts: List[str],
    output_dir: str,
    weight_bits: int = 8,
    act_bits: int = 8,
    num_timestep_buckets: int = 10,
    num_calibration_samples: int = 32,
    calibration_steps: int = 20,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    batch_size: Union[int, str] = "auto",
    start_idx: int = 0,
    device: str = DEVICE,
    disable_quant: bool = False,
):
    """
    Run quantized image generation with specified method.

    Args:
        method: "qdiffusion" or "ptq4dm"
        prompts: List of text prompts
        output_dir: Directory to save images
        weight_bits: Bit-width for weights
        act_bits: Bit-width for activations
        num_timestep_buckets: Number of timestep buckets (PTQ4DM only)
        num_calibration_samples: Number of samples for calibration
        calibration_steps: Inference steps during calibration
        num_inference_steps: Inference steps for generation
        guidance_scale: CFG scale
        batch_size: Batch size for generation
        start_idx: Starting index for image numbering
        device: Device to run on
        disable_quant: If True, disable quantization for testing
    """
    print(f"\n{'='*60}")
    print(f"QUANTIZED GENERATION: {method.upper()}")
    print(f"{'='*60}")
    print(f"Weight bits: {weight_bits}, Activation bits: {act_bits}")
    print(f"Calibration samples: {num_calibration_samples}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Create quantizer based on method
    if method.lower() == "qdiffusion":
        quantizer = create_qdiffusion_quantizer(
            weight_bits=weight_bits,
            act_bits=act_bits,
            device=device
        )
    elif method.lower() == "ptq4dm":
        quantizer = create_ptq4dm_quantizer(
            weight_bits=weight_bits,
            act_bits=act_bits,
            num_timestep_buckets=num_timestep_buckets,
            device=device
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Inject quantization
    quantizer.inject_quantization()

    # Run calibration (skip if quantization is disabled)
    if not disable_quant:
        quantizer.calibrate(
            num_calibration_samples=num_calibration_samples,
            num_inference_steps=calibration_steps,
            guidance_scale=guidance_scale,
        )

    # Print stats
    stats = quantizer.get_quantization_stats()
    print(f"\nQuantization stats: {stats}")

    # Generate images
    quantizer.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        start_idx=start_idx,
        disable_quant=disable_quant,
    )

    # Cleanup
    quantizer.cleanup()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with quantized diffusion models (Q-Diffusion / PTQ4DM)"
    )

    # Method selection
    parser.add_argument("--method", type=str, choices=["qdiffusion", "ptq4dm", "both"],
                       default="qdiffusion", help="Quantization method to use")

    # Quantization settings
    parser.add_argument("--weight-bits", type=int, default=8,
                       help="Bit-width for weights (4 or 8)")
    parser.add_argument("--act-bits", type=int, default=8,
                       help="Bit-width for activations (8)")
    parser.add_argument("--num-timestep-buckets", type=int, default=10,
                       help="Number of timestep buckets for PTQ4DM")

    # Calibration settings
    parser.add_argument("--num-calibration-samples", type=int, default=128,
                       help="Number of samples for calibration (more = better quality)")
    parser.add_argument("--calibration-steps", type=int, default=20,
                       help="Inference steps during calibration")

    # Generation settings
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of images to generate")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale for CFG")
    parser.add_argument("--batch-size", type=str, default="auto",
                       help="Batch size: int or 'auto' (default). "
                            "'auto' runs a two-point VRAM probe and picks "
                            "the largest batch that fits in 60%% of total VRAM, "
                            "with halve-on-OOM retry inside the generation loop.")
    parser.add_argument("--start-ind", type=int, default=0,
                       help="Starting index for generation")

    # Input/Output
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--prompt-file", type=str, default=None,
                       help="Path to file containing prompts")
    parser.add_argument("--coco-captions", type=str, default=None,
                       help="Path to COCO captions JSON")

    # FID calculation
    parser.add_argument("--calculate-fid", action="store_true",
                       help="Calculate FID score after generation")
    parser.add_argument("--reference-dir", type=str, default=None,
                       help="Reference directory for FID calculation")

    # Debug options
    parser.add_argument("--disable-quant", action="store_true",
                       help="Disable quantization for testing (verify pipeline works)")

    args = parser.parse_args()

    # Normalize --batch-size: keep "auto" as a string, else parse to int.
    if isinstance(args.batch_size, str) and args.batch_size.lower() != "auto":
        try:
            args.batch_size = int(args.batch_size)
        except ValueError:
            parser.error(f"--batch-size must be an integer or 'auto', got {args.batch_size!r}")
    elif isinstance(args.batch_size, str):
        args.batch_size = "auto"

    # Load prompts.  We need start_ind + num_samples captions so that sharded
    # runs (different --start-ind on different nodes) get disjoint slices of
    # the same deterministically-shuffled COCO list.
    total_needed = args.num_samples + args.start_ind
    if args.coco_captions:
        prompts = load_coco_captions(args.coco_captions, num_samples=total_needed)
    elif args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
    else:
        prompts = generate_sample_prompts(total_needed)

    # Apply start index
    if args.start_ind > 0:
        if args.start_ind >= len(prompts):
            print(f"Error: start_ind ({args.start_ind}) >= available prompts ({len(prompts)})")
            return
        prompts = prompts[args.start_ind:]
        print(f"Starting from index {args.start_ind}")

    prompts = prompts[:args.num_samples]

    # Run generation
    methods_to_run = []
    if args.method == "both":
        methods_to_run = ["qdiffusion", "ptq4dm"]
    else:
        methods_to_run = [args.method]

    all_stats = {}

    for method in methods_to_run:
        method_output_dir = Path(args.output_dir) / method / f"w{args.weight_bits}a{args.act_bits}"

        stats = run_quantized_generation(
            method=method,
            prompts=prompts,
            output_dir=str(method_output_dir / "images"),
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            num_timestep_buckets=args.num_timestep_buckets,
            num_calibration_samples=args.num_calibration_samples,
            calibration_steps=args.calibration_steps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            start_idx=args.start_ind,
            disable_quant=args.disable_quant,
        )

        all_stats[method] = stats

        # Calculate FID if requested
        if args.calculate_fid and args.reference_dir:
            fid_score = calculate_fid(args.reference_dir, str(method_output_dir / "images"))
            if fid_score is not None:
                all_stats[method]["fid_score"] = fid_score

    # Save stats
    stats_file = Path(args.output_dir) / "quantization_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_file}")

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    for method, stats in all_stats.items():
        print(f"\n{method.upper()}:")
        print(f"  Quantization: W{stats['weight_bits']}A{stats['act_bits']}")
        print(f"  Layers quantized: {stats['num_quantized_layers']}")
        if "fid_score" in stats:
            print(f"  FID Score: {stats['fid_score']:.2f}")


if __name__ == "__main__":
    main()
