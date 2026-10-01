#!/usr/bin/env python3
"""
Chameleon Generation: Image generation with dynamic kurtosis/SNR-based quantization

Usage:
    python chameleon_generation.py --num-samples 100 --output-dir ./output

    # With COCO captions
    python chameleon_generation.py --coco-captions /path/to/captions.json --num-samples 5000

    # Save/load weight quantization (includes activation LUT if calibration was run)
    python chameleon_generation.py --save-weights /path/to/weights.json
    python chameleon_generation.py --load-weights /path/to/weights.json

    # Separate activation LUT
    python chameleon_generation.py --save-activation-lut /path/to/lut.json
    python chameleon_generation.py --load-activation-lut /path/to/lut.json
"""

import argparse
import json
import os
import random
import sys
from typing import List, Optional

import torch

from chameleon_quant import ChameleonQuantizer, create_chameleon_quantizer
from snr_format_selector import QuantFormat


def load_coco_captions(
    json_path: str,
    num_samples: int,
    start_idx: int = 0,
    shuffle: bool = True,
) -> List[str]:
    """Load captions from COCO annotations file (same shuffle seed as q-baselines)."""
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
    """Default prompts for testing."""
    base = [
        "A photo of a cat sitting on a windowsill",
        "A beautiful sunset over the ocean",
        "A futuristic city skyline at night",
        "A serene mountain landscape with snow",
        "An astronaut riding a horse on Mars",
        "A cozy coffee shop interior",
        "A colorful butterfly on a flower",
        "A majestic lion in the savanna",
        "A Japanese garden with cherry blossoms",
        "A steampunk mechanical robot",
        "A tropical beach with palm trees",
        "A medieval castle on a cliff",
        "A portrait of a wise wizard",
        "A bowl of fresh fruits",
        "A golden retriever running on a beach",
    ]
    prompts = []
    while len(prompts) < num_samples:
        prompts.extend(base)
    return prompts[:num_samples]


def run_chameleon_generation(
    output_dir:              str,
    num_samples:             int   = 100,
    start_idx:               int   = 0,
    coco_captions:           Optional[str] = None,
    num_inference_steps:     int   = 50,
    guidance_scale:          float = 7.5,
    batch_size:              int   = 1,
    num_calibration_samples: int   = 128,
    min_snr_threshold:       float = 20.0,
    mx_block_size:           int   = 32,
    num_buckets:             int   = 10,
    skip_attention_qkv:      bool  = True,
    skip_attention_out:      bool  = False,
    skip_first_last_conv:    bool  = True,
    save_weights:            Optional[str] = None,
    load_weights:            Optional[str] = None,
    save_activation_lut:     Optional[str] = None,
    load_activation_lut:     Optional[str] = None,
    verbose:                 bool  = False,
    device:                  str   = "cuda",
    weight_bits:             int   = 8,
) -> dict:
    """Run Chameleon quantized generation. Returns statistics dictionary."""

    # Prompts
    if coco_captions:
        prompts = load_coco_captions(coco_captions, num_samples, start_idx)
    else:
        prompts = get_default_prompts(num_samples)

    output_dir = os.path.join(output_dir, f"w{weight_bits}a8")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("CHAMELEON QUANTIZED GENERATION")
    print(f"{'='*60}")
    print(f"Output              : {output_dir}")
    print(f"Weight bits         : {weight_bits} (W{weight_bits}A8)")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Calibration samples : {num_calibration_samples}")
    print(f"Num buckets         : {num_buckets}")
    print(f"Min SNR threshold   : {min_snr_threshold} dB")
    print(f"MX block size       : {mx_block_size}")
    print(f"{'='*60}\n")

    # Create quantizer
    quantizer = create_chameleon_quantizer(
        min_snr_threshold=min_snr_threshold,
        mx_block_size=mx_block_size,
        device=device,
        num_buckets=num_buckets,
        weight_bits=weight_bits,
    )

    # Load model
    quantizer.load_model()

    # Step 1: Weight quantization
    if load_weights:
        quantizer.load_weight_quantization(load_weights)
    else:
        quantizer.inject_weight_quantization(
            skip_attention_qkv=skip_attention_qkv,
            skip_attention_out=skip_attention_out,
            skip_first_last_conv=skip_first_last_conv,
            verbose=verbose,
        )
        if save_weights:
            quantizer.save_weight_quantization(save_weights)

    quantizer.print_weight_stats()

    # Step 2: Activation calibration (skip if LUT already loaded from weights file)
    if not quantizer.activation_calibration_complete:
        if load_activation_lut:
            quantizer.load_activation_lut(load_activation_lut)
        else:
            quantizer.calibrate_activations(
                num_calibration_samples=num_calibration_samples,
                num_inference_steps=min(num_inference_steps, 20),
                guidance_scale=guidance_scale,
                verbose=verbose,
            )
            if save_activation_lut:
                quantizer.save_activation_lut(save_activation_lut)
            # Re-save weights with embedded LUT if weight path was given
            if save_weights:
                quantizer.save_weight_quantization(save_weights)

    # Generate images
    quantizer.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        start_idx=start_idx,
    )

    # Stats
    stats = quantizer.get_quantization_stats()

    # Serialise stats to JSON
    stats_file = os.path.join(output_dir, "chameleon_stats.json")
    bucket_size = 1000 // num_buckets
    serializable = {
        "num_quantized_layers":          stats["num_quantized_layers"],
        "weight_quantization_complete":  stats["weight_quantization_complete"],
        "activation_calibration_complete": stats["activation_calibration_complete"],
        "num_buckets":                   stats["num_buckets"],
        "weight_format_distribution": {
            (k.value if hasattr(k, 'value') else str(k)): v
            for k, v in stats["weight_format_distribution"].items()
        },
    }
    if "activation_bucket_distribution" in stats:
        serializable["activation_bucket_distribution"] = {
            f"bucket_{b}_t{b*bucket_size}-{(b+1)*bucket_size-1}": fmt_counts
            for b, fmt_counts in stats["activation_bucket_distribution"].items()
        }

    with open(stats_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Stats saved to {stats_file}")

    quantizer.cleanup()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Chameleon: dynamic kurtosis/SNR-based adaptive quantized image generation"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "chameleon/test"),
        help="Output directory for generated images",
    )
    parser.add_argument("--num-samples",  type=int, default=100)
    parser.add_argument("--start-idx",    type=int, default=0)

    # Data source
    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON file")

    # Calibration
    parser.add_argument(
        "--num-calibration-samples", type=int, default=128,
        help="Number of samples for activation calibration",
    )
    parser.add_argument(
        "--num-buckets", type=int, default=10,
        help="Number of timestep buckets for activation LUT (default: 10 for T=1000)",
    )

    # Generation
    parser.add_argument("--num-inference-steps", type=int,   default=50)
    parser.add_argument("--guidance-scale",       type=float, default=7.5)
    parser.add_argument("--batch-size",           type=int,   default=1)

    # Quantization settings
    parser.add_argument("--min-snr-threshold", type=float, default=20.0,
                        help="Minimum SNR (dB) for weight format selection")
    parser.add_argument("--mx-block-size",     type=int,   default=32,
                        help="Block size for MX formats")
    parser.add_argument("--weight-bits",       type=int,   default=8, choices=[4, 8],
                        help="Weight bit-width: 4 for W4A8, 8 for W8A8 (default: 8)")

    # Skip patterns
    parser.add_argument("--quantize-attention-qkv", action="store_true",
                        help="Quantize attention Q/K/V (not recommended)")
    parser.add_argument("--skip-attention-out",     action="store_true",
                        help="Skip attention output projection")
    parser.add_argument("--quantize-conv-io",       action="store_true",
                        help="Quantize first/last conv layers")

    # Save / load
    parser.add_argument("--save-weights",         type=str, default=None,
                        help="Save weight+LUT config to JSON")
    parser.add_argument("--load-weights",         type=str, default=None,
                        help="Load weight+LUT config from JSON")
    parser.add_argument("--save-activation-lut",  type=str, default=None,
                        help="Save activation LUT to separate JSON")
    parser.add_argument("--load-activation-lut",  type=str, default=None,
                        help="Load activation LUT from separate JSON")

    # Misc
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device",  type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    stats = run_chameleon_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        num_calibration_samples=args.num_calibration_samples,
        min_snr_threshold=args.min_snr_threshold,
        mx_block_size=args.mx_block_size,
        num_buckets=args.num_buckets,
        skip_attention_qkv=not args.quantize_attention_qkv,
        skip_attention_out=args.skip_attention_out,
        skip_first_last_conv=not args.quantize_conv_io,
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        save_activation_lut=args.save_activation_lut,
        load_activation_lut=args.load_activation_lut,
        verbose=args.verbose,
        device=args.device,
        weight_bits=args.weight_bits,
    )

    # Final summary
    bucket_size = 1000 // stats.get('num_buckets', 10)
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Quantized layers : {stats['num_quantized_layers']}")
    print(f"Num buckets      : {stats.get('num_buckets', 'N/A')}")

    print("\nWeight format distribution (per output-channel):")
    for fmt, count in stats['weight_format_distribution'].items():
        fmt_str = fmt.value if hasattr(fmt, 'value') else str(fmt)
        print(f"  {fmt_str}: {count} channels")

    if "activation_bucket_distribution" in stats:
        print("\nActivation bucket routing table:")
        for b, fmt_counts in sorted(stats["activation_bucket_distribution"].items()):
            t_lo = b * bucket_size
            t_hi = (b + 1) * bucket_size - 1
            dist = ", ".join(f"{fmt}:{cnt}" for fmt, cnt in sorted(fmt_counts.items()))
            print(f"  bucket {b:2d} (t={t_lo:4d}-{t_hi:4d}): {dist}")

    print("="*60)


if __name__ == "__main__":
    main()
