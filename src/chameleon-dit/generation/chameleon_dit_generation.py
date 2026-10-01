"""
Chameleon-DiT Generation: Adaptive Quantization for PixArt-alpha (DiT)

Fuses Chameleon's format-routing philosophy with Q-DiT's group quantization:
  Weights: per-layer optimal (g, f) from exhaustive search over
           g ∈ {32,64,128,192,288}, f ∈ bit-width-dependent format set
           4-bit: {INT4_ASYM, NF4, FP4_E2M1}
           6-bit: {INT6_SYM, INT6_ASYM}
           8-bit: {INT8_SYM, INT8_ASYM, MXINT8}
  Activations: macro-format LUT (kurtosis/SNR routing) + dynamic scale

Usage:
    # Full pipeline (4-bit weights, default): search + calibrate + generate
    python chameleon_dit_generation.py --num-samples 100

    # 8-bit weights (W8A8)
    python chameleon_dit_generation.py --weight-bits 8 --num-samples 100

    # With COCO captions
    python chameleon_dit_generation.py --coco-captions /path/to/captions.json \
                                       --num-samples 5000

    # Save the per-layer (g,f) decisions + activation LUT for reuse
    python chameleon_dit_generation.py --save-weights /path/to/config.json \
                                       --num-samples 100

    # Load saved config (skip re-calibration)
    python chameleon_dit_generation.py --load-weights /path/to/config.json \
                                       --num-samples 5000

    # 20 timestep buckets (finer activation routing)
    python chameleon_dit_generation.py --num-buckets 20 --num-samples 100
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import argparse
import json
import os
import random
from typing import List, Optional

import torch

from chameleon_dit_quant import (
    ChameleonDiTQuantizer,
    create_chameleon_dit_quantizer,
    GROUP_SIZE_SEARCH,
    FORMAT_SEARCH_BY_BITS,
    WeightFormat,
    DEFAULT_ACT_GROUP_SIZE,
    DEFAULT_ACT_MODE,
    ACT_MODE_PERTENSOR,
    ACT_MODES,
)


# ── Configuration ──────────────────────────────────────────────────────────────

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
DEFAULT_OUTPUT   = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "chameleon-dit")


# ── Prompt utilities ───────────────────────────────────────────────────────────

def load_coco_captions(
    json_path:   str,
    num_samples: int,
    start_idx:   int  = 0,
    shuffle:     bool = True,
) -> List[str]:
    """Load COCO captions with the same seed used across all baselines."""
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

def run_chameleon_dit_generation(
    output_dir:              str,
    num_samples:             int   = 100,
    start_idx:               int   = 0,
    coco_captions:           Optional[str] = None,
    prompt_file:             Optional[str] = None,
    num_inference_steps:     int   = 20,
    guidance_scale:          float = 4.5,
    batch_size:              int   = 1,
    num_calibration_samples: int   = 64,
    num_buckets:             int   = 10,
    weight_bits:             int   = 4,
    act_quant_mode:          str   = DEFAULT_ACT_MODE,
    act_group_size:          int   = DEFAULT_ACT_GROUP_SIZE,
    input_aware_weights:     bool  = False,
    clip_search:             bool  = False,
    mixed_precision:         bool  = False,
    mp_target_bits:          float = 4.5,
    mp_promote_to:           int   = 8,
    model_id:                str   = DEFAULT_MODEL_ID,
    save_weights:            Optional[str] = None,
    load_weights:            Optional[str] = None,
    device:                  str   = DEVICE,
    verbose:                 bool  = False,
) -> dict:
    """Run Chameleon-DiT quantised generation. Returns statistics dictionary."""

    # Load prompts
    if coco_captions:
        prompts = load_coco_captions(coco_captions, num_samples, start_idx)
    elif prompt_file:
        all_prompts = load_prompts_from_file(prompt_file)
        end         = min(start_idx + num_samples, len(all_prompts))
        prompts     = all_prompts[start_idx:end]
        print(f"Using {len(prompts)} prompts from file (indices {start_idx}–{end})")
    else:
        prompts = get_default_prompts(num_samples)

    # Output subfolder encodes weight bit-width and buckets.  The canonical
    # 'pertensor' mode keeps the bare slot name (so the score table / score_all.sh
    # line up); the group-* ablation modes get a distinct suffix.
    subdir     = f"w{weight_bits}a8_chameleon_dit_{num_buckets}b"
    if act_quant_mode != ACT_MODE_PERTENSOR:
        subdir += f"_act{act_quant_mode}"
    if input_aware_weights:
        subdir += "_iaw"          # input-aware weight selection (output-error criterion)
    if clip_search:
        subdir += "_clip"         # weight clip-ratio search (#2)
    if mixed_precision and weight_bits == 4:
        subdir += f"_mp{mp_target_bits}"   # mixed-precision allocation (#1)
    output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    fmt_search = FORMAT_SEARCH_BY_BITS.get(weight_bits, FORMAT_SEARCH_BY_BITS[4])

    print(f"\n{'='*60}")
    print("CHAMELEON-DiT QUANTISED GENERATION")
    print(f"{'='*60}")
    print(f"Model               : {model_id}")
    print(f"Mode                : Chameleon-DiT (is_dit=True)")
    print(f"Weight bits         : {weight_bits} (W{weight_bits}A8)")
    print(f"Weight quant        : per-layer optimal (g, f) search")
    print(f"  g search space    : {GROUP_SIZE_SEARCH}")
    print(f"  f search space    : {[f.value for f in fmt_search]}")
    print(f"  selection         : {'input-aware (output error)' if (input_aware_weights or (mixed_precision and weight_bits==4)) else 'weight-only SNR'}")
    print(f"  clip search       : {'on' if clip_search else 'off'}")
    if mixed_precision and weight_bits == 4:
        print(f"  mixed precision   : on (4→{mp_promote_to}-bit, avg target {mp_target_bits})")
    print(f"Activation quant    : dynamic, mode={act_quant_mode}")
    if act_quant_mode == ACT_MODE_PERTENSOR:
        print(f"                      (canonical: per-tensor scale, macro-routed format)")
    else:
        print(f"                      (ABLATION: group-wise, group={act_group_size} — "
              f"underperforms per-tensor on DiT)")
    print(f"Num buckets         : {num_buckets}")
    print(f"Output              : {output_dir}")
    print(f"Samples             : {len(prompts)}, start={start_idx}")
    print(f"Inference steps     : {num_inference_steps}")
    print(f"Guidance scale      : {guidance_scale}")
    print(f"Batch size          : {batch_size}")
    print(f"Device              : {device}")
    print(f"{'='*60}\n")

    quantizer = create_chameleon_dit_quantizer(
        model_id=model_id,
        device=device,
        num_buckets=num_buckets,
        weight_bits=weight_bits,
        act_group_size=act_group_size,
        act_mode=act_quant_mode,
        input_aware_weights=input_aware_weights,
        clip_search=clip_search,
        mixed_precision=mixed_precision,
        mp_target_bits=mp_target_bits,
        mp_promote_to=mp_promote_to,
    )

    if load_weights:
        # ── Load path: skip weight search + calibration ───────────────────────
        print(f"Loading saved config from {load_weights}...")
        quantizer.load_weight_quantization(load_weights)
    else:
        # ── Full path: search + calibrate ─────────────────────────────────────
        quantizer.load_model()
        quantizer.inject_weight_quantization(verbose=verbose)

        print(f"\nCalibrating activations ({num_calibration_samples} samples)...")
        quantizer.calibrate_activations(
            num_calibration_samples=num_calibration_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            verbose=verbose,
        )

        if save_weights:
            quantizer.save_weight_quantization(save_weights)

    # ── Generate images ────────────────────────────────────────────────────────
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
        'num_samples':         len(prompts),
        'start_idx':           start_idx,
        'num_inference_steps': num_inference_steps,
        'guidance_scale':      guidance_scale,
        'output_dir':          output_dir,
    })

    stats_file = os.path.join(output_dir, "chameleon_dit_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

    quantizer.cleanup()
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chameleon-DiT: adaptive-format quantization for PixArt-alpha"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Base output directory (subfolder w<W>a8_chameleon_dit_<N>b appended)",
    )
    parser.add_argument("--num-samples",  type=int,   default=100)
    parser.add_argument("--start-idx",   type=int,   default=0,
                        help="Starting index for image naming / caption slice")

    # Prompt source
    parser.add_argument("--coco-captions", type=str, default=None,
                        help="Path to COCO captions JSON (captions_val2014.json)")
    parser.add_argument("--prompt-file",   type=str, default=None,
                        help="Text file with prompts (one per line)")

    # Model
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID",
    )

    # Chameleon-DiT specific
    parser.add_argument(
        "--weight-bits", type=int, default=4, choices=[4, 6, 8],
        help="Weight bit-width: 4=W4A8, 6=W6A8, 8=W8A8 (default: 4)",
    )
    parser.add_argument(
        "--num-buckets", type=int, default=10,
        help="Number of timestep buckets for macro activation routing (default: 10)",
    )
    parser.add_argument(
        "--num-calibration-samples", type=int, default=64,
        help="Calibration samples for activation LUT (default: 64)",
    )
    parser.add_argument(
        "--act-quant-mode", type=str, default=DEFAULT_ACT_MODE, choices=list(ACT_MODES),
        help="Activation quant mode. 'pertensor' (default, CANONICAL): one global "
             "scale/zp per tensor in the macro-routed format. 'group-macro' / "
             "'group-int8' (ABLATION): group-wise scaling — empirically WORSE on "
             "PixArt-alpha (kept for the ablation table only).",
    )
    parser.add_argument(
        "--act-group-size", type=int, default=DEFAULT_ACT_GROUP_SIZE,
        help=f"Channel group size for the group-* ablation modes (default: "
             f"{DEFAULT_ACT_GROUP_SIZE}, Q-DiT's default). Ignored for 'pertensor'.",
    )
    parser.add_argument(
        "--input-aware-weights", action="store_true",
        help="Select per-layer (g,f) by OUTPUT error instead of weight-only SNR: "
             "a calibration pre-pass estimates per-input-channel activation energy "
             "E[x_i^2] and the search minimises Sum_i E[x_i^2]||dW[:,i]||^2 "
             "(diagonal-Hessian / GPTQ / Q-DiT criterion). Targets the W4A8 gap vs "
             "Q-DiT. Output dir gets an '_iaw' suffix.",
    )
    parser.add_argument(
        "--clip-search", action="store_true",
        help="#2: also search a per-group weight clip ratio (trade clipping error "
             "for lower rounding error, AWQ/OMSE-style). Adds a clip axis to the "
             "(g,f) search. Output dir gets a '_clip' suffix.",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="#1 (W4 only): keep most layers at 4-bit but promote the most "
             "output-sensitive layers to --mp-promote-to bits under an average-bit "
             "budget (--mp-target-bits). Forces input-aware sensitivity scoring. "
             "Output dir gets an '_mp<target>' suffix.",
    )
    parser.add_argument("--mp-target-bits", type=float, default=4.5,
                        help="Average weight bit-width budget for --mixed-precision (default: 4.5).")
    parser.add_argument("--mp-promote-to", type=int, default=8, choices=[6, 8],
                        help="Bit-width sensitive layers are promoted to (default: 8).")

    # Generation parameters
    parser.add_argument("--num-inference-steps", type=int,   default=20)
    parser.add_argument("--guidance-scale",       type=float, default=4.5)
    parser.add_argument("--batch-size",           type=int,   default=1)

    # Save / load
    parser.add_argument("--save-weights", type=str, default=None,
                        help="Save per-layer (g,f) decisions + activation LUT to JSON")
    parser.add_argument("--load-weights", type=str, default=None,
                        help="Load saved config (skips search + calibration)")

    # Misc
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device",  type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    stats = run_chameleon_dit_generation(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        coco_captions=args.coco_captions,
        prompt_file=args.prompt_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        num_calibration_samples=args.num_calibration_samples,
        num_buckets=args.num_buckets,
        weight_bits=args.weight_bits,
        act_quant_mode=args.act_quant_mode,
        act_group_size=args.act_group_size,
        input_aware_weights=args.input_aware_weights,
        clip_search=args.clip_search,
        mixed_precision=args.mixed_precision,
        mp_target_bits=args.mp_target_bits,
        mp_promote_to=args.mp_promote_to,
        model_id=args.model_id,
        save_weights=args.save_weights,
        load_weights=args.load_weights,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model            : {stats['model_id']}")
    print(f"Weight bits      : {stats['weight_bits']} (W{stats['weight_bits']}A8)")
    print(f"Quantised layers : {stats['num_quant_layers']}")
    print(f"Num buckets      : {stats['num_buckets']}")
    print(f"Images           : {stats['num_samples']}")
    print(f"Output           : {stats['output_dir']}")
    print(f"Mean weight SNR  : {stats['mean_weight_snr_db']:.2f} dB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
