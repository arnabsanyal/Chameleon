"""
FP16 SDXL baseline image generation for FID / CLIP evaluation.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import torch
from diffusers import StableDiffusionXLPipeline
import os
from tqdm import tqdm
import argparse
import sys as _sys
from pathlib import Path

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))
from perf_instrument import PerfTracker  # noqa: E402

# CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # FP16 reference precision
OUTPUT_DIR = os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs"))


class ImageBaselineGenerator:
    """Generate images using SDXL for FID calculation"""

    def __init__(self, device=DEVICE, dtype=DTYPE, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.pipe = None

    def setup_pipeline(self):
        """Load SDXL pipeline with optimizations"""
        print(f"Loading SDXL from {self.model_id}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        )

        # Move to device (respects CUDA_VISIBLE_DEVICES)
        self.pipe.to(self.device)

        # Optional: Enable xformers for memory efficiency
        if self.device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception as e:
                print(f"xformers not available: {e}")

        # Print device info
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            print(f"Pipeline loaded on {torch.cuda.get_device_name(device_idx)} (cuda:{device_idx})")
        else:
            print("Pipeline loaded on CPU")
        return self.pipe

    def generate_images(self, prompts, output_dir, num_inference_steps=50, guidance_scale=7.5, batch_size=4, start_idx=0):
        """
        Generate images for FID calculation.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            batch_size: Number of images to generate in parallel
            start_idx: Starting index for image numbering (default: 0)
        """
        if self.pipe is None:
            self.setup_pipeline()

        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {len(prompts)} baseline images with batch size {batch_size}...")
        print(f"Starting from index {start_idx}, generating indices {start_idx} to {start_idx + len(prompts) - 1}")

        tracker = PerfTracker(label="sdxl_fp16", device=self.device)
        tracker.reset()

        # Process prompts in batches
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            try:
                # Generate batch of images
                # Create separate generators for each image in batch for reproducibility
                # Use global index (start_idx + batch_start + j) for seed to ensure consistency
                generators = [torch.Generator(device=self.device).manual_seed(start_idx + batch_start + j)
                             for j in range(len(batch_prompts))]

                tracker.batch_start()
                result = self.pipe(
                    prompt=batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators
                )
                images = result.images
                tracker.batch_end(len(batch_prompts))

                # Save each image in the batch with global index
                for j, image in enumerate(images):
                    image_idx = start_idx + batch_start + j
                    image.save(f"{output_dir}/{image_idx:05d}.png")

            except Exception as e:
                print(f"Error generating batch starting at {start_idx + batch_start}: {e}")
                # Fallback to single-image generation for this batch
                print(f"Falling back to single-image generation for batch {start_idx + batch_start}-{start_idx + batch_end}")
                for j, prompt in enumerate(batch_prompts):
                    try:
                        image_idx = start_idx + batch_start + j
                        result = self.pipe(
                            prompt=prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=torch.Generator(device=self.device).manual_seed(image_idx)
                        )
                        image = result.images[0]
                        image.save(f"{output_dir}/{image_idx:05d}.png")
                    except Exception as e2:
                        print(f"Error generating image {image_idx}: {e2}")
                        continue

        tracker.finish()
        tracker.dump_json(Path(output_dir) / "perf.json")
        print(f"Generated {len(prompts)} images in {output_dir}")

    def cleanup(self):
        """Free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()


def load_prompts_from_file(prompt_file):
    """Load prompts from a text file (one prompt per line)"""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_coco_captions(annotations_file, num_samples=None, shuffle=True):
    """
    Load captions from COCO annotations JSON file.

    Args:
        annotations_file: Path to captions_val2014.json or captions_train2014.json
        num_samples: Number of captions to return (None for all)
        shuffle: Whether to shuffle captions randomly

    Returns:
        List of caption strings
    """
    import json
    import random

    print(f"Loading COCO captions from {annotations_file}...")

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Extract captions
    captions = [ann['caption'].strip() for ann in data['annotations']]

    print(f"Loaded {len(captions)} captions")

    if shuffle:
        random.seed(42)  # For reproducibility
        random.shuffle(captions)

    if num_samples is not None:
        captions = captions[:num_samples]

    return captions


def generate_sample_prompts(num_samples=100):
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


def calculate_fid(real_images_dir, generated_images_dir):
    """
    Calculate FID score using clean-fid library.
    Install: pip install clean-fid
    """
    try:
        from cleanfid import fid

        print(f"Calculating FID between {real_images_dir} and {generated_images_dir}")
        score = fid.compute_fid(real_images_dir, generated_images_dir)
        print(f"FID Score: {score:.2f}")
        return score
    except ImportError:
        print("clean-fid not installed. Install with: pip install clean-fid")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate FP16 SDXL baseline images")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--prompt-file", type=str, default=None,
                       help="Path to file containing prompts (one per line)")
    parser.add_argument("--coco-captions", type=str, default=None,
                       help="Path to COCO captions JSON (e.g., annotations/captions_val2014.json)")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                       help="Number of inference steps for image generation")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale for image generation")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for image generation (default: 4)")
    parser.add_argument("--start-ind", type=int, default=0,
                       help="Starting index for generation (default: 0). Use to resume from a specific point.")
    parser.add_argument("--calculate-fid", action="store_true",
                       help="Calculate FID score after generation")
    parser.add_argument("--reference-dir", type=str, default=None,
                       help="Reference directory for FID calculation")

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"

    # Generate images
    print("\n" + "="*50)
    print("GENERATING BASELINE IMAGES")
    print("="*50 + "\n")

    # Load or generate prompts
    if args.coco_captions:
        prompts = load_coco_captions(args.coco_captions, num_samples=args.num_samples)
    elif args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
    else:
        prompts = generate_sample_prompts(args.num_samples)

    # Limit to requested number
    prompts = prompts[:args.num_samples]

    # Apply start index - slice prompts from start_ind onwards
    start_ind = args.start_ind
    if start_ind > 0:
        if start_ind >= len(prompts):
            print(f"Error: start_ind ({start_ind}) >= num_samples ({len(prompts)})")
            return
        prompts = prompts[start_ind:]
        print(f"Resuming from index {start_ind}, generating {len(prompts)} images (indices {start_ind} to {start_ind + len(prompts) - 1})")
    else:
        print(f"Using {len(prompts)} prompts")

    # Generate images
    generator = ImageBaselineGenerator()
    generator.generate_images(
        prompts,
        str(images_dir),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        start_idx=start_ind
    )
    generator.cleanup()

    # Calculate FID if requested
    if args.calculate_fid and args.reference_dir:
        calculate_fid(args.reference_dir, str(images_dir))

    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Images: {images_dir}")


if __name__ == "__main__":
    main()
