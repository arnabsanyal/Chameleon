"""
Example usage of baseline generation code
Run this after installing requirements
"""

import os
from baseline_generation import (
    ImageBaselineGenerator,
    VideoBaselineGenerator,
    generate_sample_prompts
)
from pathlib import Path

def example_image_generation():
    """Example: Generate baseline images"""
    print("Example 1: Image Generation with SDXL\n")

    # Create generator
    generator = ImageBaselineGenerator()

    # Generate sample prompts
    prompts = generate_sample_prompts(num_samples=10)
    print(f"Generated {len(prompts)} prompts")
    print(f"First prompt: {prompts[0]}\n")

    # Generate images
    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "example_images")
    generator.generate_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=30,  # Reduced for faster generation
        guidance_scale=7.5,
        batch_size=4  # Generate 4 images in parallel
    )

    print(f"\nImages saved to: {output_dir}")
    generator.cleanup()


def example_video_generation():
    """Example: Generate baseline videos from images"""
    print("\n" + "="*60)
    print("Example 2: Video Generation with SVD\n")

    # First, we need some input images
    # Let's use the images from the previous example
    input_images_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "example_images")

    if not Path(input_images_dir).exists():
        print(f"Input directory {input_images_dir} not found.")
        print("Please run example_image_generation() first.")
        return

    # Load input images
    input_images = sorted(list(Path(input_images_dir).glob("*.png")))[:5]  # Use first 5 images

    if not input_images:
        print("No images found in input directory")
        return

    print(f"Using {len(input_images)} input images")

    # Create generator
    generator = VideoBaselineGenerator()

    # Generate videos
    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "example_videos")
    generator.generate_videos(
        input_images=[str(img) for img in input_images],
        output_dir=output_dir,
        fps=7,
        num_frames=14  # Shorter for quick testing
    )

    print(f"\nVideos saved to: {output_dir}")
    generator.cleanup()


def example_custom_prompts():
    """Example: Use custom prompts from a list"""
    print("\n" + "="*60)
    print("Example 3: Custom Prompts\n")

    # Define custom prompts
    custom_prompts = [
        "A serene mountain landscape at sunset with snow-capped peaks",
        "A futuristic city with flying cars and neon lights",
        "A close-up portrait of a wise old wizard with a long beard",
        "A tropical beach with crystal clear water and palm trees",
        "An abstract painting with vibrant colors and geometric shapes"
    ]

    # Create generator
    generator = ImageBaselineGenerator()

    # Generate images
    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "custom_prompts")
    generator.generate_images(
        prompts=custom_prompts,
        output_dir=output_dir,
        num_inference_steps=40,
        guidance_scale=8.0,  # Higher guidance for more prompt adherence
        batch_size=4  # Generate 4 images in parallel
    )

    print(f"\nImages saved to: {output_dir}")
    generator.cleanup()


if __name__ == "__main__":
    print("="*60)
    print("Baseline Generation Examples")
    print("="*60)
    print("\nThese examples demonstrate the baseline generation pipeline")
    print("Make sure you have a CUDA-capable GPU and all requirements installed\n")

    # Run examples
    try:
        # Example 1: Generate images
        example_image_generation()

        # Example 2: Generate videos from the images
        example_video_generation()

        # Example 3: Custom prompts
        example_custom_prompts()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Make sure you have:")
        print("1. CUDA-capable GPU with sufficient memory")
        print("2. All requirements installed (pip install -r requirements.txt)")
        print("3. Sufficient disk space for models and outputs")
