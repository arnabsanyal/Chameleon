"""
Example usage of quantized baseline generation code
Demonstrates Q-Diffusion and PTQ4DM quantization techniques
"""

import os
from qdiffusion import QDiffusionQuantizer, create_qdiffusion_quantizer
from ptq4dm import PTQ4DMQuantizer, create_ptq4dm_quantizer


def example_qdiffusion_w8a8():
    """Example: Q-Diffusion with 8-bit weights and activations"""
    print("\n" + "="*60)
    print("Example 1: Q-Diffusion W8A8 Quantization")
    print("="*60 + "\n")

    # Create quantizer
    quantizer = create_qdiffusion_quantizer(
        weight_bits=8,
        act_bits=8,
        device="cuda"
    )

    # Inject quantization nodes
    quantizer.inject_quantization()

    # Calibrate on sample data
    # In practice, use more samples (32-64) for better calibration
    quantizer.calibrate(
        num_calibration_samples=8,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    # Print stats
    stats = quantizer.get_quantization_stats()
    print(f"Quantization stats: {stats}")

    # Generate test images
    test_prompts = [
        "A photo of a golden retriever playing in a park",
        "A futuristic cityscape at sunset",
        "A bowl of ramen with steam rising",
        "An astronaut floating in space",
        "A cozy cabin in a snowy forest"
    ]

    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/example_qdiff_w8a8")
    quantizer.generate_images(
        prompts=test_prompts,
        output_dir=output_dir,
        num_inference_steps=30,
        guidance_scale=7.5
    )

    print(f"\nImages saved to: {output_dir}")
    quantizer.cleanup()


def example_qdiffusion_w4a8():
    """Example: Q-Diffusion with aggressive 4-bit weight quantization"""
    print("\n" + "="*60)
    print("Example 2: Q-Diffusion W4A8 Quantization (Aggressive)")
    print("="*60 + "\n")

    # Create quantizer with 4-bit weights
    quantizer = create_qdiffusion_quantizer(
        weight_bits=4,  # Aggressive!
        act_bits=8,
        device="cuda"
    )

    quantizer.inject_quantization()
    quantizer.calibrate(num_calibration_samples=8, num_inference_steps=20)

    stats = quantizer.get_quantization_stats()
    print(f"Quantization stats: {stats}")

    test_prompts = [
        "A majestic lion in the savanna",
        "A vintage car on Route 66",
        "A Japanese zen garden"
    ]

    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/example_qdiff_w4a8")
    quantizer.generate_images(
        prompts=test_prompts,
        output_dir=output_dir,
        num_inference_steps=30
    )

    print(f"\nImages saved to: {output_dir}")
    quantizer.cleanup()


def example_ptq4dm():
    """Example: PTQ4DM with timestep-aware quantization"""
    print("\n" + "="*60)
    print("Example 3: PTQ4DM Timestep-Aware Quantization")
    print("="*60 + "\n")

    # Create PTQ4DM quantizer
    # Key difference: uses timestep buckets to handle changing
    # activation distributions across diffusion steps
    quantizer = create_ptq4dm_quantizer(
        weight_bits=8,
        act_bits=8,
        num_timestep_buckets=10,  # 10 buckets across t=0 to t=1000
        device="cuda"
    )

    quantizer.inject_quantization()

    # Calibrate with timestep-aware statistics collection
    quantizer.calibrate(
        num_calibration_samples=8,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    stats = quantizer.get_quantization_stats()
    print(f"Quantization stats: {stats}")

    test_prompts = [
        "A serene mountain lake at dawn",
        "A steampunk mechanical owl",
        "A tropical beach with palm trees",
        "A medieval castle on a cliff",
        "A colorful hot air balloon festival"
    ]

    output_dir = os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/example_ptq4dm")
    quantizer.generate_images(
        prompts=test_prompts,
        output_dir=output_dir,
        num_inference_steps=30,
        guidance_scale=7.5
    )

    print(f"\nImages saved to: {output_dir}")
    quantizer.cleanup()


def example_comparison():
    """Example: Compare Q-Diffusion vs PTQ4DM on same prompts"""
    print("\n" + "="*60)
    print("Example 4: Comparison - Q-Diffusion vs PTQ4DM")
    print("="*60 + "\n")

    # Common test prompts for fair comparison
    test_prompts = [
        "A photorealistic portrait of a wise old wizard",
        "A cyberpunk street scene with neon signs",
        "A still life painting of fruits and flowers"
    ]

    # Q-Diffusion
    print(">>> Running Q-Diffusion W8A8...")
    qdiff = create_qdiffusion_quantizer(weight_bits=8, act_bits=8)
    qdiff.inject_quantization()
    qdiff.calibrate(num_calibration_samples=8)

    qdiff.generate_images(
        prompts=test_prompts,
        output_dir=os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/comparison/qdiffusion"),
        num_inference_steps=30
    )
    qdiff.cleanup()

    # PTQ4DM
    print("\n>>> Running PTQ4DM W8A8...")
    ptq4dm = create_ptq4dm_quantizer(weight_bits=8, act_bits=8)
    ptq4dm.inject_quantization()
    ptq4dm.calibrate(num_calibration_samples=8)

    ptq4dm.generate_images(
        prompts=test_prompts,
        output_dir=os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/comparison/ptq4dm"),
        num_inference_steps=30
    )
    ptq4dm.cleanup()

    print("\n" + "="*60)
    print("Comparison complete!")
    print("Q-Diffusion: $CHAMELEON_OUTPUT_ROOT/quantized/comparison/qdiffusion/")
    print("PTQ4DM: $CHAMELEON_OUTPUT_ROOT/quantized/comparison/ptq4dm/")
    print("="*60)


def example_programmatic_usage():
    """Example: Programmatic usage without CLI"""
    print("\n" + "="*60)
    print("Example 5: Programmatic Usage")
    print("="*60 + "\n")

    from quantized_generation import run_quantized_generation

    prompts = [
        "A cute robot serving coffee",
        "A dragon flying over mountains"
    ]

    # Run Q-Diffusion programmatically
    stats = run_quantized_generation(
        method="qdiffusion",
        prompts=prompts,
        output_dir=os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "quantized/programmatic"),
        weight_bits=8,
        act_bits=8,
        num_calibration_samples=4,
        num_inference_steps=20,
    )

    print(f"\nReturned stats: {stats}")


if __name__ == "__main__":
    print("="*60)
    print("Quantized Baseline Generation Examples")
    print("="*60)
    print("\nThese examples demonstrate Q-Diffusion and PTQ4DM quantization")
    print("Make sure you have a CUDA-capable GPU and all requirements installed\n")

    try:
        # Example 1: Q-Diffusion W8A8
        example_qdiffusion_w8a8()

        # Example 2: Q-Diffusion W4A8 (aggressive)
        # Uncomment to test 4-bit quantization
        # example_qdiffusion_w4a8()

        # Example 3: PTQ4DM
        example_ptq4dm()

        # Example 4: Comparison
        # Uncomment to run comparison
        # example_comparison()

        # Example 5: Programmatic usage
        # Uncomment to test programmatic API
        # example_programmatic_usage()

        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Make sure you have:")
        print("1. CUDA-capable GPU with sufficient memory (24GB+ recommended)")
        print("2. All requirements installed (pip install -r requirements.txt)")
        print("3. Sufficient disk space for models and outputs")
        raise
