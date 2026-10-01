#!/usr/bin/env python3
"""
Quick test script for Chameleon quantization.
Tests the SNR format selection and basic generation.
"""

import torch
import sys

def test_snr_format_selector():
    """Test SNR-based format selection on synthetic data."""
    print("="*60)
    print("TEST 1: SNR Format Selector")
    print("="*60)

    from snr_format_selector import (
        QuantFormat,
        compute_snr_db,
        compute_snr_for_all_formats,
        select_best_format_for_channel,
    )

    # Create synthetic weight tensor
    torch.manual_seed(42)
    weights = torch.randn(256)  # Simulate one channel's weights

    print(f"Test tensor shape: {weights.shape}")
    print(f"Test tensor stats: min={weights.min():.4f}, max={weights.max():.4f}, std={weights.std():.4f}")

    # Compute SNR for all formats
    print("\nComputing SNR for all formats...")
    snrs = compute_snr_for_all_formats(weights)

    print("\nFormat SNR (higher is better):")
    for fmt, snr in sorted(snrs.items(), key=lambda x: -x[1]):
        print(f"  {fmt.value:20s}: {snr:8.2f} dB")

    # Select best format
    best = select_best_format_for_channel(weights)
    print(f"\nSelected format: {best.selected_format.value} (SNR: {best.snr_db:.2f} dB)")

    print("\nTEST 1 PASSED")
    return True


def test_chameleon_quantizer_init():
    """Test ChameleonQuantizer initialization without loading the full model."""
    print("\n" + "="*60)
    print("TEST 2: ChameleonQuantizer Initialization")
    print("="*60)

    from chameleon_quant import ChameleonQuantizer

    # Just test the class can be instantiated
    quantizer = ChameleonQuantizer(
        device="cpu",  # Use CPU for quick test
        min_snr_threshold=20.0,
    )

    print(f"Quantizer created successfully")
    print(f"  Device: {quantizer.device}")
    print(f"  Min SNR threshold: {quantizer.format_selector.min_snr_threshold}")
    print(f"  MX block size: {quantizer.mx_block_size}")

    print("\nTEST 2 PASSED")
    return True


def test_quantized_layer():
    """Test ChameleonQuantizedLinear with synthetic data."""
    print("\n" + "="*60)
    print("TEST 3: ChameleonQuantizedLinear")
    print("="*60)

    from snr_format_selector import SNRFormatSelector
    from chameleon_quant import ChameleonQuantizedLinear
    import torch.nn as nn

    # Create a simple linear layer
    torch.manual_seed(42)
    original = nn.Linear(64, 32)

    # Analyze with format selector
    selector = SNRFormatSelector(min_snr_threshold=20.0)
    format_info = selector.analyze_weights(
        weight_tensor=original.weight.data,
        layer_name="test_layer",
        is_conv=False,
        verbose=True,
    )

    # Create quantized layer
    quant_layer = ChameleonQuantizedLinear(original, format_info)

    # Test forward pass
    x = torch.randn(2, 64)
    y_original = original(x)
    y_quantized = quant_layer(x)

    # Compare
    diff = (y_original - y_quantized).abs()
    print(f"\nForward pass comparison:")
    print(f"  Original output shape: {y_original.shape}")
    print(f"  Quantized output shape: {y_quantized.shape}")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Mean absolute difference: {diff.mean():.6f}")

    print("\nTEST 3 PASSED")
    return True


def test_full_pipeline():
    """Test full pipeline with model loading (requires GPU and model weights)."""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline (requires GPU)")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping full pipeline test")
        return True

    from chameleon_quant import create_chameleon_quantizer

    print("Creating Chameleon quantizer...")
    quantizer = create_chameleon_quantizer(
        min_snr_threshold=20.0,
        device="cuda",
    )

    print("Loading SDXL model...")
    quantizer.load_model()

    print("\nStep 1: Injecting weight quantization...")
    quantizer.inject_weight_quantization(verbose=False)

    print("\nStep 2: Calibrating activations (this may take a few minutes)...")
    quantizer.calibrate_activations(
        num_calibration_samples=16,  # Use fewer for quick test
        num_inference_steps=10,
        verbose=False,
    )

    # Get stats
    stats = quantizer.get_quantization_stats()
    print(f"\nQuantization complete!")
    print(f"  Base method: {stats['base_method']}")
    print(f"  Quantized layers: {stats['num_quantized_layers']}")
    print(f"  Weight format distribution:")
    for fmt, count in stats['weight_format_distribution'].items():
        print(f"    {fmt.value}: {count} channels")
    print(f"  Activation format distribution:")
    for fmt, count in stats['activation_format_distribution'].items():
        print(f"    {fmt.value}: {count} channels")

    # Generate one test image
    print("\nGenerating test image...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        quantizer.generate_images(
            prompts=["A photo of a cat"],
            output_dir=tmpdir,
            num_inference_steps=10,  # Fast test
            batch_size=1,
        )

        # Check image was created
        images = os.listdir(tmpdir)
        print(f"Generated images: {images}")

        if len(images) > 0:
            print("\nTEST 4 PASSED")
        else:
            print("\nTEST 4 FAILED - no images generated")
            return False

    quantizer.cleanup()
    return True


def main():
    print("="*60)
    print("CHAMELEON QUANTIZATION TEST SUITE")
    print("="*60)
    print()

    tests = [
        ("SNR Format Selector", test_snr_format_selector),
        ("Quantizer Init", test_chameleon_quantizer_init),
        ("Quantized Layer", test_quantized_layer),
    ]

    # Add full pipeline test if requested
    if "--full" in sys.argv:
        tests.append(("Full Pipeline", test_full_pipeline))
    else:
        print("Note: Run with --full to test the complete pipeline (requires GPU)\n")

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nTEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
