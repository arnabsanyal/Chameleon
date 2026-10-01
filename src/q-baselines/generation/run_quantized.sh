#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Quantized Baseline Generation Run Script
# Quick launcher for Q-Diffusion and PTQ4DM quantized generation

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Quantized Baseline Generation Script"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Warning: PyTorch not installed or CUDA not available"
    echo "Install requirements: pip install -r requirements.txt"
}

# Function to display menu
show_menu() {
    echo "Select operation:"
    echo "1) Q-Diffusion W8A8 - Quick test (100 images)"
    echo "2) Q-Diffusion W4A8 - Quick test (100 images)"
    echo "3) PTQ4DM W8A8 - Quick test (100 images)"
    echo "4) Q-Diffusion W8A8 - Full benchmark (2048 images)"
    echo "5) PTQ4DM W8A8 - Full benchmark (2048 images)"
    echo "6) Both methods - Full benchmark"
    echo "7) COCO captions evaluation (Q-Diffusion)"
    echo "8) COCO captions evaluation (PTQ4DM)"
    echo "9) Compare all quantization configs"
    echo "10) Custom generation (interactive)"
    echo "11) Run example script"
    echo "12) Exit"
    echo ""
}

# Q-Diffusion quick test W8A8
qdiff_quick_w8a8() {
    echo "Running Q-Diffusion W8A8 quick test..."
    python quantized_generation.py \
        --method qdiffusion \
        --weight-bits 8 \
        --act-bits 8 \
        --num-samples 100 \
        --num-calibration-samples 16 \
        --calibration-steps 20 \
        --num-inference-steps 30 \
        --batch-size auto \
        --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/quick_test
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/quantized/quick_test/qdiffusion/"
}

# Q-Diffusion quick test W4A8
qdiff_quick_w4a8() {
    echo "Running Q-Diffusion W4A8 quick test..."
    python quantized_generation.py \
        --method qdiffusion \
        --weight-bits 4 \
        --act-bits 8 \
        --num-samples 100 \
        --num-calibration-samples 16 \
        --calibration-steps 20 \
        --num-inference-steps 30 \
        --batch-size auto \
        --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/quick_test
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/quantized/quick_test/qdiffusion/"
}

# PTQ4DM quick test
ptq4dm_quick() {
    echo "Running PTQ4DM W8A8 quick test..."
    python quantized_generation.py \
        --method ptq4dm \
        --weight-bits 8 \
        --act-bits 8 \
        --num-timestep-buckets 10 \
        --num-samples 100 \
        --num-calibration-samples 16 \
        --calibration-steps 20 \
        --num-inference-steps 30 \
        --batch-size auto \
        --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/quick_test
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/quantized/quick_test/ptq4dm/"
}

# Q-Diffusion full benchmark
qdiff_full() {
    echo "Running Q-Diffusion W8A8 full benchmark (2048 images)..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python quantized_generation.py \
            --method qdiffusion \
            --weight-bits 8 \
            --act-bits 8 \
            --num-samples 2048 \
            --num-calibration-samples 64 \
            --calibration-steps 20 \
            --num-inference-steps 50 \
            --batch-size auto \
            --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/benchmark
        echo "Done!"
    fi
}

# PTQ4DM full benchmark
ptq4dm_full() {
    echo "Running PTQ4DM W8A8 full benchmark (2048 images)..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python quantized_generation.py \
            --method ptq4dm \
            --weight-bits 8 \
            --act-bits 8 \
            --num-timestep-buckets 10 \
            --num-samples 2048 \
            --num-calibration-samples 64 \
            --calibration-steps 20 \
            --num-inference-steps 50 \
            --batch-size auto \
            --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/benchmark
        echo "Done!"
    fi
}

# Both methods benchmark
both_full() {
    echo "Running both Q-Diffusion and PTQ4DM benchmarks..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python quantized_generation.py \
            --method both \
            --weight-bits 8 \
            --act-bits 8 \
            --num-samples 2048 \
            --num-calibration-samples 64 \
            --calibration-steps 20 \
            --num-inference-steps 50 \
            --batch-size auto \
            --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/benchmark
        echo "Done!"
    fi
}

# COCO captions Q-Diffusion
coco_qdiff() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
    COCO_IMAGES="$CHAMELEON_DATA_ROOT/coco/val2014"

    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "Weight bits (4 or 8, default: 8):"
    read weight_bits
    weight_bits=${weight_bits:-8}

    echo "Number of samples to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_ind
    start_ind=${start_ind:-0}

    echo ""
    echo "Configuration: W${weight_bits}A8 Q-Diffusion"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python quantized_generation.py \
            --method qdiffusion \
            --weight-bits $weight_bits \
            --act-bits 8 \
            --num-samples $num_samples \
            --start-ind $start_ind \
            --num-calibration-samples 128 \
            --calibration-steps 20 \
            --num-inference-steps 50 \
            --batch-size auto \
            --coco-captions "$COCO_CAPTIONS" \
            --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/coco_eval
        echo "Done!"
    fi
}

# COCO captions PTQ4DM
coco_ptq4dm() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"

    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "Weight bits (4 or 8, default: 8):"
    read weight_bits
    weight_bits=${weight_bits:-8}

    echo "Number of samples to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_ind
    start_ind=${start_ind:-0}

    echo ""
    echo "Configuration: W${weight_bits}A8 PTQ4DM"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python quantized_generation.py \
            --method ptq4dm \
            --weight-bits $weight_bits \
            --act-bits 8 \
            --num-timestep-buckets 10 \
            --num-samples $num_samples \
            --start-ind $start_ind \
            --num-calibration-samples 128 \
            --calibration-steps 20 \
            --num-inference-steps 50 \
            --batch-size auto \
            --coco-captions "$COCO_CAPTIONS" \
            --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/coco_eval
        echo "Done!"
    fi
}

# Compare all configs
compare_all() {
    echo "=== Comparing All Quantization Configurations ==="
    echo "This will generate images with multiple configs for comparison"
    echo "Configs: W8A8, W4A8 for both Q-Diffusion and PTQ4DM"

    read -p "Number of samples per config (default: 100): " num_samples
    num_samples=${num_samples:-100}

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Q-Diffusion W8A8
        echo ">>> Q-Diffusion W8A8"
        python quantized_generation.py --method qdiffusion --weight-bits 8 --act-bits 8 \
            --num-samples $num_samples --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/comparison --batch-size auto

        # Q-Diffusion W4A8
        echo ">>> Q-Diffusion W4A8"
        python quantized_generation.py --method qdiffusion --weight-bits 4 --act-bits 8 \
            --num-samples $num_samples --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/comparison --batch-size auto

        # PTQ4DM W8A8
        echo ">>> PTQ4DM W8A8"
        python quantized_generation.py --method ptq4dm --weight-bits 8 --act-bits 8 \
            --num-samples $num_samples --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/comparison --batch-size auto

        # PTQ4DM W4A8
        echo ">>> PTQ4DM W4A8"
        python quantized_generation.py --method ptq4dm --weight-bits 4 --act-bits 8 \
            --num-samples $num_samples --output-dir $CHAMELEON_OUTPUT_ROOT/quantized/comparison --batch-size auto

        echo "Comparison complete! Results in $CHAMELEON_OUTPUT_ROOT/quantized/comparison/"
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom Quantized Generation ==="

    echo "Method (qdiffusion/ptq4dm/both):"
    read method

    echo "Weight bits (4 or 8, default: 8):"
    read weight_bits
    weight_bits=${weight_bits:-8}

    echo "Activation bits (default: 8):"
    read act_bits
    act_bits=${act_bits:-8}

    echo "Number of samples (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Number of calibration samples (default: 32):"
    read calib_samples
    calib_samples=${calib_samples:-32}

    echo "Number of inference steps (default: 50):"
    read inf_steps
    inf_steps=${inf_steps:-50}

    echo "Output directory (default: $CHAMELEON_OUTPUT_ROOT/quantized/custom):"
    read output_dir
    output_dir=${output_dir:-$CHAMELEON_OUTPUT_ROOT/quantized/custom}

    cmd="python quantized_generation.py --method $method --weight-bits $weight_bits --act-bits $act_bits"
    cmd="$cmd --num-samples $num_samples --num-calibration-samples $calib_samples"
    cmd="$cmd --num-inference-steps $inf_steps --output-dir $output_dir"

    echo "Running: $cmd"
    eval $cmd
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-12]: " choice
    echo ""

    case $choice in
        1) qdiff_quick_w8a8 ;;
        2) qdiff_quick_w4a8 ;;
        3) ptq4dm_quick ;;
        4) qdiff_full ;;
        5) ptq4dm_full ;;
        6) both_full ;;
        7) coco_qdiff ;;
        8) coco_ptq4dm ;;
        9) compare_all ;;
        10) custom_generation ;;
        11)
            echo "Running example script..."
            python example_usage.py
            ;;
        12)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    echo ""
done
