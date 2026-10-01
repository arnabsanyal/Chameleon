#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# FP16 SDXL baseline: quick launcher for common image-generation tasks

set -e  # Exit on error
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=========================================="
echo "FP16 SDXL Baseline Generation"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Activate the 'chameleon' conda environment first."
    exit 1
fi

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Warning: PyTorch not installed or CUDA not available"
    echo "See install/install.sh in the repository root"
}

show_menu() {
    echo "Select operation:"
    echo "1) Generate 100 sample images (quick test)"
    echo "2) Generate 2048 images for FID (research baseline)"
    echo "3) Generate with COCO captions (FID evaluation)"
    echo "4) Run example script"
    echo "5) Custom generation (interactive)"
    echo "6) Exit"
    echo ""
}

quick_test() {
    echo "Running quick test with 100 samples..."
    python baseline_generation.py \
        --num-samples 100 \
        --batch-size 16 \
        --num-inference-steps 30 \
        --output-dir "$CHAMELEON_OUTPUT_ROOT/quick_test"
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/quick_test/images/"
}

research_baseline() {
    echo "Generating 2048 images for FID calculation..."
    echo "This will take a while (est. 2-4 hours on A100)..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python baseline_generation.py \
            --num-samples 2048 \
            --batch-size 16 \
            --num-inference-steps 50 \
            --output-dir "$CHAMELEON_OUTPUT_ROOT/research_baseline"
        echo "Done! Check $CHAMELEON_OUTPUT_ROOT/research_baseline/images/"
    fi
}

coco_baseline() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
    COCO_IMAGES="$CHAMELEON_DATA_ROOT/coco/val2014"

    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        echo "Download with:"
        echo "  mkdir -p $CHAMELEON_DATA_ROOT/coco && cd $CHAMELEON_DATA_ROOT/coco"
        echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        echo "  unzip annotations_trainval2014.zip"
        return
    fi

    echo "Generating images from COCO captions for FID evaluation..."
    echo "Number of samples to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0, use to resume from a specific point):"
    read start_ind
    start_ind=${start_ind:-0}

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python baseline_generation.py \
            --num-samples $num_samples \
            --start-ind $start_ind \
            --batch-size 16 \
            --num-inference-steps 50 \
            --coco-captions "$COCO_CAPTIONS" \
            --output-dir "$CHAMELEON_OUTPUT_ROOT/coco_baseline"

        echo ""
        echo "Done! Images saved to $CHAMELEON_OUTPUT_ROOT/coco_baseline/images/"
        echo ""
        echo "To score (clean-FID vs. val2014 and CLIP), run from the repository root:"
        echo "  python src/paper_figures/score_24k.py $CHAMELEON_OUTPUT_ROOT/coco_baseline/images --label sdxl_fp16"
    fi
}

custom_generation() {
    echo "=== Custom Generation ==="

    echo "Number of samples:"
    read num_samples

    echo "Output directory (default: $CHAMELEON_OUTPUT_ROOT/custom_outputs):"
    read output_dir
    output_dir=${output_dir:-$CHAMELEON_OUTPUT_ROOT/custom_outputs}

    echo "Number of inference steps (default: 50):"
    read steps
    steps=${steps:-50}

    echo "Guidance scale (default: 7.5):"
    read guidance
    guidance=${guidance:-7.5}

    python baseline_generation.py \
        --num-samples "$num_samples" \
        --output-dir "$output_dir" \
        --num-inference-steps "$steps" \
        --guidance-scale "$guidance"
}

while true; do
    show_menu
    read -p "Enter choice [1-6]: " choice
    echo ""

    case $choice in
        1) quick_test ;;
        2) research_baseline ;;
        3) coco_baseline ;;
        4)
            echo "Running example script..."
            python example_usage.py
            ;;
        5) custom_generation ;;
        6)
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
