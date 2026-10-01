#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Baseline Generation Run Script
# Quick launcher for common baseline generation tasks

set -e  # Exit on error

echo "=========================================="
echo "Baseline Generation Script"
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
    echo "1) Generate 100 sample images (quick test)"
    echo "2) Generate 2048 images for FID (research baseline)"
    echo "3) Generate with COCO captions (proper FID evaluation)"
    echo "4) Generate videos from existing images"
    echo "5) Generate both images and videos"
    echo "6) Run example script"
    echo "7) Custom generation (interactive)"
    echo "8) Exit"
    echo ""
}

# Function for quick test
quick_test() {
    echo "Running quick test with 100 samples..."
    python baseline_generation.py \
        --mode image \
        --num-samples 100 \
        --batch-size 16 \
        --num-inference-steps 30 \
        --output-dir $CHAMELEON_OUTPUT_ROOT/quick_test
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/quick_test/images/"
}

# Function for research baseline
research_baseline() {
    echo "Generating 2048 images for FID calculation..."
    echo "This will take a while (est. 2-4 hours on A100)..."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python baseline_generation.py \
            --mode image \
            --num-samples 2048 \
            --batch-size 16 \
            --num-inference-steps 50 \
            --output-dir $CHAMELEON_OUTPUT_ROOT/research_baseline
        echo "Done! Check $CHAMELEON_OUTPUT_ROOT/research_baseline/images/"
    fi
}

# Function for COCO captions-based generation (proper FID evaluation)
coco_baseline() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
    COCO_IMAGES="$CHAMELEON_DATA_ROOT/coco/val2014"

    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        echo "Download with:"
        echo "  cd $CHAMELEON_DATA_ROOT/coco"
        echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        echo "  unzip annotations_trainval2014.zip"
        return
    fi

    echo "Generating images using COCO captions for proper FID evaluation..."
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
            --mode image \
            --num-samples $num_samples \
            --start-ind $start_ind \
            --batch-size 16 \
            --num-inference-steps 50 \
            --coco-captions "$COCO_CAPTIONS" \
            --output-dir $CHAMELEON_OUTPUT_ROOT/coco_baseline

        echo ""
        echo "Done! Images saved to $CHAMELEON_OUTPUT_ROOT/coco_baseline/images/"
        echo ""
        echo "To compute FID, run:"
        echo "  python -c \"from cleanfid import fid; print(f'FID: {fid.compute_fid(\\\"$CHAMELEON_OUTPUT_ROOT/coco_baseline/images\\\", \\\"$COCO_IMAGES\\\")}')\" "
    fi
}

# Function for video generation
video_generation() {
    echo "Enter path to input images directory:"
    read input_dir
    if [ ! -d "$input_dir" ]; then
        echo "Error: Directory not found"
        return
    fi

    echo "Number of videos to generate:"
    read num_samples

    python baseline_generation.py \
        --mode video \
        --input-images-dir "$input_dir" \
        --num-samples "$num_samples" \
        --output-dir $CHAMELEON_OUTPUT_ROOT/video_outputs
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/video_outputs/videos/"
}

# Function for both images and videos
both_generation() {
    echo "Number of samples:"
    read num_samples

    python baseline_generation.py \
        --mode both \
        --num-samples "$num_samples" \
        --output-dir $CHAMELEON_OUTPUT_ROOT/full_baseline
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/full_baseline/"
}

# Function for custom generation
custom_generation() {
    echo "=== Custom Generation ==="

    echo "Mode (image/video/both):"
    read mode

    echo "Number of samples:"
    read num_samples

    echo "Output directory (default: $CHAMELEON_OUTPUT_ROOT/custom_outputs):"
    read output_dir
    output_dir=${output_dir:-$CHAMELEON_OUTPUT_ROOT/custom_outputs}

    if [ "$mode" = "image" ] || [ "$mode" = "both" ]; then
        echo "Number of inference steps (default: 50):"
        read steps
        steps=${steps:-50}

        echo "Guidance scale (default: 7.5):"
        read guidance
        guidance=${guidance:-7.5}

        cmd="python baseline_generation.py --mode $mode --num-samples $num_samples --output-dir $output_dir --num-inference-steps $steps --guidance-scale $guidance"
    else
        echo "Input images directory:"
        read input_dir
        cmd="python baseline_generation.py --mode $mode --num-samples $num_samples --output-dir $output_dir --input-images-dir $input_dir"
    fi

    echo "Running: $cmd"
    eval $cmd
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    echo ""

    case $choice in
        1)
            quick_test
            ;;
        2)
            research_baseline
            ;;
        3)
            coco_baseline
            ;;
        4)
            video_generation
            ;;
        5)
            both_generation
            ;;
        6)
            echo "Running example script..."
            python example_usage.py
            ;;
        7)
            custom_generation
            ;;
        8)
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
