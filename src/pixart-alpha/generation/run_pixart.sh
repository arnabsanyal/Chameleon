#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# PixArt-alpha: DiT-based Text-to-Image Baseline Generation
# Uses PixArt-alpha/PixArt-XL-2-1024-MS (Diffusion Transformer, T5 encoder)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PixArt-alpha: DiT T2I Baseline Generation"
echo "=========================================="
echo ""

# Check Python and CUDA
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Warning: PyTorch not installed or CUDA not available"
}

# Shared paths
COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/pixart-alpha-dit"
DEFAULT_MODEL="PixArt-alpha/PixArt-XL-2-1024-MS"

show_menu() {
    echo "Select operation:"
    echo ""
    echo "=== Quick Tests ==="
    echo "1) Quick test (100 images, default prompts)"
    echo "2) Quick test with COCO captions (100 images)"
    echo ""
    echo "=== COCO Evaluation ==="
    echo "3) COCO evaluation (5000 images)"
    echo "4) COCO eval with custom settings"
    echo ""
    echo "=== Model Variants ==="
    echo "5) Generate with 512x512 model"
    echo ""
    echo "=== Custom ==="
    echo "6) Custom generation (interactive)"
    echo ""
    echo "7) Exit"
    echo ""
}

# Quick test with default prompts
quick_test() {
    echo "Running quick test (100 images, default prompts)..."
    python pixart_generation.py \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --output-dir "${BASE_OUTPUT}/quick_test"
    echo "Done! Check ${BASE_OUTPUT}/quick_test/"
}

# Quick test with COCO captions
quick_test_coco() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    echo "Running quick test (100 images, COCO captions)..."
    python pixart_generation.py \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --coco-captions "$COCO_CAPTIONS" \
        --output-dir "${BASE_OUTPUT}/quick_test_coco"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco/"
}

# Full COCO evaluation (5000 images)
coco_eval() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== COCO Evaluation ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    echo ""
    echo "Configuration:"
    echo "  Model  : $DEFAULT_MODEL"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : 20"
    echo "  CFG    : 4.5"
    echo "  Batch  : $batch_size"
    echo "  Output : ${BASE_OUTPUT}/coco_eval"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python pixart_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps 20 \
            --guidance-scale 4.5 \
            --batch-size $batch_size \
            --output-dir "${BASE_OUTPUT}/coco_eval"
        echo "Done!"
    fi
}

# COCO eval with custom settings
coco_eval_custom() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== COCO Evaluation with Custom Settings ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Number of inference steps (default: 20):"
    read steps
    steps=${steps:-20}

    echo "Guidance scale (default: 4.5, paper uses 4.5 for 1024px):"
    read guidance
    guidance=${guidance:-4.5}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    echo "Output subdirectory name (default: coco_eval_custom):"
    read subdir
    subdir=${subdir:-coco_eval_custom}

    echo ""
    echo "Configuration:"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : $steps"
    echo "  CFG    : $guidance"
    echo "  Batch  : $batch_size"
    echo "  Output : ${BASE_OUTPUT}/${subdir}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python pixart_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps $steps \
            --guidance-scale $guidance \
            --batch-size $batch_size \
            --output-dir "${BASE_OUTPUT}/${subdir}"
        echo "Done!"
    fi
}

# 512x512 model variant
quick_test_512() {
    echo "Running with 512x512 model variant..."
    echo "Number of images (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        echo "Use COCO captions? (y/n, default: n):"
        read use_coco
        if [[ $use_coco =~ ^[Yy]$ ]]; then
            coco_arg="--coco-captions $COCO_CAPTIONS"
        fi
    fi

    python pixart_generation.py \
        --model-id "PixArt-alpha/PixArt-XL-2-512x512" \
        --num-samples $num_samples \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        $coco_arg \
        --output-dir "${BASE_OUTPUT}/quick_test_512"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_512/"
}

# Custom generation
custom_generation() {
    echo "=== Custom PixArt-alpha Generation ==="
    echo ""

    echo "Model ID (default: PixArt-alpha/PixArt-XL-2-1024-MS):"
    echo "  Options: PixArt-alpha/PixArt-XL-2-1024-MS"
    echo "           PixArt-alpha/PixArt-XL-2-512x512"
    read model_id
    model_id=${model_id:-PixArt-alpha/PixArt-XL-2-1024-MS}

    echo "Number of images to generate (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Number of inference steps (default: 20):"
    read steps
    steps=${steps:-20}

    echo "Guidance scale (default: 4.5):"
    read guidance
    guidance=${guidance:-4.5}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    echo "Output subdirectory (default: custom):"
    read subdir
    subdir=${subdir:-custom}

    echo "Use COCO captions? (y/n, default: n):"
    read use_coco
    coco_arg=""
    if [[ $use_coco =~ ^[Yy]$ ]]; then
        if [ -f "$COCO_CAPTIONS" ]; then
            coco_arg="--coco-captions $COCO_CAPTIONS"
        else
            echo "COCO captions not found at $COCO_CAPTIONS, using default prompts"
        fi
    fi

    echo "Prompt file (press enter to skip):"
    read prompt_file
    prompt_arg=""
    if [ -n "$prompt_file" ] && [ -f "$prompt_file" ]; then
        prompt_arg="--prompt-file $prompt_file"
    fi

    cmd="python pixart_generation.py"
    cmd="$cmd --model-id $model_id"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --num-inference-steps $steps"
    cmd="$cmd --guidance-scale $guidance"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --output-dir ${BASE_OUTPUT}/${subdir}"
    cmd="$cmd $coco_arg $prompt_arg"

    echo ""
    echo "Running: $cmd"
    eval $cmd
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-7]: " choice
    echo ""

    case $choice in
        1) quick_test ;;
        2) quick_test_coco ;;
        3) coco_eval ;;
        4) coco_eval_custom ;;
        5) quick_test_512 ;;
        6) custom_generation ;;
        7)
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
