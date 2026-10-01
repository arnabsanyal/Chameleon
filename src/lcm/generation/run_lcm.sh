#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Few-Step FP16 Baseline Generation (SDXL-Turbo)
# The few-step FP16 reference for the LCM-family table — same base model used by
# the MixDQ and Chameleon (few-step) entries — at a single, CFG-free step.
# Model: stabilityai/sdxl-turbo (Adversarial Diffusion Distillation)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Few-Step FP16 Baseline (SDXL-Turbo)"
echo "  Model  : stabilityai/sdxl-turbo"
echo "  Speed  : 1 step (CFG-free)"
echo "  Size   : 512x512"
echo "  Batch  : auto (VRAM-calibrated)"
echo "=============================================="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Warning: PyTorch not installed or CUDA not available"
}

# Shared paths
COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/lcm"
DEFAULT_MODEL="stabilityai/sdxl-turbo"

show_menu() {
    echo "Select operation:"
    echo ""
    echo "=== Quick Tests ==="
    echo "1) Quick test (100 images, default prompts, 1 step)"
    echo "2) Quick test with COCO captions (100 images, 1 step)"
    echo ""
    echo "=== COCO Evaluation ==="
    echo "3) COCO evaluation (5000 images, 1 step)"
    echo "4) COCO eval with custom settings"
    echo ""
    echo "=== Step Ablation ==="
    echo "5) Step count sweep (1 vs 2 vs 4 steps, 100 images each)"
    echo ""
    echo "=== Custom ==="
    echo "6) Custom generation (interactive)"
    echo ""
    echo "7) Exit"
    echo ""
}

# Quick test — default prompts
quick_test() {
    echo "Running quick test (100 images, default prompts, 1 step)..."
    python lcm_generation.py \
        --num-samples 100 \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --height 512 --width 512 \
        --output-dir "${BASE_OUTPUT}/quick_test"
    echo "Done! Check ${BASE_OUTPUT}/quick_test/"
}

# Quick test — COCO captions
quick_test_coco() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    echo "Running quick test (100 images, COCO captions, 1 step)..."
    python lcm_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --height 512 --width 512 \
        --output-dir "${BASE_OUTPUT}/quick_test_coco"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco/"
}

# Full COCO evaluation
coco_eval() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== SDXL-Turbo COCO Evaluation ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo ""
    echo "Configuration:"
    echo "  Model  : $DEFAULT_MODEL"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : 1"
    echo "  CFG    : 0.0 (CFG-free)"
    echo "  Size   : 512x512"
    echo "  Batch  : auto"
    echo "  Output : ${BASE_OUTPUT}/coco_eval"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python lcm_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps 1 \
            --guidance-scale 0.0 \
            --height 512 --width 512 \
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

    echo "=== SDXL-Turbo COCO Evaluation — Custom Settings ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Number of inference steps (default: 1):"
    read steps
    steps=${steps:-1}

    echo "Guidance scale (default: 0.0 — SDXL-Turbo is CFG-free):"
    read guidance
    guidance=${guidance:-0.0}

    echo "Output subdirectory name (default: coco_eval_custom):"
    read subdir
    subdir=${subdir:-coco_eval_custom}

    echo "Batch size (default: auto — VRAM-calibrated; enter a number to override):"
    read batch_size
    batch_size=${batch_size:-auto}

    echo ""
    echo "Configuration:"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : $steps"
    echo "  CFG    : $guidance"
    echo "  Size   : 512x512"
    echo "  Batch  : $batch_size"
    echo "  Output : ${BASE_OUTPUT}/${subdir}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python lcm_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps $steps \
            --guidance-scale $guidance \
            --height 512 --width 512 \
            --batch-size $batch_size \
            --output-dir "${BASE_OUTPUT}/${subdir}"
        echo "Done!"
    fi
}

# Step count sweep (SDXL-Turbo supports 1-4 steps)
step_sweep() {
    echo "=== SDXL-Turbo Step Count Sweep (1, 2, 4 steps — 100 images each) ==="
    echo ""
    echo "SDXL-Turbo produces high-quality images in a single step;"
    echo "this ablation shows quality vs step budget (Turbo degrades past ~4)."
    echo ""

    n=100
    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        echo "Use COCO captions? (y/n, default: n):"
        read use_coco
        if [[ $use_coco =~ ^[Yy]$ ]]; then
            coco_arg="--coco-captions $COCO_CAPTIONS"
        fi
    fi

    echo "Number of images per step setting (default: 100):"
    read n_input
    n=${n_input:-100}

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for steps in 1 2 4; do
            echo "--- Running with ${steps} step(s) ---"
            python lcm_generation.py \
                $coco_arg \
                --num-samples $n \
                --num-inference-steps $steps \
                --guidance-scale 0.0 \
                --height 512 --width 512 \
                --output-dir "${BASE_OUTPUT}/step_sweep_${steps}steps"
            echo "Done for ${steps} step(s). Check ${BASE_OUTPUT}/step_sweep_${steps}steps/"
        done
        echo ""
        echo "Step sweep complete!"
        echo "Results:"
        for steps in 1 2 4; do
            echo "  ${steps} steps → ${BASE_OUTPUT}/step_sweep_${steps}steps/"
        done
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom SDXL-Turbo Generation ==="
    echo ""

    echo "Model ID (default: stabilityai/sdxl-turbo):"
    read model_id
    model_id=${model_id:-stabilityai/sdxl-turbo}

    echo "Number of images to generate (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Number of inference steps (default: 1):"
    read steps
    steps=${steps:-1}

    echo "Guidance scale (default: 0.0 — SDXL-Turbo is CFG-free):"
    read guidance
    guidance=${guidance:-0.0}

    echo "Image height in pixels (default: 512):"
    read height
    height=${height:-512}

    echo "Image width in pixels (default: 512):"
    read width
    width=${width:-512}

    echo "Batch size (default: auto — VRAM-calibrated; enter a number to override):"
    read batch_size
    batch_size=${batch_size:-auto}

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

    cmd="python lcm_generation.py"
    cmd="$cmd --model-id $model_id"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --num-inference-steps $steps"
    cmd="$cmd --guidance-scale $guidance"
    cmd="$cmd --height $height"
    cmd="$cmd --width $width"
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
        5) step_sweep ;;
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
