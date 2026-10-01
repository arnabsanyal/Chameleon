#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Q-DiT: Post-Training Quantized PixArt-alpha (DiT) Generation
# Implements group weight quantization + sample-wise dynamic activation quantization
# Reference: Chen et al., arXiv:2406.17343

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Q-DiT: Quantized DiT Generation"
echo "  Weight quantization  : group (offline)"
echo "  Activation quant.    : sample-wise dynamic (online)"
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
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/q-dit"
DEFAULT_MODEL="PixArt-alpha/PixArt-XL-2-1024-MS"

# Paper's group size search space: {32, 64, 128, 192, 288}
# Default: 128 (best for DiT-XL/2 256x256 per Table 1)

ask_quant_config() {
    echo "Weight bits (4, 6, or 8; default: 8):"
    echo "  8 → W8A8  near-lossless"
    echo "  6 → W6A8  slight drop"
    echo "  4 → W4A8  more aggressive"
    read weight_bits
    weight_bits=${weight_bits:-8}
    if [ "$weight_bits" != "4" ] && [ "$weight_bits" != "6" ] && [ "$weight_bits" != "8" ]; then
        echo "Invalid. Using 8."
        weight_bits=8
    fi

    echo "Activation bits (8; default: 8):"
    read act_bits
    act_bits=${act_bits:-8}

    echo "Group size (32/64/128/192/288; default: 128):"
    read group_size
    group_size=${group_size:-128}
}

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
    echo "=== Ablation / Comparison ==="
    echo "5) W8A8 vs W4A8 side-by-side (100 images each)"
    echo "6) Group size sweep (g=32,128,288; 50 images each)"
    echo ""
    echo "=== Custom ==="
    echo "7) Custom generation (interactive)"
    echo ""
    echo "8) Exit"
    echo ""
}

# Quick test — default prompts
quick_test() {
    ask_quant_config
    echo "Running quick test (100 images, W${weight_bits}A${act_bits} g=${group_size})..."
    python q_dit_generation.py \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --weight-bits $weight_bits \
        --act-bits $act_bits \
        --group-size $group_size \
        --output-dir "${BASE_OUTPUT}/quick_test"
    echo "Done! Check ${BASE_OUTPUT}/quick_test/w${weight_bits}a${act_bits}_g${group_size}/"
}

# Quick test — COCO captions
quick_test_coco() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    ask_quant_config
    echo "Running quick test (100 images, COCO, W${weight_bits}A${act_bits} g=${group_size})..."
    python q_dit_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --weight-bits $weight_bits \
        --act-bits $act_bits \
        --group-size $group_size \
        --output-dir "${BASE_OUTPUT}/quick_test_coco"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco/w${weight_bits}a${act_bits}_g${group_size}/"
}

# Full COCO evaluation
coco_eval() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== Q-DiT COCO Evaluation ==="
    echo ""

    echo "Number of images (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    ask_quant_config

    echo ""
    echo "Configuration:"
    echo "  Model  : $DEFAULT_MODEL"
    echo "  Quant  : W${weight_bits}A${act_bits}, group=${group_size}"
    echo "  Images : $num_samples (start=$start_idx)"
    echo "  Steps  : 20, CFG: 4.5"
    echo "  Output : ${BASE_OUTPUT}/coco_eval/w${weight_bits}a${act_bits}_g${group_size}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python q_dit_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps 20 \
            --guidance-scale 4.5 \
            --batch-size $batch_size \
            --weight-bits $weight_bits \
            --act-bits $act_bits \
            --group-size $group_size \
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

    echo "=== Q-DiT COCO Eval — Custom Settings ==="
    echo ""

    echo "Number of images (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Inference steps (default: 20):"
    read steps
    steps=${steps:-20}

    echo "Guidance scale (default: 4.5):"
    read guidance
    guidance=${guidance:-4.5}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    ask_quant_config

    echo "Output subdirectory (default: coco_eval_custom):"
    read subdir
    subdir=${subdir:-coco_eval_custom}

    echo ""
    echo "Configuration:"
    echo "  Quant  : W${weight_bits}A${act_bits}, group=${group_size}"
    echo "  Images : $num_samples (start=$start_idx)"
    echo "  Steps  : $steps, CFG: $guidance"
    echo "  Output : ${BASE_OUTPUT}/${subdir}/w${weight_bits}a${act_bits}_g${group_size}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python q_dit_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps $steps \
            --guidance-scale $guidance \
            --batch-size $batch_size \
            --weight-bits $weight_bits \
            --act-bits $act_bits \
            --group-size $group_size \
            --output-dir "${BASE_OUTPUT}/${subdir}"
        echo "Done!"
    fi
}

# W8A8 vs W4A8 side-by-side
ablation_bits() {
    echo "=== W8A8 vs W4A8 Ablation (100 images each) ==="
    echo ""

    n=100
    steps=20
    cfg=4.5
    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        coco_arg="--coco-captions $COCO_CAPTIONS"
    fi

    echo "Group size (default: 128):"
    read group_size
    group_size=${group_size:-128}

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "--- W8A8 (near-lossless) ---"
        python q_dit_generation.py \
            $coco_arg --num-samples $n \
            --num-inference-steps $steps --guidance-scale $cfg \
            --weight-bits 8 --act-bits 8 --group-size $group_size \
            --output-dir "${BASE_OUTPUT}/ablation_bits"

        echo "--- W4A8 (aggressive) ---"
        python q_dit_generation.py \
            $coco_arg --num-samples $n \
            --num-inference-steps $steps --guidance-scale $cfg \
            --weight-bits 4 --act-bits 8 --group-size $group_size \
            --output-dir "${BASE_OUTPUT}/ablation_bits"

        echo "Done! Results in ${BASE_OUTPUT}/ablation_bits/"
    fi
}

# Group size sweep
ablation_group_size() {
    echo "=== Group Size Sweep (g=32, 128, 288; 50 images each) ==="
    echo ""

    n=50
    steps=20
    cfg=4.5
    wb=8
    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        coco_arg="--coco-captions $COCO_CAPTIONS"
    fi

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for gs in 32 128 288; do
            echo "--- Group size = $gs ---"
            python q_dit_generation.py \
                $coco_arg --num-samples $n \
                --num-inference-steps $steps --guidance-scale $cfg \
                --weight-bits $wb --act-bits 8 --group-size $gs \
                --output-dir "${BASE_OUTPUT}/ablation_group"
        done
        echo "Done! Results in ${BASE_OUTPUT}/ablation_group/"
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom Q-DiT Generation ==="
    echo ""

    echo "Model ID (default: PixArt-alpha/PixArt-XL-2-1024-MS):"
    read model_id
    model_id=${model_id:-PixArt-alpha/PixArt-XL-2-1024-MS}

    echo "Number of images (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Inference steps (default: 20):"
    read steps
    steps=${steps:-20}

    echo "Guidance scale (default: 4.5):"
    read guidance
    guidance=${guidance:-4.5}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    ask_quant_config

    echo "Output subdirectory (default: custom):"
    read subdir
    subdir=${subdir:-custom}

    echo "Use COCO captions? (y/n, default: n):"
    read use_coco
    coco_arg=""
    if [[ $use_coco =~ ^[Yy]$ ]] && [ -f "$COCO_CAPTIONS" ]; then
        coco_arg="--coco-captions $COCO_CAPTIONS"
    fi

    echo "Prompt file? (path or enter to skip):"
    read prompt_file
    prompt_arg=""
    if [ -n "$prompt_file" ] && [ -f "$prompt_file" ]; then
        prompt_arg="--prompt-file $prompt_file"
    fi

    echo "Verbose output? (y/n, default: n):"
    read verbose
    verbose_arg=""
    if [[ $verbose =~ ^[Yy]$ ]]; then
        verbose_arg="--verbose"
    fi

    cmd="python q_dit_generation.py"
    cmd="$cmd --model-id $model_id"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --num-inference-steps $steps"
    cmd="$cmd --guidance-scale $guidance"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --weight-bits $weight_bits"
    cmd="$cmd --act-bits $act_bits"
    cmd="$cmd --group-size $group_size"
    cmd="$cmd --output-dir ${BASE_OUTPUT}/${subdir}"
    cmd="$cmd $coco_arg $prompt_arg $verbose_arg"

    echo ""
    echo "Running: $cmd"
    eval $cmd
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    echo ""

    case $choice in
        1) quick_test ;;
        2) quick_test_coco ;;
        3) coco_eval ;;
        4) coco_eval_custom ;;
        5) ablation_bits ;;
        6) ablation_group_size ;;
        7) custom_generation ;;
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
