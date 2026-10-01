#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Chameleon-LCM: Few-Step Quantized Diffusion Generation (Chameleon-on-MixDQ)
#
# Chameleon-LCM builds ON TOP of MixDQ: it keeps MixDQ's calibrated static
# activation quantisation (act_scales + BOS + mixed precision) and replaces only
# the *weight* fold with Chameleon's adaptive palette (per-output-channel SNR
# selection @ W8; per-layer (group_size, format) search @ W4).
#
# Every option here passes --chameleon-weights and writes to the chameleon-lcm
# output dir.  For the plain MixDQ baseline use the separate folder:
#     src/mixdq-lcm/generation/run_mixdq_lcm.sh
#
# Base : stabilityai/sdxl-turbo  —  1 step, CFG-free (guidance 0.0), 512×512

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Chameleon-LCM: Few-Step Quantized Generation"
echo "  Base   : stabilityai/sdxl-turbo"
echo "  Method : Chameleon weights on MixDQ activations"
echo "  Speed  : 1 step, CFG-free"
echo "  Size   : 512×512"
echo "  (MixDQ baseline lives in ../../mixdq-lcm)"
echo "=============================================="
echo ""

if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Warning: PyTorch not installed or CUDA not available"
}

python -c "import mixdq_extension" 2>/dev/null || {
    echo ""
    echo "  WARNING: mixdq-extension not found (W8A8 falls back to fake-quant)."
    echo "  Install with: pip install -i https://pypi.org/simple/ mixdq-extension"
    echo ""
}

# Shared paths
COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/chameleon-lcm"
BASE_MODEL="stabilityai/sdxl-turbo"

# Common generation flags for every Chameleon-LCM run.
CHAMELEON_FLAGS="--chameleon-weights --num-inference-steps 1 --guidance-scale 0.0 --height 512 --width 512"

show_menu() {
    echo "Select operation (all runs use the Chameleon weight palette):"
    echo ""
    echo "=== Quick Tests (100 images) ==="
    echo "1) Quick test — Chameleon W8A8 (default prompts)"
    echo "2) Quick test — Chameleon W8A8 (COCO captions)"
    echo "3) Quick test — Chameleon W4A8 (COCO captions)"
    echo ""
    echo "=== COCO Evaluation ==="
    echo "4) COCO eval — Chameleon W8A8 (per-channel SNR weight selection)"
    echo "5) COCO eval — Chameleon W4A8 (per-layer (group,format) weight search)"
    echo ""
    echo "=== Custom ==="
    echo "6) Custom generation (interactive)"
    echo ""
    echo "7) Exit"
    echo ""
    echo "Note: W8A8 weights ≈ lossless (MixDQ already ~21.4 FID); the headline"
    echo "opportunity is W4A8 beating MixDQ's 24.50 with the adaptive palette."
    echo ""
}

# Quick test — default prompts
quick_test() {
    local w_bit="$1"
    echo "Running quick test (100 images, Chameleon W${w_bit}A8, default prompts)..."
    python chameleon_lcm_generation.py \
        --num-samples 100 \
        --w-bit $w_bit --a-bit 8 $CHAMELEON_FLAGS \
        --output-dir "${BASE_OUTPUT}/quick_test_w${w_bit}a8"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_w${w_bit}a8/"
}

# Quick test — COCO captions
quick_test_coco() {
    local w_bit="$1"
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    echo "Running quick test (100 images, Chameleon W${w_bit}A8, COCO captions)..."
    python chameleon_lcm_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --w-bit $w_bit --a-bit 8 $CHAMELEON_FLAGS \
        --output-dir "${BASE_OUTPUT}/quick_test_coco_w${w_bit}a8"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco_w${w_bit}a8/"
}

# COCO evaluation — writes to the slot score_all.sh expects
coco_eval() {
    local w_bit="$1"
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== Chameleon-LCM W${w_bit}A8 COCO Evaluation ==="
    echo ""
    if [ "$w_bit" = "8" ]; then
        echo "Weights: per-output-channel SNR over {INT8_SYM, INT8_ASYM, MXINT8}."
    else
        echo "Weights: per-layer (group_size, format) over"
        echo "  {INT4_ASYM, NF4, FP4_E2M1, MXINT4, MXFP4_E2M1} x {32,64,128,192,288}."
    fi
    echo "Activations: MixDQ calibrated static scales (untouched)."
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    local outdir="${BASE_OUTPUT}/coco_eval_w${w_bit}a8"
    echo ""
    echo "Configuration:"
    echo "  Model  : $BASE_MODEL"
    echo "  Quant  : Chameleon W${w_bit}A8 (adaptive weights + MixDQ activations)"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : 1   CFG: 0.0   Size: 512×512   Batch: auto"
    echo "  Output : ${outdir}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_lcm_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --w-bit $w_bit --a-bit 8 $CHAMELEON_FLAGS \
            --output-dir "${outdir}"
        echo "Done! Score with the same FID/CLIP harness as the other entries."
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom Chameleon-LCM Generation ==="
    echo ""

    echo "Number of images (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Weight bit-width — 4 or 8 (default: 4):"
    read w_bit
    w_bit=${w_bit:-4}
    if [[ "$w_bit" != "4" && "$w_bit" != "8" ]]; then
        echo "Invalid w_bit=$w_bit (must be 4 or 8). Using 4."
        w_bit=4
    fi

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

    echo "Prompt file? (path or enter to skip):"
    read prompt_file
    prompt_arg=""
    if [ -n "$prompt_file" ] && [ -f "$prompt_file" ]; then
        prompt_arg="--prompt-file $prompt_file"
    fi

    cmd="python chameleon_lcm_generation.py"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --w-bit $w_bit --a-bit 8 $CHAMELEON_FLAGS"
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
        1) quick_test 8 ;;
        2) quick_test_coco 8 ;;
        3) quick_test_coco 4 ;;
        4) coco_eval 8 ;;
        5) coco_eval 4 ;;
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
