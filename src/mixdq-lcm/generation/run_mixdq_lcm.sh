#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# MixDQ: Mixed-Precision Quantized Few-Step Diffusion Generation
# Quantizes SDXL Turbo UNet in-place via pipe.quantize_unet().
# Default: W8A8 — ~2x weight compression, near-lossless quality.
# W4A8   — ~4x weight compression, minor quality trade-off.
#
# Reference: Zhao et al., "MixDQ", ECCV 2024  arXiv:2405.17873
# Model    : stabilityai/sdxl-turbo + nics-efc/MixDQ custom pipeline
# Steps    : 1 (CFG-free, guidance_scale=0.0)
# Size     : 512×512
#
# Requirement: pip install -i https://pypi.org/simple/ mixdq-extension
#   (officially supports Python 3.8–3.10; for 3.11+ build from source)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "MixDQ: Quantized Few-Step Diffusion Generation"
echo "  Base   : stabilityai/sdxl-turbo"
echo "  Quant  : W8A8 or W4A8 (MixDQ PTQ)"
echo "  Speed  : 1 step, CFG-free"
echo "  Size   : 512×512"
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

# Check mixdq-extension
python -c "import mixdq_extension" 2>/dev/null || {
    echo ""
    echo "  WARNING: mixdq-extension not found."
    echo "  Install with: pip install -i https://pypi.org/simple/ mixdq-extension"
    echo "  (Supports Python 3.8–3.10; for 3.11+ build from source:"
    echo "   https://github.com/A-suozhang/MixDQ)"
    echo ""
}

# Shared paths
COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/mixdq-lcm"
BASE_MODEL="stabilityai/sdxl-turbo"

# MixDQ sensitivity-derived mixed-precision configs (vendored from
# A-suozhang/MixDQ @ mixed_precision_scripts/mixed_percision_config/sdxl_turbo/).
# weight_4.00 averages ~4 bits/weight via per-layer W2/W4/W8 mix; act_8.00 is
# uniform A8 (matches the paper's W4A8 result).
W4A8_W_CONFIG="${SCRIPT_DIR}/configs/weight_4.00.yaml"
W4A8_A_CONFIG="${SCRIPT_DIR}/configs/act_8.00.yaml"

show_menu() {
    echo "Select operation:"
    echo ""
    echo "=== Quick Tests ==="
    echo "1) Quick test — W8A8 (100 images, default prompts)"
    echo "2) Quick test — W8A8 (100 images, COCO captions)"
    echo "3) Quick test — W4A8 (100 images, COCO captions, fake-quant)"
    echo ""
    echo "=== COCO Evaluation ==="
    echo "4) COCO evaluation — W8A8 (5000 images)"
    echo "5) COCO evaluation — W4A8 (5000 images, fake-quant)"
    echo ""
    echo "=== Custom ==="
    echo "6) Custom generation (interactive)"
    echo ""
    echo "7) Exit"
    echo ""
    echo "Note: W8A8 uses the accelerated qlinear/qconv2d kernels shipped with"
    echo "the public MixDQ release. W4A8 runs as fake-quant (no W4 kernel"
    echo "exists in the release); it reuses the W4 scales already present in"
    echo "quant_para_wsym_fp16.pt (delta_list[1]) via a local patch to"
    echo "pipeline.py that relaxes the W8-only gate."
    echo ""
}

# Quick test — W8A8, default prompts
quick_test_w8a8() {
    echo "Running quick test (100 images, W8A8, default prompts)..."
    python mixdq_lcm_generation.py \
        --num-samples 100 \
        --w-bit 8 \
        --a-bit 8 \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --output-dir "${BASE_OUTPUT}/quick_test_w8a8"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_w8a8/"
}

# Quick test — W8A8, COCO captions
quick_test_coco_w8a8() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    echo "Running quick test (100 images, W8A8, COCO captions)..."
    python mixdq_lcm_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --w-bit 8 \
        --a-bit 8 \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --output-dir "${BASE_OUTPUT}/quick_test_coco_w8a8"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco_w8a8/"
}

# Quick test — W4A8 mixed-precision sensitivity config (fake-quant)
quick_test_coco_w4a8() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    if [ ! -f "$W4A8_W_CONFIG" ] || [ ! -f "$W4A8_A_CONFIG" ]; then
        echo "Error: MixDQ sensitivity YAMLs not found under ${SCRIPT_DIR}/configs/"
        echo "Expected:"
        echo "  $W4A8_W_CONFIG"
        echo "  $W4A8_A_CONFIG"
        return
    fi
    echo "Running quick test (100 images, W4A8 mixed-precision fake-quant, COCO captions)..."
    echo "  Weight config: $W4A8_W_CONFIG (avg ~4 bits/weight via W2/W4/W8 mix)"
    echo "  Act    config: $W4A8_A_CONFIG (uniform A8)"
    python mixdq_lcm_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --w-bit 8 \
        --a-bit 8 \
        --w-config "$W4A8_W_CONFIG" \
        --a-config "$W4A8_A_CONFIG" \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --output-dir "${BASE_OUTPUT}/quick_test_coco_w4a8"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco_w4a8/"
}

# COCO evaluation — W8A8
coco_eval_w8a8() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== MixDQ W8A8 COCO Evaluation ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo ""
    echo "Configuration:"
    echo "  Model  : $BASE_MODEL"
    echo "  Quant  : W8A8"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : 1"
    echo "  CFG    : 0.0"
    echo "  Size   : 512×512"
    echo "  Batch  : auto"
    echo "  Output : ${BASE_OUTPUT}/coco_eval_w8a8"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python mixdq_lcm_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --w-bit 8 \
            --a-bit 8 \
            --num-inference-steps 1 \
            --guidance-scale 0.0 \
            --output-dir "${BASE_OUTPUT}/coco_eval_w8a8"
        echo "Done!"
    fi
}

# COCO evaluation — W4A8 mixed-precision sensitivity config (fake-quant)
coco_eval_w4a8() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    if [ ! -f "$W4A8_W_CONFIG" ] || [ ! -f "$W4A8_A_CONFIG" ]; then
        echo "Error: MixDQ sensitivity YAMLs not found under ${SCRIPT_DIR}/configs/"
        echo "Expected:"
        echo "  $W4A8_W_CONFIG"
        echo "  $W4A8_A_CONFIG"
        return
    fi

    echo "=== MixDQ W4A8 COCO Evaluation (mixed-precision, fake-quant) ==="
    echo ""
    echo "W4A8 runs as fake-quant using MixDQ's sensitivity-derived"
    echo "mixed-precision allocation (W2/W4/W8 per layer, avg ~4 bits)."
    echo "Weights are rounded per-channel to the shipped W{2,4,8} scales;"
    echo "A8 is fake-quantized at forward time. Throughput is lower than"
    echo "W8A8 because no accelerated W4 kernel exists — the path exercises"
    echo "F.linear/F.conv2d in fp16."
    echo ""
    echo "  Weight config: $W4A8_W_CONFIG (avg ~4 bits/weight via W2/W4/W8 mix)"
    echo "  Act    config: $W4A8_A_CONFIG (uniform A8)"
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo ""
    echo "Configuration:"
    echo "  Model  : $BASE_MODEL"
    echo "  Quant  : W4A8 mixed-precision (sensitivity-derived, fake-quant)"
    echo "  Images : $num_samples (starting at $start_idx)"
    echo "  Steps  : 1"
    echo "  CFG    : 0.0"
    echo "  Size   : 512×512"
    echo "  Batch  : auto"
    echo "  Output : ${BASE_OUTPUT}/coco_eval_w4a8"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python mixdq_lcm_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --w-bit 8 \
            --a-bit 8 \
            --w-config "$W4A8_W_CONFIG" \
            --a-config "$W4A8_A_CONFIG" \
            --num-inference-steps 1 \
            --guidance-scale 0.0 \
            --output-dir "${BASE_OUTPUT}/coco_eval_w4a8"
        echo "Done!"
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom MixDQ Generation ==="
    echo ""

    echo "Base model ID (default: stabilityai/sdxl-turbo):"
    read base_model
    base_model=${base_model:-stabilityai/sdxl-turbo}

    echo "Number of images (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Weight bit-width — 4 or 8 (default: 8):"
    echo "  4 → fake-quant path (uses shipped W4 scales, no accelerated kernel)"
    echo "  8 → accelerated qlinear/qconv2d kernels"
    read w_bit
    w_bit=${w_bit:-8}
    if [[ "$w_bit" != "4" && "$w_bit" != "8" ]]; then
        echo "Invalid w_bit=$w_bit (must be 4 or 8). Using 8."
        w_bit=8
    fi
    a_bit=8
    echo "Quant bit-widths    : W${w_bit}A${a_bit}"

    echo "Weight config YAML (optional, leave blank for uniform bit-width):"
    read w_config_path
    w_config_arg=""
    if [ -n "$w_config_path" ] && [ -f "$w_config_path" ]; then
        w_config_arg="--w-config $w_config_path"
    fi

    echo "Number of inference steps (default: 1):"
    read steps
    steps=${steps:-1}

    echo "Guidance scale (default: 0.0 — CFG-free for SDXL Turbo):"
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

    echo "Enable CUDA graph acceleration? (y/n, default: n):"
    read use_cuda_graph
    cuda_graph_arg=""
    if [[ $use_cuda_graph =~ ^[Yy]$ ]]; then
        cuda_graph_arg="--cuda-graph"
    fi

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

    cmd="python mixdq_lcm_generation.py"
    cmd="$cmd --base-model $base_model"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --w-bit $w_bit"
    cmd="$cmd --a-bit $a_bit"
    cmd="$cmd --num-inference-steps $steps"
    cmd="$cmd --guidance-scale $guidance"
    cmd="$cmd --height $height"
    cmd="$cmd --width $width"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --output-dir ${BASE_OUTPUT}/${subdir}"
    cmd="$cmd $cuda_graph_arg $coco_arg $prompt_arg $w_config_arg"

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
        1) quick_test_w8a8 ;;
        2) quick_test_coco_w8a8 ;;
        3) quick_test_coco_w4a8 ;;
        4) coco_eval_w8a8 ;;
        5) coco_eval_w4a8 ;;
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
