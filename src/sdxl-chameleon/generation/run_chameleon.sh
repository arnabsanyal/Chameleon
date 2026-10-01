#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Chameleon: SNR-based Adaptive Quantization for SDXL
# Per-channel weight format selection + Timestep-aware activation quantization

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Chameleon: Adaptive Quantized Generation"
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

ask_weight_bits() {
    echo "Weight bit-width (4 or 8, default: 8):"
    read weight_bits
    weight_bits=${weight_bits:-8}
    if [ "$weight_bits" != "4" ] && [ "$weight_bits" != "8" ]; then
        echo "Invalid choice. Using default (8)."
        weight_bits=8
    fi
}

show_menu() {
    echo "Select operation:"
    echo ""
    echo "=== Quick Tests ==="
    echo "1) Quick test (100 images)"
    echo "2) Quick test with saved weights"
    echo ""
    echo "=== COCO Evaluation ==="
    echo "3) COCO evaluation (5000 images)"
    echo "4) COCO eval with custom bucket count"
    echo ""
    echo "=== Weight Management ==="
    echo "5) Analyze & save weights only (no generation)"
    echo "6) Load saved weights and generate"
    echo ""
    echo "=== Custom ==="
    echo "7) Custom generation (interactive)"
    echo ""
    echo "8) Exit"
    echo ""
}

# Quick test
quick_test() {
    WEIGHTS_FILE="$CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json"

    ask_weight_bits
    echo "Running quick test (100 images, W${weight_bits}A8)..."
    echo "Weights will be saved to: $WEIGHTS_FILE"
    python chameleon_generation.py \
        --num-samples 100 \
        --num-calibration-samples 16 \
        --num-inference-steps 30 \
        --batch-size 1 \
        --min-snr-threshold 20.0 \
        --weight-bits $weight_bits \
        --save-weights "$WEIGHTS_FILE" \
        --output-dir $CHAMELEON_OUTPUT_ROOT/chameleon/quick_test
    echo "Done! Check $CHAMELEON_OUTPUT_ROOT/chameleon/quick_test/w${weight_bits}a8/"
    echo "Weight config saved to: $WEIGHTS_FILE"
}

# Quick test with saved weights
quick_test_saved() {
    WEIGHTS_FILE="$CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json"

    ask_weight_bits
    if [ -f "$WEIGHTS_FILE" ]; then
        echo "Loading weights from $WEIGHTS_FILE..."
        python chameleon_generation.py \
            --num-samples 100 \
            --num-calibration-samples 16 \
            --num-inference-steps 30 \
            --batch-size 1 \
            --weight-bits $weight_bits \
            --load-weights "$WEIGHTS_FILE" \
            --output-dir $CHAMELEON_OUTPUT_ROOT/chameleon/quick_test_loaded
    else
        echo "Weights file not found. Running analysis and saving..."
        python chameleon_generation.py \
            --num-samples 100 \
            --num-calibration-samples 16 \
            --num-inference-steps 30 \
            --batch-size 1 \
            --weight-bits $weight_bits \
            --save-weights "$WEIGHTS_FILE" \
            --output-dir $CHAMELEON_OUTPUT_ROOT/chameleon/quick_test_saved
    fi
    echo "Done!"
}

# COCO evaluation
coco_eval() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"

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

    echo "Number of calibration samples (default: 128):"
    read calib_samples
    calib_samples=${calib_samples:-128}

    echo "Min SNR threshold in dB (default: 20.0):"
    read snr_threshold
    snr_threshold=${snr_threshold:-20.0}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    ask_weight_bits

    echo ""
    echo "Configuration:"
    echo "  Images: $num_samples (starting at $start_idx)"
    echo "  Calibration samples: $calib_samples"
    echo "  Min SNR: $snr_threshold dB"
    echo "  Batch size: $batch_size"
    echo "  Weight bits: $weight_bits (W${weight_bits}A8)"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-calibration-samples $calib_samples \
            --num-inference-steps 50 \
            --guidance-scale 7.5 \
            --batch-size $batch_size \
            --min-snr-threshold $snr_threshold \
            --weight-bits $weight_bits \
            --output-dir $CHAMELEON_OUTPUT_ROOT/chameleon/coco_eval
        echo "Done!"
    fi
}

# COCO eval with custom bucket count
coco_eval_custom_timesteps() {
    COCO_CAPTIONS="$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"

    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== COCO Evaluation with Custom Bucket Count ==="
    echo ""

    echo "Number of images to generate (default: 5000):"
    read num_samples
    num_samples=${num_samples:-5000}

    echo "Number of timestep buckets (default: 10, for T=1000 gives bucket_size=100):"
    read num_buckets
    num_buckets=${num_buckets:-10}

    echo "Number of calibration samples (default: 128):"
    read calib_samples
    calib_samples=${calib_samples:-128}

    ask_weight_bits

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --num-calibration-samples $calib_samples \
            --num-buckets $num_buckets \
            --num-inference-steps 50 \
            --guidance-scale 7.5 \
            --batch-size 1 \
            --weight-bits $weight_bits \
            --output-dir $CHAMELEON_OUTPUT_ROOT/chameleon/coco_eval_custom
        echo "Done!"
    fi
}

# Analyze weights only
analyze_weights() {
    echo "=== Analyzing SDXL UNet Weights ==="
    echo ""

    echo "Save weights config to (default: $CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json):"
    read save_path
    save_path=${save_path:-$CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json}

    echo "Min SNR threshold in dB (default: 20.0):"
    read snr_threshold
    snr_threshold=${snr_threshold:-20.0}

    ask_weight_bits

    echo ""
    echo "This will:"
    echo "  1. Load the SDXL model"
    echo "  2. Analyze weight distributions per channel"
    echo "  3. Select optimal ${weight_bits}-bit format per channel using SNR"
    echo "  4. Save configuration to $save_path"
    echo ""

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -c "
import torch
from chameleon_quant import create_chameleon_quantizer

print('Creating Chameleon quantizer (W${weight_bits}A8)...')
quantizer = create_chameleon_quantizer(
    min_snr_threshold=${snr_threshold},
    weight_bits=${weight_bits},
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

quantizer.load_model()

print('\\nAnalyzing weights with SNR-based format selection...')
quantizer.inject_weight_quantization(verbose=False)

print('\\nSaving weight quantization config...')
quantizer.save_weight_quantization('${save_path}')

print('\\nWeight Statistics:')
quantizer.print_weight_stats()

quantizer.cleanup()
print('\\nDone!')
"
    fi
}

# Load weights and generate
load_weights_generate() {
    echo "=== Load Weights and Generate ==="
    echo ""

    echo "Weights config file (default: $CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json):"
    read weights_file
    weights_file=${weights_file:-$CHAMELEON_OUTPUT_ROOT/chameleon/weights_config.json}

    if [ ! -f "$weights_file" ]; then
        echo "Error: Weights file not found at $weights_file"
        return
    fi

    echo "Number of images to generate (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Output directory (default: $CHAMELEON_OUTPUT_ROOT/chameleon/loaded_weights):"
    read output_dir
    output_dir=${output_dir:-$CHAMELEON_OUTPUT_ROOT/chameleon/loaded_weights}

    ask_weight_bits

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_generation.py \
            --num-samples $num_samples \
            --num-calibration-samples 64 \
            --num-inference-steps 50 \
            --batch-size 1 \
            --weight-bits $weight_bits \
            --load-weights "$weights_file" \
            --output-dir "$output_dir"
        echo "Done!"
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom Chameleon Generation ==="
    echo ""

    echo "Number of images to generate (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Number of calibration samples (default: 128):"
    read calib_samples
    calib_samples=${calib_samples:-128}

    echo "Min SNR threshold in dB (default: 20.0):"
    read snr_threshold
    snr_threshold=${snr_threshold:-20.0}

    echo "MX block size (default: 32):"
    read mx_block
    mx_block=${mx_block:-32}

    ask_weight_bits

    echo "Number of inference steps (default: 50):"
    read inf_steps
    inf_steps=${inf_steps:-50}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    echo "Output directory (default: $CHAMELEON_OUTPUT_ROOT/chameleon/custom):"
    read output_dir
    output_dir=${output_dir:-$CHAMELEON_OUTPUT_ROOT/chameleon/custom}

    echo "Use COCO captions? (y/n, default: n):"
    read use_coco
    coco_arg=""
    if [[ $use_coco =~ ^[Yy]$ ]]; then
        coco_arg="--coco-captions $CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json"
    fi

    echo "Number of timestep buckets (default: 10, press enter to use default):"
    read num_buckets
    buckets_arg=""
    if [ -n "$num_buckets" ]; then
        buckets_arg="--num-buckets $num_buckets"
    fi

    echo "Save weights config? (path or press enter to skip):"
    read save_weights
    save_arg=""
    if [ -n "$save_weights" ]; then
        save_arg="--save-weights $save_weights"
    fi

    echo "Verbose output? (y/n, default: n):"
    read verbose
    verbose_arg=""
    if [[ $verbose =~ ^[Yy]$ ]]; then
        verbose_arg="--verbose"
    fi

    cmd="python chameleon_generation.py"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --num-calibration-samples $calib_samples"
    cmd="$cmd --min-snr-threshold $snr_threshold"
    cmd="$cmd --mx-block-size $mx_block"
    cmd="$cmd --weight-bits $weight_bits"
    cmd="$cmd --num-inference-steps $inf_steps"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --output-dir $output_dir"
    cmd="$cmd $coco_arg $buckets_arg $save_arg $verbose_arg"

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
        2) quick_test_saved ;;
        3) coco_eval ;;
        4) coco_eval_custom_timesteps ;;
        5) analyze_weights ;;
        6) load_weights_generate ;;
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
