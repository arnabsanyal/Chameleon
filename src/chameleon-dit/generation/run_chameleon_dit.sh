#!/bin/bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"


# Chameleon-DiT: Adaptive Format + Granularity Quantization for PixArt-alpha
# Fuses Chameleon's macro-routing with Q-DiT's group quantization:
#   Weights: per-layer (g, f) search over {32,64,128,192,288} × bit-width-dependent formats
#            4-bit: {INT4_ASYM, NF4, FP4_E2M1}
#            6-bit: {INT6_SYM, INT6_ASYM}
#            8-bit: {INT8_SYM, INT8_ASYM, MXINT8}
#   Activations: dynamic micro-scaling, macro-routed format. Modes:
#            pertensor   (DEFAULT, canonical): one global scale/zp per tensor
#            group-macro (ablation): group-wise quant in the macro format
#            group-int8  (ablation): group-wise asym INT8 (routing off)
#            [group-* underperform per-tensor on PixArt-alpha — ablation only]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Chameleon-DiT: Adaptive DiT Quantization"
echo "  Weights: (g, f) evolutionary search"
echo "  Act.   : per-tensor micro-scale (default)"
echo "           +group-macro / group-int8 ablations"
echo "=========================================="
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
BASE_OUTPUT="$CHAMELEON_OUTPUT_ROOT/chameleon-dit"
DEFAULT_MODEL="PixArt-alpha/PixArt-XL-2-1024-MS"
WEIGHTS_DIR="${BASE_OUTPUT}/saved_configs"

ask_weight_bits() {
    echo "Weight bit-width (4, 6, or 8; default: 4):"
    echo "  4 → W4A8  {INT4_ASYM, NF4, FP4_E2M1}"
    echo "  6 → W6A8  {INT6_SYM, INT6_ASYM}"
    echo "  8 → W8A8  {INT8_SYM, INT8_ASYM, MXINT8}"
    read weight_bits
    weight_bits=${weight_bits:-4}
    if [ "$weight_bits" != "4" ] && [ "$weight_bits" != "6" ] && [ "$weight_bits" != "8" ]; then
        echo "Invalid choice. Using default (4)."
        weight_bits=4
    fi
}

ask_num_buckets() {
    echo "Number of timestep buckets for macro activation routing (default: 10):"
    echo "  5  → coarser routing (faster calibration)"
    echo "  10 → paper default"
    echo "  20 → finer routing"
    read num_buckets
    num_buckets=${num_buckets:-10}
}

ask_calibration_samples() {
    echo "Calibration samples (default: 64, use 128+ for production):"
    read cal_samples
    cal_samples=${cal_samples:-64}
}

# Weight-side modifiers (#1 mixed precision, #2 clip search, input-aware selection).
# Sets globals: wmod_args (CLI flags) and wmod_suffix (output-dir suffix).
# Requires $weight_bits to be set first (ask_weight_bits).
ask_weight_modifiers() {
    wmod_args=""; wmod_suffix=""

    echo "Input-aware weight selection? (y/n, default: n):"
    echo "  y → pick (g,f) by OUTPUT error (GPTQ/Q-DiT criterion); targets the W4A8 gap"
    read iaw
    if [[ $iaw =~ ^[Yy]$ ]]; then wmod_args="$wmod_args --input-aware-weights"; wmod_suffix="${wmod_suffix}_iaw"; fi

    echo "Weight clip-ratio search (#2)? (y/n, default: n):"
    read clipsearch
    if [[ $clipsearch =~ ^[Yy]$ ]]; then wmod_args="$wmod_args --clip-search"; wmod_suffix="${wmod_suffix}_clip"; fi

    if [ "$weight_bits" = "4" ]; then
        echo "Mixed-precision allocation (#1, W4 only)? (y/n, default: n):"
        echo "  y → promote sensitive layers to 8-bit under an avg-bit budget"
        read mp
        if [[ $mp =~ ^[Yy]$ ]]; then
            echo "  Average-bit target (default: 4.5):"
            read mptgt; mptgt=${mptgt:-4.5}
            wmod_args="$wmod_args --mixed-precision --mp-target-bits $mptgt"
            wmod_suffix="${wmod_suffix}_mp${mptgt}"
        fi
    fi
}

ask_act_mode() {
    echo "Activation quant mode (default: pertensor — the canonical model):"
    echo "  pertensor   → one global scale/zp, macro-routed format (BEST on DiT)"
    echo "  group-macro → ABLATION: group-wise quant in the macro format"
    echo "  group-int8  → ABLATION: group-wise asym INT8 everywhere (routing off)"
    echo "  (group-* underperform per-tensor on PixArt-alpha — for the ablation table)"
    read act_mode
    act_mode=${act_mode:-pertensor}
    if [ "$act_mode" != "pertensor" ] && [ "$act_mode" != "group-macro" ] && [ "$act_mode" != "group-int8" ]; then
        echo "Invalid choice. Using default (pertensor)."
        act_mode=pertensor
    fi
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
    echo "=== Save / Load ==="
    echo "5) Calibrate and save config (then generate)"
    echo "6) Load saved config and generate"
    echo ""
    echo "=== Ablation ==="
    echo "7) Bucket count sweep (5 vs 10 vs 20 buckets)"
    echo ""
    echo "=== Custom ==="
    echo "8) Custom generation (interactive)"
    echo ""
    echo "9) Exit"
    echo ""
}

# Quick test — default prompts
quick_test() {
    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples
    ask_act_mode
    ask_weight_modifiers
    echo "Running quick test (100 images, W${weight_bits}A8, ${num_buckets} buckets, act=${act_mode})..."
    python chameleon_dit_generation.py \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --weight-bits $weight_bits \
        --num-buckets $num_buckets \
        --num-calibration-samples $cal_samples \
        --act-quant-mode $act_mode $wmod_args \
        --output-dir "${BASE_OUTPUT}/quick_test"
    echo "Done! Check ${BASE_OUTPUT}/quick_test/w${weight_bits}a8_chameleon_dit_${num_buckets}b/"
}

# Quick test — COCO captions
quick_test_coco() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi
    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples
    ask_act_mode
    ask_weight_modifiers
    echo "Running quick test (100 images, COCO, W${weight_bits}A8, ${num_buckets} buckets, act=${act_mode})..."
    python chameleon_dit_generation.py \
        --coco-captions "$COCO_CAPTIONS" \
        --num-samples 100 \
        --num-inference-steps 20 \
        --guidance-scale 4.5 \
        --batch-size 1 \
        --weight-bits $weight_bits \
        --num-buckets $num_buckets \
        --num-calibration-samples $cal_samples \
        --act-quant-mode $act_mode $wmod_args \
        --output-dir "${BASE_OUTPUT}/quick_test_coco"
    echo "Done! Check ${BASE_OUTPUT}/quick_test_coco/w${weight_bits}a8_chameleon_dit_${num_buckets}b/"
}

# Full COCO evaluation
coco_eval() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== Chameleon-DiT COCO Evaluation ==="
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

    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples
    ask_act_mode
    ask_weight_modifiers

    # pertensor keeps the canonical slot name; modifiers append suffixes (driver mirrors this).
    out_suffix=""; [ "$act_mode" != "pertensor" ] && out_suffix="_act${act_mode}"
    out_suffix="${out_suffix}${wmod_suffix}"
    weights_path="${WEIGHTS_DIR}/coco_eval_w${weight_bits}a8_${num_buckets}b${out_suffix}.json"
    echo ""
    echo "Configuration:"
    echo "  Model   : $DEFAULT_MODEL"
    echo "  Weights : W${weight_bits}A8 $([ -n "$iaw_arg" ] && echo '(input-aware selection)')"
    echo "  Actmode : $act_mode"
    echo "  Images  : $num_samples (start=$start_idx)"
    echo "  Steps   : 20, CFG: 4.5"
    echo "  Buckets : $num_buckets"
    echo "  Cal.    : $cal_samples samples"
    echo "  Config  : ${weights_path} (saved for reuse)"
    echo "  Output  : ${BASE_OUTPUT}/coco_eval/w${weight_bits}a8_chameleon_dit_${num_buckets}b${out_suffix}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$WEIGHTS_DIR"
        python chameleon_dit_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps 20 \
            --guidance-scale 4.5 \
            --batch-size $batch_size \
            --weight-bits $weight_bits \
            --num-buckets $num_buckets \
            --num-calibration-samples $cal_samples \
            --act-quant-mode $act_mode $wmod_args \
            --save-weights "$weights_path" \
            --output-dir "${BASE_OUTPUT}/coco_eval"
        echo "Done! Config saved to ${weights_path}"
    fi
}

# COCO eval with custom settings
coco_eval_custom() {
    if [ ! -f "$COCO_CAPTIONS" ]; then
        echo "Error: COCO captions not found at $COCO_CAPTIONS"
        return
    fi

    echo "=== Chameleon-DiT COCO Eval — Custom Settings ==="
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

    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples
    ask_act_mode

    echo "Output subdirectory (default: coco_eval_custom):"
    read subdir
    subdir=${subdir:-coco_eval_custom}

    echo ""
    echo "Configuration:"
    echo "  Weights : W${weight_bits}A8"
    echo "  Actmode : $act_mode (group-wise)"
    echo "  Images  : $num_samples (start=$start_idx)"
    echo "  Steps   : $steps, CFG: $guidance"
    echo "  Buckets : $num_buckets"
    echo "  Output  : ${BASE_OUTPUT}/${subdir}/w${weight_bits}a8_chameleon_dit_${num_buckets}b"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_dit_generation.py \
            --coco-captions "$COCO_CAPTIONS" \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps $steps \
            --guidance-scale $guidance \
            --batch-size $batch_size \
            --weight-bits $weight_bits \
            --num-buckets $num_buckets \
            --num-calibration-samples $cal_samples \
            --act-quant-mode $act_mode \
            --output-dir "${BASE_OUTPUT}/${subdir}"
        echo "Done!"
    fi
}

# Calibrate and save config
calibrate_and_save() {
    echo "=== Calibrate Chameleon-DiT and Save Config ==="
    echo ""

    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples

    echo "Config name (default: chameleon_dit_w${weight_bits}a8_${num_buckets}b):"
    read config_name
    config_name=${config_name:-chameleon_dit_w${weight_bits}a8_${num_buckets}b}

    weights_path="${WEIGHTS_DIR}/${config_name}.json"

    echo "Number of images to generate after calibration (default: 100):"
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

    echo ""
    echo "Will calibrate (W${weight_bits}A8, ${num_buckets} buckets) and save to: ${weights_path}"

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$WEIGHTS_DIR"
        python chameleon_dit_generation.py \
            $coco_arg \
            --num-samples $num_samples \
            --num-inference-steps 20 \
            --guidance-scale 4.5 \
            --batch-size 1 \
            --weight-bits $weight_bits \
            --num-buckets $num_buckets \
            --num-calibration-samples $cal_samples \
            --save-weights "$weights_path" \
            --output-dir "${BASE_OUTPUT}/calibration_test"
        echo "Done! Config saved to ${weights_path}"
        echo "Use option 6 to load and generate with this config."
    fi
}

# Load saved config and generate
load_and_generate() {
    echo "=== Load Saved Config and Generate ==="
    echo ""

    # List available configs
    if [ -d "$WEIGHTS_DIR" ]; then
        echo "Available saved configs in ${WEIGHTS_DIR}:"
        ls "${WEIGHTS_DIR}"/*.json 2>/dev/null || echo "  (none found)"
        echo ""
    fi

    echo "Path to saved config JSON:"
    read weights_path
    if [ ! -f "$weights_path" ]; then
        echo "Error: config not found at $weights_path"
        return
    fi

    echo "Number of images (default: 100):"
    read num_samples
    num_samples=${num_samples:-100}

    echo "Starting index (default: 0):"
    read start_idx
    start_idx=${start_idx:-0}

    echo "Batch size (default: 1):"
    read batch_size
    batch_size=${batch_size:-1}

    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        echo "Use COCO captions? (y/n, default: n):"
        read use_coco
        if [[ $use_coco =~ ^[Yy]$ ]]; then
            coco_arg="--coco-captions $COCO_CAPTIONS"
        fi
    fi

    echo "Output subdirectory (default: from_saved):"
    read subdir
    subdir=${subdir:-from_saved}

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python chameleon_dit_generation.py \
            $coco_arg \
            --num-samples $num_samples \
            --start-idx $start_idx \
            --num-inference-steps 20 \
            --guidance-scale 4.5 \
            --batch-size $batch_size \
            --load-weights "$weights_path" \
            --output-dir "${BASE_OUTPUT}/${subdir}"
        echo "Done! Check ${BASE_OUTPUT}/${subdir}/"
    fi
}

# Bucket count sweep (ablation)
bucket_sweep() {
    echo "=== Bucket Count Sweep (5 vs 10 vs 20 buckets; 50 images each) ==="
    echo ""

    ask_weight_bits

    n=50
    cal=64
    coco_arg=""
    if [ -f "$COCO_CAPTIONS" ]; then
        coco_arg="--coco-captions $COCO_CAPTIONS"
    fi

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for nb in 5 10 20; do
            echo "--- W${weight_bits}A8, Num buckets = $nb ---"
            python chameleon_dit_generation.py \
                $coco_arg --num-samples $n \
                --num-inference-steps 20 --guidance-scale 4.5 \
                --weight-bits $weight_bits \
                --num-buckets $nb \
                --num-calibration-samples $cal \
                --output-dir "${BASE_OUTPUT}/ablation_buckets"
        done
        echo "Done! Results in ${BASE_OUTPUT}/ablation_buckets/"
    fi
}

# Custom generation
custom_generation() {
    echo "=== Custom Chameleon-DiT Generation ==="
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

    ask_weight_bits
    ask_num_buckets
    ask_calibration_samples

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

    echo "Load saved config? (path or enter to skip):"
    read load_path
    load_arg=""
    if [ -n "$load_path" ] && [ -f "$load_path" ]; then
        load_arg="--load-weights $load_path"
    fi

    echo "Save config after calibration? (path or enter to skip):"
    read save_path
    save_arg=""
    if [ -n "$save_path" ]; then
        mkdir -p "$(dirname "$save_path")"
        save_arg="--save-weights $save_path"
    fi

    echo "Verbose output? (y/n, default: n):"
    read verbose
    verbose_arg=""
    if [[ $verbose =~ ^[Yy]$ ]]; then
        verbose_arg="--verbose"
    fi

    cmd="python chameleon_dit_generation.py"
    cmd="$cmd --model-id $model_id"
    cmd="$cmd --num-samples $num_samples"
    cmd="$cmd --start-idx $start_idx"
    cmd="$cmd --num-inference-steps $steps"
    cmd="$cmd --guidance-scale $guidance"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --weight-bits $weight_bits"
    cmd="$cmd --num-buckets $num_buckets"
    cmd="$cmd --num-calibration-samples $cal_samples"
    cmd="$cmd --output-dir ${BASE_OUTPUT}/${subdir}"
    cmd="$cmd $coco_arg $prompt_arg $load_arg $save_arg $verbose_arg"

    echo ""
    echo "Running: $cmd"
    eval $cmd
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-9]: " choice
    echo ""

    case $choice in
        1) quick_test ;;
        2) quick_test_coco ;;
        3) coco_eval ;;
        4) coco_eval_custom ;;
        5) calibrate_and_save ;;
        6) load_and_generate ;;
        7) bucket_sweep ;;
        8) custom_generation ;;
        9)
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
