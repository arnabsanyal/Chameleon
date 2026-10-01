#!/usr/bin/env bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# FP16 unquantized companion to run.sh: SDXL, LCM, DiT × 4 prompts = 12 PNGs.
# Outputs under $OUT_ROOT/unquantized/{sdxl,lcm,dit}/prompt_*.png.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/src/ablations/common/paths.sh"

OUT_ROOT="${PAPER_OUT_ROOT:-$CHAMELEON_OUTPUT_ROOT/paper_figures}"
UNQUANT_DIR="$OUT_ROOT/unquantized"
mkdir -p "$UNQUANT_DIR"

PROMPT_FILE="$SCRIPT_DIR/prompts.txt"
PROMPT_LABELS=(1_car 2_cat 3_dog 4_car_night)
N=${#PROMPT_LABELS[@]}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

collect_outputs() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    mapfile -t pngs < <(find "$src" -type f -name '*.png' -printf '%T@ %p\n' \
                        | sort -n | awk '{print $2}')
    local i=0
    for p in "${pngs[@]}"; do
        if (( i >= N )); then break; fi
        cp "$p" "$dst/prompt_${PROMPT_LABELS[$i]}.png"
        i=$((i+1))
    done
}

run_sdxl_fp16() {
    local out="$UNQUANT_DIR/sdxl"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/baseline/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[fp16] sdxl"
    python baseline_generation.py \
        --mode image \
        --num-samples "$N" \
        --num-inference-steps 50 \
        --batch-size 1 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_lcm_fp16() {
    local out="$UNQUANT_DIR/lcm"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/lcm/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[fp16] lcm"
    python lcm_generation.py \
        --num-samples "$N" \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_dit_fp16() {
    local out="$UNQUANT_DIR/dit"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/pixart-alpha/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[fp16] dit (pixart-alpha)"
    python pixart_generation.py \
        --num-samples "$N" \
        --num-inference-steps 20 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

START=$(date +%s)
echo "[fp16] start  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}  $(date +%T)"

FAILED=()
SUCCEEDED=()
run_cell() {
    local name="$1"; shift
    local t0=$(date +%s)
    echo "[fp16] >>> $name"
    if "$@"; then
        SUCCEEDED+=("$name ($(( $(date +%s) - t0 ))s)")
        echo "[fp16] <<< $name OK"
    else
        FAILED+=("$name ($(( $(date +%s) - t0 ))s)")
        echo "[fp16] <<< $name FAILED rc=$?"
    fi
}

# Cheap → expensive
run_cell lcm_fp16   run_lcm_fp16
run_cell dit_fp16   run_dit_fp16
run_cell sdxl_fp16  run_sdxl_fp16

DUR=$(( $(date +%s) - START ))
echo "[fp16] ============================================"
echo "[fp16] done   total $((DUR/60))m $((DUR%60))s"
echo "[fp16] succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}"
echo "[fp16] failed    (${#FAILED[@]}): ${FAILED[*]}"
echo "[fp16] outputs: $UNQUANT_DIR"
ls "$UNQUANT_DIR"
