#!/usr/bin/env bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# Generate paper-figure images: 6 (model, config) × 4 prompts = 24 PNGs.
# Reuses W4A8 calibration caches from the ablation cache dir.
#
# Usage:
#     CUDA_VISIBLE_DEVICES=3 bash run.sh        # one GPU, ~30-40 min
#
# Outputs land in $OUT_ROOT/<config_dir>/prompt_<i>_<subject>.png with a
# prompts.txt copied at the top level for traceability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Reuse the ablation pipeline's conda activation + path constants.
source "$REPO_ROOT/src/ablations/common/paths.sh"

OUT_ROOT="${PAPER_OUT_ROOT:-$CHAMELEON_OUTPUT_ROOT/paper_figures}"
# Canonical DiT W4A8 config -- the input-aware (g,f) selection that produced the
# headline Table 1 row, not the ablation cache (which is a separate calibration).
DIT_IAW_WEIGHTS="${DIT_IAW_WEIGHTS:-$CHAMELEON_OUTPUT_ROOT/chameleon-dit/saved_configs/coco_eval_w4a8_10b_iaw.json}"
mkdir -p "$OUT_ROOT"
cp "$SCRIPT_DIR/prompts.txt" "$OUT_ROOT/prompts.txt"

PROMPT_FILE="$SCRIPT_DIR/prompts.txt"
PROMPT_LABELS=(1_car 2_cat 3_dog 4_car_night)
N=${#PROMPT_LABELS[@]}

# OOM resilience for shared GPUs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Find every PNG in $1 (recursive) sorted by mtime, rename to prompt_<i>_<label>.png
# and move into $2. Assumes generation order matches PROMPT_LABELS order
# (true for --prompt-file consumers; for single-prompt runs there's only 1).
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

# Single-prompt runner: one image per invocation. Used for SDXL chameleon
# because chameleon_generation.py has no --prompt-file (only --coco-captions,
# which shuffles).
run_sdxl_chameleon() {
    local out="$OUT_ROOT/sdxl_chameleon_w4a8"
    mkdir -p "$out"
    cd "$SDXL_DIR"
    for label in "${PROMPT_LABELS[@]}"; do
        local tmp; tmp="$(mktemp -d)"
        echo "[paper] sdxl_chameleon  prompt_${label}"
        python chameleon_generation.py \
            --weight-bits 4 \
            --num-samples 1 \
            --num-inference-steps 50 \
            --coco-captions "$SCRIPT_DIR/prompt_${label}.json" \
            --load-weights "$SDXL_W4A8_WEIGHTS" \
            --load-activation-lut "$SDXL_W4A8_LUT" \
            --output-dir "$tmp"
        local png; png="$(find "$tmp" -type f -name '*.png' | head -1)"
        if [[ -n "$png" ]]; then
            cp "$png" "$out/prompt_${label}.png"
        else
            echo "[paper] WARN: no png produced for sdxl_chameleon prompt_${label}" >&2
        fi
        rm -rf "$tmp"
    done
}

run_sdxl_baseline() {
    local out="$OUT_ROOT/sdxl_baseline_w4a8"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/q-baselines/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[paper] sdxl_baseline (qdiffusion W4A8)"
    python quantized_generation.py \
        --method qdiffusion \
        --weight-bits 4 --act-bits 8 \
        --num-samples "$N" \
        --num-inference-steps 50 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_lcm_chameleon() {
    local out="$OUT_ROOT/lcm_chameleon_w4a8"
    mkdir -p "$out"
    cd "$LCM_DIR"
    local tmp; tmp="$(mktemp -d)"
    echo "[paper] lcm_chameleon"
    python chameleon_lcm_generation.py \
        --w-bit 4 --a-bit 8 --chameleon-weights \
        --num-samples "$N" \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_lcm_baseline() {
    local out="$OUT_ROOT/lcm_baseline_w4a8"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/mixdq-lcm/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[paper] lcm_baseline (mixdq W4A8)"
    # MixDQ's W4A8 is per-layer MIXED precision averaging 4 bits (its
    # metric-decoupled IP assignment), NOT uniform 4-bit -- this matches the
    # configuration that produced the Table 1 row. Passing --w-bit 4 instead
    # forces uniform 4-bit on every layer, which MixDQ never intended and
    # which shatters the output.
    python mixdq_lcm_generation.py \
        --w-bit 8 --a-bit 8 \
        --w-config configs/weight_4.00.yaml \
        --a-config configs/act_8.00.yaml \
        --num-samples "$N" \
        --num-inference-steps 1 \
        --guidance-scale 0.0 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_dit_chameleon() {
    local out="$OUT_ROOT/dit_chameleon_w4a8"
    mkdir -p "$out"
    cd "$DIT_DIR"
    local tmp; tmp="$(mktemp -d)"
    echo "[paper] dit_chameleon"
    python chameleon_dit_generation.py \
        --weight-bits 4 \
        --num-samples "$N" \
        --num-inference-steps 20 \
        --prompt-file "$PROMPT_FILE" \
        --load-weights "$DIT_IAW_WEIGHTS" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

run_dit_baseline() {
    local out="$OUT_ROOT/dit_baseline_w4a8"
    mkdir -p "$out"
    cd "$REPO_ROOT/src/q-dit/generation"
    local tmp; tmp="$(mktemp -d)"
    echo "[paper] dit_baseline (q-dit W4A8)"
    python q_dit_generation.py \
        --weight-bits 4 --act-bits 8 \
        --num-samples "$N" \
        --num-inference-steps 20 \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$tmp"
    collect_outputs "$tmp" "$out"
    rm -rf "$tmp"
}

START=$(date +%s)
echo "[paper] start  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}  $(date +%T)"

FAILED=()
SUCCEEDED=()
run_cell() {
    local name="$1"; shift
    local t0=$(date +%s)
    echo "[paper] >>> $name"
    if "$@"; then
        SUCCEEDED+=("$name ($(( $(date +%s) - t0 ))s)")
        echo "[paper] <<< $name OK"
    else
        FAILED+=("$name ($(( $(date +%s) - t0 ))s)")
        echo "[paper] <<< $name FAILED rc=$?"
    fi
}

# Cheap → expensive ordering: DiT (20 steps) and LCM (4 steps) finish fastest,
# SDXL (50 steps) carries the bulk.
run_cell lcm_baseline      run_lcm_baseline
run_cell lcm_chameleon     run_lcm_chameleon
run_cell dit_baseline      run_dit_baseline
run_cell dit_chameleon     run_dit_chameleon
run_cell sdxl_baseline     run_sdxl_baseline
run_cell sdxl_chameleon    run_sdxl_chameleon

DUR=$(( $(date +%s) - START ))
echo "[paper] ============================================"
echo "[paper] done   total $((DUR/60))m $((DUR%60))s"
echo "[paper] succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}"
echo "[paper] failed    (${#FAILED[@]}): ${FAILED[*]}"
echo "[paper] outputs: $OUT_ROOT"
ls "$OUT_ROOT"
