#!/usr/bin/env bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# ============================================================================
# Chameleon paper ablations @ 5,000 images (the ablation budget; the headline
# Table 1 stays at 24,576).  Two parts:
#
#   PART A — GENERATE + score every ablation arm that is NOT already on disk,
#            at 5k, via the shared GPU dispatcher (run_all.sh machinery).
#   PART B — SUB-SCORE the first 5k of the existing 24k baseline/quantized/
#            negative-result folders (paired comparison; no regeneration).
#
# Why 5k and why sub-scoring works: FID is sample-size-biased, so you can never
# compare a 5k arm against the 24k headline. But within a fixed N the relative
# ordering is what ablations need, and the first 5k PNGs of a 24k run are
# byte-identical to a fresh 5k run (deterministic per-index seeds). So we run
# every NEW arm at 5k and sub-score every EXISTING baseline at its first 5k —
# all paired, all comparable, none regenerated twice.
#
# Usage:
#   NUM_GPUS=4 bash run_ablations_5k.sh            # generate + sub-score (default)
#   NUM_GPUS=4 bash run_ablations_5k.sh generate   # PART A only
#   bash run_ablations_5k.sh score                 # PART B only (1 GPU is fine)
#   GPU_IDS="4,5,6,7" bash run_ablations_5k.sh      # explicit GPU pool
#
# What PART A generates (none overlap the data we already have at 24k):
#   SDXL  (ablation_1) routing: single-E4M3, kurt-only, snr-only, both, perturbed-thresholds
#   SDXL  (ablation_2) palette : single-INT8, single-FP8E4M3, drop-MXFP8(shortcut),
#                                drop-NF4, drop-MXFP4, full
#   SDXL  (ablation_4) buckets : B = 1, 5, 10, 20  (B=1 == no temporal routing)
#   LCM   (ablation_3) BAQ on/off  (few-step, SDXL-Turbo 1-step)
#   DiT   (ablation_3) macro-only / micro-only / macro+micro
#
# Already on disk at 24k — PART B sub-scores, does NOT regenerate:
#   FP16 + Chameleon (SDXL/LCM/DiT), Q-Diffusion, PTQ4DM, MixDQ, Q-DiT,
#   and the DiT negative results (per-tensor vs group-wise, mixed-precision,
#   input-aware vs weight-only).
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 5k budget + SDXL-Turbo 1-step for LCM (env-overridable, consumed by paths.sh).
export NUM_SAMPLES="${NUM_SAMPLES:-5000}"
export LCM_STEPS="${LCM_STEPS:-1}"
export NUM_GPUS="${NUM_GPUS:-4}"

MODE="${1:-all}"
LIMIT=5000
PY="${PY:-python}"
SCORER="$ROOT/../paper_figures/score_24k.py"

# ── PART A: generate + score all new ablation arms at 5k ────────────────────
generate() {
    echo "=== PART A: generating ablation arms at ${NUM_SAMPLES} images ==="
    # run_all.sh 'all' caches calibration then dispatches ablations 1,2,4,3.
    # NUM_SAMPLES/LCM_STEPS are honoured because paths.sh now reads them from env.
    bash "$ROOT/run_all.sh" all
}

# ── PART B: sub-score the first 5k of every existing full-set folder ────────
# Auto-discovers folders from their existing coco_score.json (image_dir+label),
# re-scores the first $LIMIT with the same ref, and writes coco_score_5k.json
# (never clobbers the 24k coco_score.json).
score_baselines() {
    echo "=== PART B: sub-scoring first ${LIMIT} of existing full-set folders ==="
    local GEN_ROOT="$CHAMELEON_OUTPUT_ROOT"
    find "$GEN_ROOT" -name "coco_score.json" 2>/dev/null | while read -r sj; do
        local img_dir label
        img_dir="$($PY -c "import json,sys;print(json.load(open('$sj'))['image_dir'])")"
        label="$($PY -c "import json,sys;print(json.load(open('$sj')).get('label') or '')")"
        [ -d "$img_dir" ] || { echo "  skip (missing dir): $img_dir"; continue; }
        # Skip our own ablation outputs (PART A already scored those at 5k).
        case "$img_dir" in *"/ablations/"*) continue ;; esac
        echo "  sub-scoring ${label:-$(basename "$img_dir")} ..."
        $PY "$SCORER" "$img_dir" --label "${label}_5k" --limit "$LIMIT" || \
            echo "  WARN: sub-score failed for $img_dir"
    done
    echo "Sub-scores written as coco_score_${LIMIT}.json next to each coco_score.json."
}

case "$MODE" in
    all)      generate; score_baselines ;;
    generate) generate ;;
    score)    score_baselines ;;
    *) echo "usage: $0 [all|generate|score]" >&2; exit 2 ;;
esac

echo "=== done ($MODE). Ablation arms under \$OUT_ROOT; baseline 5k scores as coco_score_${LIMIT}.json ==="
