#!/usr/bin/env bash
# Master driver — keeps 4 GPUs continuously busy across ALL ablations.
#
#   bash run_all.sh                # interactive menu
#   bash run_all.sh calibrate      # phase 1: cache calibration in parallel
#   bash run_all.sh 1              # ablation 1 only (own dispatcher)
#   bash run_all.sh 1 2            # ablations 1 + 2 (shared dispatcher)
#   bash run_all.sh all            # everything: calibrate, then all 20 rows
#                                  # in ONE shared queue (no per-ablation idle)
#
# GPU pool selection (pick whichever fits your machine):
#   GPU_IDS="4,5,6,7" bash run_all.sh all     # explicit list, any ids
#   NUM_GPUS=8        bash run_all.sh all     # uses ids 0..N-1
#   bash run_all.sh all                        # defaults to 0..3
#
# Verifier checkpoints: every generation row spawns a background watchdog
# that reads its output dir at 20/40/60/80/100% of NUM_SAMPLES and writes
# <row>/progress.json with pixel-stats + a DIRECTION verdict
# (OK / REGRESSION / DEGENERATE). Set VERIFIER_DISABLE=1 to opt out.
#
# Output root: $OUT_ROOT (see common/paths.sh).

set -euo pipefail
# Use a name that ablation_*/run.sh won't clobber. Each child script resets
# its own SCRIPT_DIR when sourced, which previously turned the second
# submit_ablation call into a nested path like
# ablation_1_routing/ablation_2_palette/run.sh and aborted the master.
ABLATIONS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$ABLATIONS_ROOT"
source "$ABLATIONS_ROOT/common/paths.sh"
source "$ABLATIONS_ROOT/common/dispatch.sh"

NUM_GPUS="${NUM_GPUS:-4}"
export NUM_GPUS

run_calibrate() { bash "$COMMON_DIR/cache_calibration.sh"; }

# Submit one ablation's jobs into the SHARED dispatcher (already initialised).
submit_ablation() {
    case "$1" in
        1) source "$ABLATIONS_ROOT/ablation_1_routing/run.sh" ;;
        2) source "$ABLATIONS_ROOT/ablation_2_palette/run.sh" ;;
        3) source "$ABLATIONS_ROOT/ablation_3_arch_fork/run.sh" ;;
        4) source "$ABLATIONS_ROOT/ablation_4_buckets/run.sh" ;;
        *) echo "unknown ablation: $1" >&2; return 2 ;;
    esac
}

run_set() {
    local first=1
    for sel in "$@"; do
        if [[ "$sel" == "calibrate" || "$sel" == "cal" || "$sel" == "c" ]]; then
            run_calibrate
            continue
        fi
        if [[ "$first" == "1" ]]; then
            dispatch_init "$NUM_GPUS"
            first=0
        fi
        submit_ablation "$sel"
    done
    if [[ "$first" == "0" ]]; then
        dispatch_wait
    fi
}

run_all() {
    run_calibrate

    # Submit ALL 20 generation rows into one shared queue. LPT ordering would
    # finish slightly faster than ablation order, but the queue empties in
    # ~9 wall-clock hours either way on 4× A100, so we keep ablation order
    # for log readability.
    dispatch_init "$NUM_GPUS"
    submit_ablation 1     # SDXL × 5 rows (headline)
    submit_ablation 2     # SDXL × 6 rows (palette + shortcut)
    submit_ablation 4     # SDXL × 4 rows (bucket sweep)
    submit_ablation 3     # LCM × 2 + DiT × 3 (cheap, fills tail)
    dispatch_wait
}

if [[ $# -gt 0 ]]; then
    if [[ "$1" == "all" ]]; then
        run_all
    else
        run_set "$@"
    fi
    exit 0
fi

if [[ -n "${GPU_IDS:-}" ]]; then
    pool_desc="GPU_IDS=${GPU_IDS}"
else
    pool_desc="GPUs 0..$((NUM_GPUS-1))"
fi
cat <<EOF
Chameleon Ablation Driver  (dispatcher pool: $pool_desc)
   Output root: $OUT_ROOT

  c) Cache calibration (parallel, ~30 min on 4× A100)
  1) Ablation 1 — Routing (κ vs SNR vs both)        SDXL  ~6 h sequential / ~2 h on 4 GPUs
  2) Ablation 2 — Format palette                    SDXL  ~7 h sequential / ~2 h on 4 GPUs
  3) Ablation 3 — Architectural fork (BAQ, m+m)     LCM+DiT  ~2.5 h sequential / ~40 min on 4 GPUs
  4) Ablation 4 — Bucket count sweep                SDXL  ~5 h sequential / ~2 h on 4 GPUs
  a) Run all (1+2+4+3 in ONE shared queue, ~9 h on 4 GPUs)
  q) Quit
EOF
read -rp "Select: " choice
case "$choice" in
    c) run_calibrate ;;
    1|2|3|4) run_set "$choice" ;;
    a) run_all ;;
    q|*) exit 0 ;;
esac
