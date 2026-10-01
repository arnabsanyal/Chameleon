#!/usr/bin/env bash
# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# Score 11 ready 24k-image dirs for Tables 1/2/3.
# Bounded concurrency: at most ${#POOL[@]} jobs in flight, one per GPU.
#
# Usage:  GPU_IDS="3,6" bash score_all.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ablations/common/paths.sh"

GPUS_RAW="${GPU_IDS:-3,6}"
IFS=',' read -ra POOL <<< "$GPUS_RAW"
N_SLOTS=${#POOL[@]}

LOG_DIR="$OUT_ROOT/_score_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[score_all] log dir: $LOG_DIR"
echo "[score_all] GPUs: ${POOL[*]} ($N_SLOTS slots)"

JOBS=(
    "sdxl_fp16             $CHAMELEON_OUTPUT_ROOT/coco_baseline/images"
    "sdxl_chameleon_w8a8   $CHAMELEON_OUTPUT_ROOT/chameleon/coco_eval/w8a8"
    "sdxl_chameleon_w4a8   $CHAMELEON_OUTPUT_ROOT/chameleon/coco_eval/w4a8"
    "lcm_fp16              $CHAMELEON_OUTPUT_ROOT/lcm/coco_eval"
    "lcm_mixdq_w8a8        $CHAMELEON_OUTPUT_ROOT/mixdq-lcm/coco_eval_w8a8"
    "lcm_mixdq_w4a8        $CHAMELEON_OUTPUT_ROOT/mixdq-lcm/coco_eval_w4a8"
    "lcm_chameleon_w8a8    $CHAMELEON_OUTPUT_ROOT/chameleon-lcm/coco_eval_w8a8"
    "lcm_chameleon_w4a8    $CHAMELEON_OUTPUT_ROOT/chameleon-lcm/coco_eval_w4a8"
    "dit_pixart_fp16       $CHAMELEON_OUTPUT_ROOT/pixart-alpha-dit/coco_eval"
    "dit_qdit_w8a8         $CHAMELEON_OUTPUT_ROOT/q-dit/coco_eval/w8a8_g128"
    "dit_chameleon         $CHAMELEON_OUTPUT_ROOT/chameleon-dit/coco_eval/chameleon_dit_10b"
)

# FIFO semaphore so each job pops one GPU id and pushes it back when done.
fifo="$(mktemp -u --tmpdir score_all.XXXXXX)"
mkfifo "$fifo"
exec 7<>"$fifo"
rm -f "$fifo"
for g in "${POOL[@]}"; do echo "$g" >&7; done

PIDS=()
NAMES=()

for line in "${JOBS[@]}"; do
    label=$(echo "$line" | awk '{print $1}')
    dir=$(  echo "$line" | awk '{print $2}')
    if [[ ! -d "$dir" ]]; then
        echo "[score_all] SKIP $label: missing $dir"
        continue
    fi
    # Block here until a GPU slot is available
    read -u 7 gpu
    log="$LOG_DIR/${label}.log"
    echo "[score_all] start  $label  GPU=$gpu  $(date +%T)"
    (
        CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/score_24k.py" \
            "$dir" --label "$label" > "$log" 2>&1
        rc=$?
        # Return the GPU to the pool
        echo "$gpu" >&7
        if (( rc == 0 )); then
            echo "[score_all] DONE   $label  GPU=$gpu  $(date +%T)"
        else
            echo "[score_all] FAIL   $label  GPU=$gpu  rc=$rc  (tail $log)" >&2
            tail -5 "$log" >&2
        fi
        exit $rc
    ) &
    PIDS+=("$!")
    NAMES+=("$label")
done

fail=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        fail=$((fail+1))
    fi
done

exec 7>&- || true

echo
echo "[score_all] all done; failures=$fail"
echo
echo "=== summary ==="
for line in "${JOBS[@]}"; do
    label=$(echo "$line" | awk '{print $1}')
    dir=$(  echo "$line" | awk '{print $2}')
    out="$dir/coco_score.json"
    if [[ -f "$out" ]]; then
        python -c "
import json
d = json.load(open('$out'))
print(f\"  {'$label':<22} FID={d['fid']:6.2f}  CLIP={d['clip_mean']:5.2f}  n={d['n_images']}\")
"
    else
        printf "  %-22s (no result)\n" "$label"
    fi
done
