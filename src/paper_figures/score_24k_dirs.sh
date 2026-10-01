#!/usr/bin/env bash
# Score the 11 ready 24,576-image directories for paper Tables 1, 2, 3.
# Writes <dir>/coco_score.json with FID + CLIP mean/std.
#
# Usage:
#     bash score_24k_dirs.sh                     # uses GPU_IDS env or 0,1,7
#     GPU_IDS="0,1,7" bash score_24k_dirs.sh     # explicit
#
# Each dir gets one GPU; FID and CLIP run sequentially on that GPU. Multiple
# dirs run in parallel across the GPU pool via a FIFO semaphore.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ablations/common/paths.sh"

# Default GPU pool: 0, 1, 7 (the dispatcher uses 0,1,2,5,6,7 for ab3 right now;
# 0/1/7 had 45 GB+ free at last check, plenty for CLIP-L/14 + cleanfid).
GPU_IDS_DEFAULT="0,1,7"
GPU_IDS="${GPU_IDS:-$GPU_IDS_DEFAULT}"
IFS=',' read -ra GPU_POOL <<< "$GPU_IDS"
N_GPUS=${#GPU_POOL[@]}

LOG_DIR="$OUT_ROOT/_score_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[score24k] log dir: $LOG_DIR"
echo "[score24k] GPU pool: ${GPU_POOL[*]}"

# FIFO semaphore for GPU slots.
fifo="$(mktemp -u --tmpdir score24k.XXXXXX)"
mkfifo "$fifo"
exec 8<>"$fifo"
rm -f "$fifo"
for g in "${GPU_POOL[@]}"; do echo "$g" >&8; done

# All directories to score, with a short label.
declare -A DIRS=(
    [sdxl_fp16]="$OUT_ROOT/../coco_baseline/images"
    [sdxl_chameleon_w8a8]="$OUT_ROOT/../chameleon/coco_eval/w8a8"
    [sdxl_chameleon_w4a8]="$OUT_ROOT/../chameleon/coco_eval/w4a8"
    [lcm_fp16]="$OUT_ROOT/../lcm/coco_eval"
    [lcm_mixdq_w8a8]="$OUT_ROOT/../mixdq-lcm/coco_eval_w8a8"
    [lcm_mixdq_w4a8]="$OUT_ROOT/../mixdq-lcm/coco_eval_w4a8"
    [lcm_chameleon_w8a8]="$OUT_ROOT/../chameleon-lcm/coco_eval_w8a8"
    [lcm_chameleon_w4a8]="$OUT_ROOT/../chameleon-lcm/coco_eval_w4a8"
    [dit_pixart_fp16]="$OUT_ROOT/../pixart-alpha-dit/coco_eval"
    [dit_qdit_w8a8]="$OUT_ROOT/../q-dit/coco_eval/w8a8_g128"
    [dit_chameleon]="$OUT_ROOT/../chameleon-dit/coco_eval/chameleon_dit_10b"
)

PIDS=()
NAMES=()

score_one() {
    local label="$1" img_dir="$2" gpu="$3"
    local log="$LOG_DIR/${label}.log"
    local out="$img_dir/coco_score.json"

    {
        echo "[score24k] start $label  GPU=$gpu  dir=$img_dir  $(date +%T)"

        # ── Clean-FID ───────────────────────────────────────────────────────
        cd "$REPO_ROOT/src/test"
        local fid_raw
        fid_raw=$(CUDA_VISIBLE_DEVICES="$gpu" python cfid.py \
                    --path "$img_dir" --ref "$COCO_REF_DIR" 2>&1)
        local fid
        fid=$(echo "$fid_raw" | awk '/FID Score:/ {print $3}' | tail -1)
        echo "$fid_raw"
        echo "[score24k] $label fid=$fid"
        if [[ -z "$fid" ]]; then
            echo "[score24k] $label FID FAILED" >&2
            exit 1
        fi

        # ── CLIP score (cap end-ind to image count for speed) ───────────────
        local n_pngs
        n_pngs=$(ls "$img_dir"/*.png 2>/dev/null | wc -l)
        local clip_tmp="$LOG_DIR/${label}.clip.json"
        CUDA_VISIBLE_DEVICES="$gpu" python clip_score.py \
            --images "$img_dir" \
            --coco-captions "$COCO_CAPTIONS" \
            --end-ind "$n_pngs" \
            --batch-size 64 \
            --output "$clip_tmp" 2>&1

        # ── Merge into coco_score.json ──────────────────────────────────────
        python - <<PY
import json, pathlib
clip = json.loads(pathlib.Path("$clip_tmp").read_text())
stats = clip.get("statistics", {})
result = {
    "fid":         float("$fid"),
    "clip_mean":   stats.get("mean"),
    "clip_std":    stats.get("std"),
    "clip_median": stats.get("median"),
    "n_images":    stats.get("num_images"),
    "n_missing":   stats.get("num_missing"),
    "image_dir":   "$img_dir",
    "ref_dir":     "$COCO_REF_DIR",
    "captions":    "$COCO_CAPTIONS",
    "label":       "$label",
}
pathlib.Path("$out").write_text(json.dumps(result, indent=2))
print(f"[score24k] $label FID={result['fid']:.2f} CLIP={result['clip_mean']:.3f} → $out")
PY
        rm -f "$clip_tmp"
        echo "[score24k] end   $label  $(date +%T)"
    } > "$log" 2>&1
    local rc=$?
    echo "$gpu" >&8       # release GPU
    echo "[score24k] $label finished rc=$rc"
    return $rc
}

for label in "${!DIRS[@]}"; do
    img_dir="${DIRS[$label]}"
    if [[ ! -d "$img_dir" ]]; then
        echo "[score24k] SKIP $label: dir does not exist ($img_dir)"
        continue
    fi
    n=$(ls "$img_dir"/*.png 2>/dev/null | wc -l)
    if (( n == 0 )); then
        echo "[score24k] SKIP $label: 0 PNGs in $img_dir"
        continue
    fi

    read -u 8 gpu
    echo "[score24k] queued $label → GPU $gpu  ($n images)"
    score_one "$label" "$img_dir" "$gpu" &
    PIDS+=("$!")
    NAMES+=("$label")
done

# Wait for all backgrounded jobs.
fail=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[score24k] FAILED ${NAMES[$i]}" >&2
        fail=$((fail+1))
    fi
done

exec 8>&- || true
echo
echo "[score24k] all done; failures=$fail"
echo "[score24k] log dir: $LOG_DIR"
echo
echo "=== summary ==="
for label in "${!DIRS[@]}"; do
    img_dir="${DIRS[$label]}"
    out="$img_dir/coco_score.json"
    if [[ -f "$out" ]]; then
        python -c "
import json, sys
d = json.load(open('$out'))
print(f\"  {'$label':<25} FID={d['fid']:6.2f}  CLIP={d['clip_mean']:5.2f}  n={d['n_images']}\")
"
    else
        printf "  %-25s (no score)\n" "$label"
    fi
done
