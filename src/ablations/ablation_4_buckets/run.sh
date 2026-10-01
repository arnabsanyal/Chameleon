#!/usr/bin/env bash
# Ablation 4 — Bucket count sweep on SDXL W4A8.
# B ∈ {1, 5, 10, 20}. B=1 doubles as the no-temporal-routing baseline.
#
# Each B needs its own LUT calibration (LUT length depends on B). We bundle
# calib + generation into ONE per-B job so the dispatcher schedules the whole
# B=k unit on a single GPU end-to-end — no inter-GPU coordination needed and
# no idle time waiting for a separate calibration phase.
#
# Weight quantization is reused from $SDXL_W4A8_WEIGHTS (independent of B).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
source "$SCRIPT_DIR/../common/dispatch.sh"

OWN_DISPATCH=0
if [[ "${DISPATCH_INITIALIZED:-0}" != "1" ]]; then
    dispatch_init "${NUM_GPUS:-4}"
    OWN_DISPATCH=1
fi

OUT="$OUT_ROOT/ablation_4_buckets"
mkdir -p "$OUT"

for B in 1 5 10 20; do
    ROW="$OUT/b${B}"
    LUT="$CACHE_DIR/sdxl_w4a8_lut_b${B}.pt"
    needs_run "$ROW" || continue
emit_row_job "ab4_b${B}" "$ROW" <<EOF
if [[ ! -f "$LUT" ]]; then
    echo "=== B=$B : calibrating LUT ==="
    cd "$SDXL_DIR"
    python chameleon_generation.py \\
        --weight-bits 4 \\
        --num-buckets $B \\
        --num-calibration-samples $AB_CALIB_SAMPLES \\
        --num-samples 0 \\
        --load-weights "$SDXL_W4A8_WEIGHTS" \\
        --save-activation-lut "$LUT"
fi
echo "=== B=$B : generating ==="
cd "$SDXL_DIR"
python chameleon_generation.py \\
    --weight-bits 4 \\
    --num-buckets $B \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $SDXL_STEPS \\
    --batch-size $AB_BATCH_SIZE \\
    --coco-captions "$COCO_CAPTIONS" \\
    --load-weights "$SDXL_W4A8_WEIGHTS" \\
    --load-activation-lut "$LUT" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
done

if [[ "$OWN_DISPATCH" == "1" ]]; then
    dispatch_wait
    echo "=== Ablation 4 complete: $OUT ==="
fi
