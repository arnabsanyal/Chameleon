#!/usr/bin/env bash
# Phase 1: cache calibration artifacts in parallel across all visible GPUs.
# Only the bit-widths that ablation rows actually load are calibrated:
#     SDXL W4A8   (used by ablations 1, 2, 4)
#     DiT  W4A8   (used by ablation 3)
# Note: Chameleon-LCM is now Chameleon-on-MixDQ and has NO calibration step
# (it uses MixDQ's shipped activation scales + an offline weight SNR search at
# injection), so there is no LCM calibration artifact to cache here.
# Add SDXL W8A8 calls below if you extend the ablation pack to W8A8.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
source "$SCRIPT_DIR/dispatch.sh"

CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
NUM_BUCKETS="${NUM_BUCKETS:-10}"
NUM_GPUS="${NUM_GPUS:-4}"

dispatch_init "$NUM_GPUS"

# ── SDXL W4A8 ────────────────────────────────────────────────────────────────
if [[ ! -f "$SDXL_W4A8_WEIGHTS" || ! -f "$SDXL_W4A8_LUT" ]]; then
    emit_job "calib_sdxl_w4a8" <<EOF
cd "$SDXL_DIR"
python chameleon_generation.py \\
    --weight-bits 4 \\
    --num-buckets $NUM_BUCKETS \\
    --num-calibration-samples $CALIB_SAMPLES \\
    --num-samples 0 \\
    --save-weights "$SDXL_W4A8_WEIGHTS" \\
    --save-activation-lut "$SDXL_W4A8_LUT"
EOF
fi

# ── LCM W4A8 ── retired: Chameleon-on-MixDQ needs no calibration artifact ─────

# ── DiT W4A8 ─────────────────────────────────────────────────────────────────
if [[ ! -f "$DIT_W4A8_WEIGHTS" ]]; then
    emit_job "calib_dit_w4a8" <<EOF
cd "$DIT_DIR"
python chameleon_dit_generation.py \\
    --weight-bits 4 \\
    --num-samples 0 \\
    --save-weights "$DIT_W4A8_WEIGHTS"
EOF
fi

dispatch_wait
echo "=== Calibration cache ready at $CACHE_DIR ==="
