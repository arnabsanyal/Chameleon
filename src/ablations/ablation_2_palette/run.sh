#!/usr/bin/env bash
# Ablation 2 — Format palette (single-format and drop-one). 6 SDXL W4A8 rows.
# drop_mxfp8 is the shortcut-handling claim (MXFP8 is only used on shortcuts).
#
# Sourceable: enqueues into shared dispatcher if active, else runs standalone.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
source "$SCRIPT_DIR/../common/dispatch.sh"

OWN_DISPATCH=0
if [[ "${DISPATCH_INITIALIZED:-0}" != "1" ]]; then
    dispatch_init "${NUM_GPUS:-4}"
    OWN_DISPATCH=1
fi

OUT="$OUT_ROOT/ablation_2_palette"
mkdir -p "$OUT"

# Helper: emit one SDXL row. $1 = row name, $2 = patch script, $3+ = patch args.
emit_sdxl_row() {
    local name="$1"; local patch="$2"; shift 2
    local row_out="$OUT/$name"
    needs_run "$row_out" || return 0
    local patch_args="$*"
emit_row_job "ab2_$name" "$row_out" <<EOF
cd "$PATCH_DIR"
python "$patch" --target sdxl $patch_args -- \\
    --weight-bits 4 \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $SDXL_STEPS \\
    --batch-size $AB_BATCH_SIZE \\
    --num-calibration-samples $AB_CALIB_SAMPLES \\
    --coco-captions "$COCO_CAPTIONS" \\
    --output-dir "$row_out"
bash "$COMMON_DIR/score.sh" "$row_out"
EOF
}

# Single-format baselines (every layer, including shortcuts).
emit_sdxl_row single_int8     palette_force_format.py --format INT8_ASYM   --include-shortcut
emit_sdxl_row single_fp8_e4m3 palette_force_format.py --format FLOAT8_E4M3 --include-shortcut

# Drop-one (activation palette).
emit_sdxl_row drop_mxfp8      palette_drop_format.py  --drop MXFP8_E4M3   # ← shortcut claim

# Drop-one (W4 weight palette).
emit_sdxl_row drop_nf4        palette_drop_format.py  --weight-drop NF4
emit_sdxl_row drop_mxfp4      palette_drop_format.py  --weight-drop MXFP4_E2M1

# Full palette (default), via direct CLI.
ROW="$OUT/full_palette"
if needs_run "$ROW"; then
emit_row_job "ab2_full_palette" "$ROW" <<EOF
cd "$SDXL_DIR"
python chameleon_generation.py \\
    --weight-bits 4 \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $SDXL_STEPS \\
    --batch-size $AB_BATCH_SIZE \\
    --num-calibration-samples $AB_CALIB_SAMPLES \\
    --coco-captions "$COCO_CAPTIONS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

if [[ "$OWN_DISPATCH" == "1" ]]; then
    dispatch_wait
    echo "=== Ablation 2 complete: $OUT ==="
fi
