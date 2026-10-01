#!/usr/bin/env bash
# Ablation 1 — Routing variable contribution (κ vs SNR vs both).
# 4 main rows + 1 threshold-perturbation addendum on SDXL W4A8.
#
# Sourceable: if dispatcher is already active (DISPATCH_INITIALIZED=1) this
# only enqueues jobs. If run standalone, it inits its own 4-GPU dispatcher.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
source "$SCRIPT_DIR/../common/dispatch.sh"

OWN_DISPATCH=0
if [[ "${DISPATCH_INITIALIZED:-0}" != "1" ]]; then
    dispatch_init "${NUM_GPUS:-4}"
    OWN_DISPATCH=1
fi

OUT="$OUT_ROOT/ablation_1_routing"
mkdir -p "$OUT"

_sdxl_w4a8_args() {
    # NB: --load-weights / --load-activation-lut intentionally removed.
    # The cached weights file embeds the activation LUT (calibration_complete=
    # True), which short-circuits the runtime routing function — so monkey-
    # patches on `route_format_by_kurtosis_snr` would never run. Each row now
    # does full calibration from scratch with its patch active.
    cat <<ARG
--weight-bits 4 \\
--num-samples $NUM_SAMPLES \\
--num-inference-steps $SDXL_STEPS \\
--batch-size $AB_BATCH_SIZE \\
--num-calibration-samples $AB_CALIB_SAMPLES \\
--coco-captions "$COCO_CAPTIONS"
ARG
}

# ── Row 1: single format baseline (everything FP8 E4M3, including shortcuts) ─
ROW="$OUT/single_e4m3"
if needs_run "$ROW"; then
emit_row_job "ab1_single_e4m3" "$ROW" <<EOF
cd "$PATCH_DIR"
python palette_force_format.py --target sdxl \\
    --format FLOAT8_E4M3 --include-shortcut -- \\
    $(_sdxl_w4a8_args) \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

# ── Row 2: kurtosis-only (SNR ignored) ───────────────────────────────────────
ROW="$OUT/kurt_only"
if needs_run "$ROW"; then
emit_row_job "ab1_kurt_only" "$ROW" <<EOF
cd "$PATCH_DIR"
python routing_kurt_only.py --target sdxl -- \\
    $(_sdxl_w4a8_args) \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

# ── Row 3: SNR-only (κ forced constant) ──────────────────────────────────────
ROW="$OUT/snr_only"
if needs_run "$ROW"; then
emit_row_job "ab1_snr_only" "$ROW" <<EOF
cd "$PATCH_DIR"
python routing_snr_only.py --target sdxl -- \\
    $(_sdxl_w4a8_args) \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

# ── Row 4: both (default Chameleon) — direct CLI, no patch ──────────────────
ROW="$OUT/both_default"
if needs_run "$ROW"; then
emit_row_job "ab1_both_default" "$ROW" <<EOF
cd "$SDXL_DIR"
python chameleon_generation.py \\
    $(_sdxl_w4a8_args) \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

# ── Row 5 (addendum): threshold robustness, (3,5) → (3.5,5.5) ───────────────
ROW="$OUT/both_perturbed_thresholds"
if needs_run "$ROW"; then
emit_row_job "ab1_both_perturbed" "$ROW" <<EOF
cd "$PATCH_DIR"
python routing_perturb_thresholds.py --target sdxl -- \\
    $(_sdxl_w4a8_args) \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

if [[ "$OWN_DISPATCH" == "1" ]]; then
    dispatch_wait
    echo "=== Ablation 1 complete: $OUT ==="
fi
