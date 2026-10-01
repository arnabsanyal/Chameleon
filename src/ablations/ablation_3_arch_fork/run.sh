#!/usr/bin/env bash
# Ablation 3 — Architectural-fork additions.
# 2 LCM W4A8 rows (BAQ on/off) + 3 PixArt-DiT W4A8 rows (m+m / macro / micro).
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

OUT="$OUT_ROOT/ablation_3_arch_fork"
mkdir -p "$OUT"

# ── LCM: BAQ on/off ──────────────────────────────────────────────────────────
# Both rows run at W8A8 under --chameleon-weights. NOTE: MixDQ's native BAQ
# lives only in the accelerated integer qlinear forward; the chameleon fold
# routes every layer through the use_fake_quant_w4 branch, which returns before
# the BOS branch — so the driver's bare bos flag is INERT here (on/off were
# byte-identical, caught by md5 diff). The "on" row therefore uses
# lcm_enable_baq.py, which re-implements BAQ's semantics in the fake-quant path
# (BOS token exempt from A8 quant on attn2 to_k/to_v, computed fp16 with the
# true folded weights).
ROW="$OUT/lcm_baq_on"
if needs_run "$ROW"; then
emit_row_job "ab3_lcm_baq_on" "$ROW" <<EOF
cd "$PATCH_DIR"
python lcm_enable_baq.py --target lcm -- \\
    --w-bit 8 --a-bit 8 --chameleon-weights \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $LCM_STEPS \\
    --guidance-scale 0.0 \\
    --batch-size $AB_BATCH_SIZE \\
    --coco-captions "$COCO_CAPTIONS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

ROW="$OUT/lcm_baq_off"
if needs_run "$ROW"; then
emit_row_job "ab3_lcm_baq_off" "$ROW" <<EOF
cd "$PATCH_DIR"
python lcm_disable_baq.py --target lcm -- \\
    --w-bit 8 --a-bit 8 --chameleon-weights \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $LCM_STEPS \\
    --guidance-scale 0.0 \\
    --batch-size $AB_BATCH_SIZE \\
    --coco-captions "$COCO_CAPTIONS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

# ── DiT: macro+micro / macro-only / micro-only ───────────────────────────────
ROW="$OUT/dit_macro_plus_micro"
if needs_run "$ROW"; then
emit_row_job "ab3_dit_macro_plus_micro" "$ROW" <<EOF
cd "$DIT_DIR"
python chameleon_dit_generation.py \\
    --weight-bits 4 \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $DIT_STEPS \\
    --batch-size $AB_DIT_BATCH \\
    --coco-captions "$COCO_CAPTIONS" \\
    --load-weights "$DIT_W4A8_WEIGHTS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

ROW="$OUT/dit_macro_only"
if needs_run "$ROW"; then
emit_row_job "ab3_dit_macro_only" "$ROW" <<EOF
cd "$PATCH_DIR"
python dit_macro_only.py --target dit -- \\
    --weight-bits 4 \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $DIT_STEPS \\
    --batch-size $AB_DIT_BATCH \\
    --coco-captions "$COCO_CAPTIONS" \\
    --load-weights "$DIT_W4A8_WEIGHTS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

ROW="$OUT/dit_micro_only"
if needs_run "$ROW"; then
emit_row_job "ab3_dit_micro_only" "$ROW" <<EOF
cd "$PATCH_DIR"
python dit_micro_only.py --target dit -- \\
    --weight-bits 4 \\
    --num-samples $NUM_SAMPLES \\
    --num-inference-steps $DIT_STEPS \\
    --batch-size $AB_DIT_BATCH \\
    --coco-captions "$COCO_CAPTIONS" \\
    --load-weights "$DIT_W4A8_WEIGHTS" \\
    --output-dir "$ROW"
bash "$COMMON_DIR/score.sh" "$ROW"
EOF
fi

if [[ "$OWN_DISPATCH" == "1" ]]; then
    dispatch_wait
    echo "=== Ablation 3 complete: $OUT ==="
fi
