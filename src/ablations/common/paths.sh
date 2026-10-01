# Paths: override CHAMELEON_OUTPUT_ROOT / CHAMELEON_DATA_ROOT; defaults live under the repo root
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHAMELEON_OUTPUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$CHAMELEON_ROOT/outputs}"
CHAMELEON_DATA_ROOT="${CHAMELEON_DATA_ROOT:-$CHAMELEON_ROOT/data}"

# Shared paths and environment for every ablation runner.
# Sourced by ablation_*/run.sh; do not execute directly.

set -euo pipefail

# Repo + per-model code dirs
REPO_ROOT="${REPO_ROOT:-$CHAMELEON_ROOT}"
SDXL_DIR="$REPO_ROOT/src/sdxl-chameleon/generation"
LCM_DIR="$REPO_ROOT/src/chameleon-lcm/generation"
DIT_DIR="$REPO_ROOT/src/chameleon-dit/generation"
PATCH_DIR="$REPO_ROOT/src/ablations/patches"
COMMON_DIR="$REPO_ROOT/src/ablations/common"

# Output root — every row writes to a sibling subdir under here.
OUT_ROOT="${OUT_ROOT:-$CHAMELEON_OUTPUT_ROOT/ablations}"

# Cached calibration artifacts (filled by common/cache_calibration.sh)
CACHE_DIR="$OUT_ROOT/_calibration_cache"
SDXL_W8A8_WEIGHTS="$CACHE_DIR/sdxl_w8a8_weights.pt"
SDXL_W8A8_LUT="$CACHE_DIR/sdxl_w8a8_lut.pt"
SDXL_W4A8_WEIGHTS="$CACHE_DIR/sdxl_w4a8_weights.pt"
SDXL_W4A8_LUT="$CACHE_DIR/sdxl_w4a8_lut.pt"
# Retired: Chameleon-LCM is now Chameleon-on-MixDQ (no cached calibration config).
# Kept as no-ops for backward compat with any external scripts that source them.
LCM_W4A8_CFG="$CACHE_DIR/lcm_w4a8_config.pt"
LCM_W8A8_CFG="$CACHE_DIR/lcm_w8a8_config.pt"
DIT_W4A8_WEIGHTS="$CACHE_DIR/dit_w4a8_weights.pt"

# COCO captions JSON. We use the full annotations file and let each row
# slice the first $NUM_SAMPLES with --num-samples; that gives every row the
# same 1k-prompt subset (deterministic seeds via global-prompt-index).
COCO_CAPTIONS="${COCO_CAPTIONS:-$CHAMELEON_DATA_ROOT/coco/annotations/captions_val2014.json}"

# Directory of reference images for FID. cfid.py wraps cleanfid which scans
# this directory recursively; the Inception activations are cached under
# ~/.cache/cleanfid/ on first use so subsequent rows are fast.
COCO_REF_DIR="${COCO_REF_DIR:-$CHAMELEON_DATA_ROOT/coco/val2014}"

# Inference protocol. Env-overridable so a 5k driver can bump NUM_SAMPLES
# without editing this file (keep it CONSISTENT across all arms of a comparison).
# LCM is now SDXL-Turbo @ 1 step (was Dreamshaper 4-step) — default fixed to 1.
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SDXL_STEPS="${SDXL_STEPS:-50}"
LCM_STEPS="${LCM_STEPS:-1}"
DIT_STEPS="${DIT_STEPS:-20}"

# Throughput knobs — these do NOT affect FID (batching is per-prompt
# deterministic; the LUT from 32 vs 128 calib samples is stable and is applied
# identically to every arm). Targets ~65 GB on 80 GB cards. SDXL/LCM use
# AB_BATCH_SIZE; DiT uses the smaller AB_DIT_BATCH because it co-loads the 4.3 B
# T5-XXL encoder. There is no auto-fill in these drivers, so tune by measuring:
# if a card sits well under 65 GB, relaunch with a bigger AB_BATCH_SIZE; if it
# OOMs, drop it. (batch 1 ≈ 15 GB, so ~16 should land near 55–65 GB for SDXL.)
AB_BATCH_SIZE="${AB_BATCH_SIZE:-16}"
AB_DIT_BATCH="${AB_DIT_BATCH:-6}"
AB_CALIB_SAMPLES="${AB_CALIB_SAMPLES:-32}"

# conda env. Activate `chameleon` via miniforge if not already on a conda env.
# We do this unconditionally rather than relying on the caller, so that
# `nohup bash run_all.sh ...` works from a fresh login shell.
CHAMELEON_CONDA_ENV="${CHAMELEON_CONDA_ENV:-chameleon}"
CHAMELEON_CONDA_BASE="${CHAMELEON_CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CHAMELEON_CONDA_ENV" ]]; then
    if [[ -f "$CHAMELEON_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # `conda activate` requires `set +u` because it touches unbound vars.
        set +u
        # shellcheck disable=SC1091
        source "$CHAMELEON_CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CHAMELEON_CONDA_ENV"
        set -u
    elif ! command -v python >/dev/null 2>&1; then
        echo "[paths.sh] ERROR: cannot find conda at $CHAMELEON_CONDA_BASE and no python on PATH" >&2
        exit 1
    fi
fi

mkdir -p "$OUT_ROOT" "$CACHE_DIR"
