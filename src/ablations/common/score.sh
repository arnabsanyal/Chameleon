#!/usr/bin/env bash
# Score one row: clean-FID against $COCO_REF_DIR + CLIP score against the
# captions file. Writes result.json into <row_dir>.
#
# Handles the subfolder naming each generator imposes:
#   SDXL → <row>/w4a8/ or <row>/w8a8/
#   DiT  → <row>/w4a8_chameleon_dit_10b/ etc.
#   LCM  → <row>/ (no subfolder)
# We resolve the actual image dir by picking the deepest subdir under <row>
# that contains *.png files.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

ROW_DIR="${1:?usage: score.sh <row_dir> [<captions>]}"
CAPTIONS="${2:-$COCO_CAPTIONS}"
RESULT_JSON="$ROW_DIR/result.json"

if [[ -f "$RESULT_JSON" ]]; then
    echo "[score] $RESULT_JSON exists, skipping"
    exit 0
fi

# Pick the directory that actually contains generated PNGs.
IMG_DIR="$(python - <<PY
import sys
from pathlib import Path
root = Path("$ROW_DIR")
pngs = list(root.rglob("*.png"))
if not pngs:
    print("", end="")
    sys.exit(0)
# All PNGs should share a single parent in our generators; if not, take the
# parent that contains the most PNGs.
from collections import Counter
parents = Counter(p.parent for p in pngs)
print(parents.most_common(1)[0][0])
PY
)"

if [[ -z "$IMG_DIR" ]]; then
    echo "[score] ERROR: no PNGs found under $ROW_DIR" >&2
    exit 1
fi
echo "[score] image dir: $IMG_DIR ($(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l) images)"

if [[ ! -d "$COCO_REF_DIR" ]]; then
    echo "[score] ERROR: COCO_REF_DIR not found: $COCO_REF_DIR" >&2
    echo "[score] set COCO_REF_DIR env var or symlink the val2014 directory." >&2
    exit 1
fi

# ── FID via cleanfid (SAME implementation as src/paper_figures/score_24k.py) ──
# Previously used src/test/cfid.py, which gave FID values that were NOT
# comparable to the headline table (different codebase). Use cleanfid.compute_fid
# exactly like the headline scorer so ablation rows sit on the same axis.
cd "$REPO_ROOT/src/test"
# cleanfid.compute_fid prints its own progress/info to stdout, so we tag the
# value with a marker and extract ONLY that line (capturing all of stdout would
# fold the info text into $fid_value and break the JSON merge below).
fid_value="$(python - <<PY 2>/dev/null | grep '^__FIDVAL__=' | tail -1 | cut -d= -f2
from cleanfid import fid as _fid
v = _fid.compute_fid('$IMG_DIR', '$COCO_REF_DIR')
print(f"__FIDVAL__={v:.6f}")
PY
)"
if [[ -z "$fid_value" ]]; then
    echo "[score] ERROR: cleanfid.compute_fid produced no value for $IMG_DIR" >&2
    exit 1
fi

# ── CLIP via clip_score.py ───────────────────────────────────────────────────
clip_json="$ROW_DIR/.clip_raw.json"
python clip_score.py \
    --images "$IMG_DIR" \
    --coco-captions "$CAPTIONS" \
    --output "$clip_json"

# ── Merge into result.json ───────────────────────────────────────────────────
python - <<PY
import json, pathlib
clip = json.loads(pathlib.Path("$clip_json").read_text())
stats = clip.get("statistics", {})
result = {
    "fid":              float("$fid_value"),
    "clip_mean":        stats.get("mean"),
    "clip_std":         stats.get("std"),
    "n_images":         stats.get("num_images"),
    "n_missing":        stats.get("num_missing"),
    "row_dir":          "$ROW_DIR",
    "image_dir":        "$IMG_DIR",
    "captions":         "$CAPTIONS",
    "ref_dir":          "$COCO_REF_DIR",
}
pathlib.Path("$RESULT_JSON").write_text(json.dumps(result, indent=2))
print(f"[score] FID={result['fid']:.2f} CLIP={result['clip_mean']:.2f}")
PY
rm -f "$clip_json"
echo "[score] wrote $RESULT_JSON"
