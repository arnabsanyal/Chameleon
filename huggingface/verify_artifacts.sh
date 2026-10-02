#!/usr/bin/env bash
# Check that a staged calibration artifact reproduces its Table 1 images before uploading it.
#
#   bash huggingface/verify_artifacts.sh hf_model/sdxl/w8a8/chameleon_sdxl_w8a8.json [num_images]
#
# Regenerates the first num_images (default 8) COCO images from the artifact with --load-weights (so no calibration
# runs) and compares them with the released folder: identical files, or PSNR >= 35 dB for GPU-nondeterminism-level
# differences, mean the artifact is the one behind the table. A different calibration gives visibly different images.
set -euo pipefail

cfg="${1:?usage: verify_artifacts.sh <artifact.json> [num_images]}"
n="${2:-8}"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${CHAMELEON_OUTPUT_ROOT:-$ROOT/outputs}"
CAPS="${CHAMELEON_DATA_ROOT:-$ROOT/data}/coco/annotations/captions_val2014.json"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

case "$(basename "$cfg")" in
  chameleon_sdxl_w8a8.json)   bits=8; ref=chameleon/coco_eval/w8a8 ;;
  chameleon_sdxl_w4a8.json)   bits=4; ref=chameleon/coco_eval/w4a8 ;;
  chameleon_pixart_w8a8.json) bits=8; ref=chameleon-dit/coco_eval/w8a8_chameleon_dit_10b ;;
  chameleon_pixart_w4a8.json) bits=4; ref=chameleon-dit/coco_eval/w4a8_chameleon_dit_10b_iaw ;;
  *) echo "unknown artifact: $cfg" >&2; exit 1 ;;
esac

case "$ref" in
  chameleon/*)
    "$PYTHON" "$ROOT/src/sdxl-chameleon/generation/chameleon_generation.py" --weight-bits "$bits" \
      --load-weights "$cfg" --coco-captions "$CAPS" --num-samples "$n" \
      --num-inference-steps 50 --guidance-scale 7.5 --output-dir "$tmp" ;;
  chameleon-dit/*)
    iaw=(); [ "$bits" = 4 ] && iaw=(--input-aware-weights)
    "$PYTHON" "$ROOT/src/chameleon-dit/generation/chameleon_dit_generation.py" --weight-bits "$bits" "${iaw[@]}" \
      --load-weights "$cfg" --coco-captions "$CAPS" --num-samples "$n" \
      --num-inference-steps 20 --guidance-scale 4.5 --output-dir "$tmp" ;;
esac

"$PYTHON" - "$tmp" "$OUT_ROOT/$ref" "$n" <<'EOF'
import glob, hashlib, os, sys
import numpy as np
from PIL import Image

new_root, ref_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
new = {os.path.basename(p): p for p in glob.glob(os.path.join(new_root, "**", "*.png"), recursive=True)}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
worst, same = float("inf"), 0
for i in range(n):
    name = f"{i:05d}.png"
    a, b = new[name], os.path.join(ref_dir, name)
    if md5(a) == md5(b):
        same += 1
        continue
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    worst = min(worst, psnr)
    print(f"  {name}: PSNR {psnr:.1f} dB")
print(f"{same}/{n} identical" + ("" if same == n else f", worst PSNR {worst:.1f} dB"))
ok = same == n or worst >= 35.0
print("MATCH: artifact reproduces the released images" if ok else "MISMATCH: do not upload this artifact as the Table 1 one")
sys.exit(0 if ok else 1)
EOF
