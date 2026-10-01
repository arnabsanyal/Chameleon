#!/usr/bin/env bash
# Build paper figures: SVG -> PDF + PNG, in one step.
#
# The SVG builders run on system python3 (they only need PIL for font metrics),
# but the rasterisers live in the chi-sched env. Keeping the two halves in one
# script stops the PDF/PNG going stale behind an edited SVG.
set -e
cd "$(dirname "$0")/../.."
RENDER="${RENDER:-python}"

FIGS=("${@:-chameleon_schematic fig3_unet_schematic fig4_dit_schematic}")
declare -A BUILDER=(
  [chameleon_schematic]=build_fig2_svg.py
  [fig3_unet_schematic]=build_fig3_svg.py
  [fig4_dit_schematic]=build_fig4_svg.py
)

for name in ${FIGS[@]}; do
    b="${BUILDER[$name]}"
    [ -n "$b" ] || { echo "unknown figure: $name" >&2; exit 1; }
    python3 "src/paper_figures/$b" >/dev/null
    "$RENDER" - "$name" <<'PY'
import sys, resvg_py, cairosvg, numpy as np
from PIL import Image
n = sys.argv[1]
svg = open(f"images/{n}.svg").read()
open(f"images/{n}.png", "wb").write(bytes(resvg_py.svg_to_bytes(svg_string=svg)))
cairosvg.svg2pdf(url=f"images/{n}.svg", write_to=f"images/{n}.pdf")
cairosvg.svg2png(url=f"images/{n}.svg", write_to="/tmp/_x.png", scale=1.0)
A = np.array(Image.open("/tmp/_x.png").convert("RGB")).astype(int)
B = np.array(Image.open(f"images/{n}.png").convert("RGB")).astype(int)
d = (np.abs(A - B).max(axis=2) > 60).mean() * 100
print(f"  {n}: svg+pdf+png written, cross-renderer diff {d:.2f}%")
PY
done
