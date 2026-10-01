#!/usr/bin/env python3
"""Composite each 2x2 cell of Figure 36images into a single image.

36 \\includegraphics become 9, dropping per-image PDF object overhead and the
nested 2x2 tabular inside \\fourblock.

Geometry is baked to match the current output exactly:
  tile        0.15\\textwidth = 0.825in at 512px  ->  620.6 dpi
  gaps        \\hspace{2pt} and \\\\[2pt] = 2pt = 17px at that dpi
  gap colour  the outer column tint, measured from chameleon.pdf page 24:
                UNet     (252, 220, 217)   Salmon!25
                few-step (199, 235, 231)   TealBlue!25
                DiT      (255, 246, 199)   Goldenrod!25
              -- the nested tabular sits inside the tinted cell, so its gaps
              show that tint, not white.

The composite is then placed at width = 0.3\\textwidth + 2pt, so the printed
size is identical.
"""
import os, sys
from PIL import Image

SRC, DST = "generation_output/paper_figures", "images/paper_figures"
TILE, GAP, Q = 512, 17, 92
PROMPTS = ["prompt_1_car", "prompt_2_cat", "prompt_3_dog", "prompt_4_car_night"]
TINT = {"sdxl": (252, 220, 217), "lcm": (199, 235, 231), "dit": (255, 246, 199)}
CELLS = [("unquantized/sdxl", "sdxl"), ("unquantized/lcm", "lcm"), ("unquantized/dit", "dit"),
         ("sdxl_baseline_w4a8", "sdxl"), ("lcm_baseline_w4a8", "lcm"), ("dit_baseline_w4a8", "dit"),
         ("sdxl_chameleon_w4a8", "sdxl"), ("lcm_chameleon_w4a8", "lcm"),
         ("dit_chameleon_w4a8", "dit")]

N = 2 * TILE + GAP
before = after = 0
for cell, fam in CELLS:
    canvas = Image.new("RGB", (N, N), TINT[fam])
    for k, name in enumerate(PROMPTS):
        src = f"{SRC}/{cell}/{name}.png"
        if not os.path.exists(src):
            sys.exit(f"missing {src}")
        before += os.path.getsize(src)
        im = Image.open(src).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.LANCZOS)
        canvas.paste(im, ((k % 2) * (TILE + GAP), (k // 2) * (TILE + GAP)))
    out = f"{DST}/{cell.replace('/', '_')}_grid.jpg"
    os.makedirs(DST, exist_ok=True)
    canvas.save(out, "JPEG", quality=Q, subsampling=0, optimize=True, progressive=True)
    after += os.path.getsize(out)

print(f"9 composites, {N}x{N} px each  ({TILE}px tiles + {GAP}px gaps)")
print(f"  36 source PNGs : {before/1024/1024:6.1f} MB")
print(f"  9 composites   : {after/1024/1024:6.1f} MB   ({before/after:.1f}x smaller)")
print(f"  written to {DST}/*_grid.jpg")
