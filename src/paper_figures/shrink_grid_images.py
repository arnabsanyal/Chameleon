#!/usr/bin/env python3
"""Downsample + re-encode the 36 qualitative-grid images for Figure 36images.

Each is placed at width=0.15\\textwidth. On the ICLR style that is
0.15 x 5.5in = 0.825in, so a 1024px source is being embedded at ~1240 dpi and
the PDF carries ~30x more pixels than any printer or screen will use. These are
photographs, so PNG (lossless) is also the wrong codec for them.

Target 512px = ~620 dpi, JPEG q=92 with 4:4:4 chroma (no subsampling, so fine
edges and text in the images stay clean).
"""
import glob, os, sys
from PIL import Image

SRC = "generation_output/paper_figures"
DST = "images/paper_figures"
PX, Q = 512, 92

pats = [f"{SRC}/unquantized/*/prompt_*.png", f"{SRC}/*_w4a8/prompt_*.png"]
files = sorted({f for p in pats for f in glob.glob(p)})
if not files:
    sys.exit("no source images found")

before = after = 0
sizes = {}
for f in files:
    im = Image.open(f).convert("RGB")
    sizes[im.size] = sizes.get(im.size, 0) + 1
    rel = os.path.relpath(f, SRC)
    out = os.path.join(DST, os.path.splitext(rel)[0] + ".jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if im.size != (PX, PX):
        im = im.resize((PX, PX), Image.LANCZOS)
    im.save(out, "JPEG", quality=Q, subsampling=0, optimize=True, progressive=True)
    before += os.path.getsize(f)
    after += os.path.getsize(out)

print("source dimensions:", ", ".join(f"{w}x{h} x{n}" for (w, h), n in sizes.items()))
print(f"{len(files)} images")
print(f"  before : {before/1024/1024:7.1f} MB  (PNG, as embedded today)")
print(f"  after  : {after/1024/1024:7.1f} MB  (JPEG {PX}px q{Q})")
print(f"  ratio  : {before/after:7.1f}x smaller")
print(f"  written to {DST}/")
