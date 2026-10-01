"""Shared drawing helpers for the Chameleon paper figures.

One visual system across Figures 2-4: rounded boxes with #2B2B2B strokes and
flat pastel fills, thick black dataflow arrows, thin gray conditioning arrows,
purple format-dispatch arrows, and dashed-border callout panels.

Type is auto-fitted. Widths are measured with Liberation Sans, metrically
identical to the Arial/Helvetica the SVG declares, so measurements match what a
viewer renders. Sizes are computed per GROUP so every box in a row shares one
main-line size instead of short labels ballooning.
"""
import re
from PIL import ImageFont

INK    = "#2B2B2B"
PURPLE = "#7B5EA7"
GRAY   = "#858585"
SW_BOX, SW_ARROW, SW_THIN = 1.7, 3.4, 1.7

FONT = "Arial, Helvetica, sans-serif"
MONO = "'DejaVu Sans Mono', 'Courier New', monospace"

# One palette for all three figures.
C = {
    "latent":   "#CCD6EA",   # tensors in / out
    "stat":     "#CCD6EA",
    "timestep": "#F7E6A2",   # timestep / schedule
    "mid":      "#F7E6A2",
    "down":     "#F6C9A1",   # resolution-changing blocks
    "attn":     "#C3DDB4",   # attention-bearing blocks
    "deep":     "#B7D3C6",
    "conv":     "#D6C3E8",   # conv_in / conv_out / projections
    "shortcut": "#F2C4D4",   # shortcut conv1 (MXFP8-routed)
    "lut":      "#CFE3F5",
    "neutral":  "#EDEDED",
    "int8":     "#CCD6EA",   # format-coded cells
    "e4m3":     "#C3DDB4",
    "e5m2":     "#F6C9A1",
}

_TTF = {"n": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "b": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "i": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "m": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"}
_REF, _cache = 200, {}
LEAD = 1.26


def measure(txt, weight="n"):
    """Width of txt at font-size 1.0, in em."""
    k = (txt, weight)
    if k not in _cache:
        _cache[k] = ImageFont.truetype(_TTF[weight], _REF).getlength(txt) / _REF
    return _cache[k]


def wfit(txt, weight, avail):
    return avail / measure(txt, weight)


def group_base(specs, cap=26.0):
    """specs = [(w, h, padx, pady, lines)] -> one main-line size for the group."""
    b = cap
    for w, h, padx, pady, lines in specs:
        main = next((t for t, rel, _ in lines if rel == 1.0), None)
        if main:
            b = min(b, wfit(main, "n", w - 2 * padx))
        b = min(b, (h - 2 * pady) / (LEAD * sum(rel for _, rel, _ in lines)))
    return b


# ---- inline code: monospace on a light-grey rounded chip (Figures 3 and 4) ----
CODE_BG, CODE_K = "#ECECEC", 0.92

def code_w(t, size):
    """Advance width of a code chip holding t at body size `size`."""
    return measure(t, "m") * size * CODE_K + 4

def chip(f, x, by, t, size, fill="#000000"):
    """Draw a code chip whose left edge is x, on baseline by. Returns its width."""
    w = code_w(t, size)
    f.rect(x, by - size * 0.80, w, size * 1.05, CODE_BG, None, 0, rx=3)
    f.text(x + 2, by, t, size * CODE_K, "n", anchor="start", family=MONO, fill=fill)
    return w

def rich(f, x, by, text, size, weight="n", anchor="middle", fill="#000000"):
    """One line of text with `backtick` spans drawn as code chips. SVG collapses
    leading spaces, so each plain run is placed after its own leading whitespace."""
    segs = [(s[1:-1], True) if s.startswith("`") else (s, False)
            for s in re.split(r"(`[^`]*`)", text) if s]
    widths = [code_w(t, size) if c else measure(t, weight) * size for t, c in segs]
    cx = x - sum(widths) / 2 if anchor == "middle" else (x - sum(widths) if anchor == "end" else x)
    for (t, c), w in zip(segs, widths):
        if c:
            chip(f, cx, by, t, size, fill)
        elif t.strip():
            lead = measure(t[:len(t) - len(t.lstrip())], weight) * size
            f.text(cx + lead, by, t.strip(), size, weight, anchor="start", fill=fill)
        cx += w
    return sum(widths)


class Fig:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.out, self.defs = [], []

    def emit(self, s):
        self.out.append(s)

    @staticmethod
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(self, x, y, txt, size, weight="n", anchor="middle",
             fill="#000000", family=None, rotate=None):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        wt = ' font-weight="bold"' if weight == "b" else ""
        it = ' font-style="italic"' if weight == "i" else ""
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        self.emit(f'  <text x="{x:.1f}" y="{y:.1f}" font-family="{family or FONT}" '
                  f'font-size="{size:.1f}"{wt}{it} fill="{fill}"{a}{tr}>{self.esc(txt)}</text>')

    def box(self, ident, x, y, w, h, fill, lines, base, padx=12, pady=7, rx=9):
        self.emit(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                  f'fill="{fill}" stroke="{INK}" stroke-width="{SW_BOX}"/>')
        sizes = [base if rel == 1.0 else min(base * rel, wfit(t, wt, w - 2 * padx))
                 for t, rel, wt in lines]
        stack = sum(s * LEAD for s in sizes)
        bl = y + h / 2 - stack / 2
        for (t, rel, wt), s in zip(lines, sizes):
            bl += s * LEAD
            self.text(x + w / 2, bl - s * 0.30, t, s, wt)

    def panel(self, ident, x, y, w, h, title=None, tsize=20, dash="10 7", fill="none"):
        self.emit(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="13" '
                  f'fill="{fill}" stroke="{INK}" stroke-width="{SW_BOX}" stroke-dasharray="{dash}"/>')
        if title:
            self.text(x + w / 2, y + 12 + tsize * 0.78, title, tsize, "b")

    def arrow(self, ident, pts, color=INK, sw=SW_ARROW, marker=None, dash=None):
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}" + "".join(
            f" L {x:.1f} {y:.1f}" for x, y in pts[1:])
        mk = marker or {INK: "ahInk", PURPLE: "ahPurple", GRAY: "ahGray"}.get(color, "ahInk")
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.emit(f'  <path id="{ident}" d="{d}" fill="none" stroke="{color}" '
                  f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="butt"{da} '
                  f'marker-end="url(#{mk})"/>')

    def curve(self, ident, x1, y1, cx, cy, x2, y2, color=INK, sw=3.0):
        mk = {INK: "ahInk", PURPLE: "ahPurple", GRAY: "ahGray"}.get(color, "ahInk")
        self.emit(f'  <path id="{ident}" d="M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} '
                  f'{x2:.1f} {y2:.1f}" fill="none" stroke="{color}" stroke-width="{sw}" '
                  f'stroke-linecap="round" marker-end="url(#{mk})"/>')

    def circle(self, ident, cx, cy, r, fill="#ffffff", label=None, lsize=13):
        self.emit(f'  <circle id="{ident}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
                  f'stroke="{INK}" stroke-width="{SW_BOX}"/>')
        if label:
            self.text(cx, cy + lsize * 0.35, label, lsize)

    def rect(self, x, y, w, h, fill, stroke=None, sw=1.0, rx=0):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.emit(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                  f'rx="{rx}" fill="{fill}"{st}/>')

    def header(self):
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" version="1.1">', '  <defs>']
        for mid, col in (("ahInk", INK), ("ahPurple", PURPLE), ("ahGray", GRAY)):
            head.append(f'    <marker id="{mid}" viewBox="0 0 10 10" refX="10" refY="5" '
                        f'markerWidth="4.0" markerHeight="4.0" markerUnits="strokeWidth" '
                        f'orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" '
                        f'fill="{col}"/></marker>')
        head += self.defs + ['  </defs>',
                             f'  <rect width="{self.w}" height="{self.h}" fill="#ffffff"/>']
        return head

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.header() + self.out + ["</svg>"]) + "\n")
        print("wrote", path)
