#!/usr/bin/env python3
"""
Figure 2 (Chameleon framework schematic) as clean, hand-editable SVG.

Every arrow is its own <path> with an id, so deleting or re-pointing one is a
one-line edit. Panels use real stroke-dasharray. Text is live <text> in
Arial/Helvetica.

Type is AUTO-FITTED: for each box the largest base size is chosen such that the
widest line still fits the inner width and the stack still fits the inner
height. Widths are measured with Liberation Sans, which is metrically identical
to Arial, so what is measured here is what a viewer renders.

Style matches Figures 3 and 4 (dit.png, unet-mod.png): rounded boxes, #2B2B2B
strokes, pastel fills, thick black dataflow arrows, thin gray leaders, and
dashed-border callout panels with a bold title over a regular subtitle.

Semantics that MUST hold (paper Sections 3.1, 4.6, related-work few-step para):
  * the weight palette feeds ALL THREE fork arms;
  * the activation LUT feeds ONLY the UNet and DiT arms -- a one-step model
    has no timestep axis to route over, so the few-step arm takes MixDQ's
    calibrated static activation scales instead;
  * the activation chain is Timestep t -> Activation Statistics -> LUT.

Regenerate:  python3 build_fig2_svg.py
Verify:      python3 check_fig2_svg.py
"""
import os
from PIL import ImageFont

W, H = 1376, 768

INK, GRAY = "#2B2B2B", "#858585"
ROYAL = "#4169E1"     # weight-palette fan
BRICK = "#A83A2C"     # activation-LUT fan
ZOOM  = "#A8A8A8"     # zoom-indicator lines: must not compete with the arrows
SW_BOX, SW_ARROW = 1.7, 3.4

FILL = {"timestep": "#F7E6A2", "stat": "#CCD6EA", "palette": "#C3DDB4",
        "lut": "#CFE3F5", "unet": "#F2C4D4", "fewstep": "#F6C9A1",
        "dit": "#D6C3E8", "dispatch": "#DDE5F0"}

FONT = "Arial, Helvetica, sans-serif"
MONO = "'DejaVu Sans Mono', 'Courier New', monospace"

_TTF = {
    "n": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "b": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "i": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "m": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}
_REF = 200          # measure at this size, scale linearly
_cache = {}


def measure(txt, weight="n"):
    """Width of txt at font-size 1.0, in em units."""
    key = (txt, weight)
    if key not in _cache:
        f = ImageFont.truetype(_TTF[weight], _REF)
        _cache[key] = f.getlength(txt) / _REF
    return _cache[key]


LEAD = 1.26         # line height as a multiple of font size


def _wfit(txt, weight, avail):
    return avail / measure(txt, weight)


def group_base(specs, cap=26.0):
    """specs = [(w, h, padx, pady, lines)]. One main-line size for the group.

    The MAIN line (rel 1.0) is sized identically across every box in the group
    so the figure reads as one system; sub-lines are width-capped per box so a
    long parenthetical shrinks on its own instead of dragging every main label
    down with it.
    """
    b = cap
    for w, h, padx, pady, lines in specs:
        main = next(t for t, rel, _ in lines if rel == 1.0)
        b = min(b, _wfit(main, "n", w - 2 * padx))
        b = min(b, (h - 2 * pady) / (LEAD * sum(rel for _, rel, _ in lines)))
    return b


out = []
def emit(s): out.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, txt, size, weight="n", anchor="middle", fill="#000000", family=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    wt = ' font-weight="bold"' if weight == "b" else ""
    it = ' font-style="italic"' if weight == "i" else ""
    emit(f'  <text x="{x:.1f}" y="{y:.1f}" font-family="{family or FONT}" '
         f'font-size="{size:.1f}"{wt}{it} fill="{fill}"{a}>{esc(txt)}</text>')


def box(ident, x, y, w, h, fill, lines, base, padx=15, pady=9):
    emit(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
         f'fill="{fill}" stroke="{INK}" stroke-width="{SW_BOX}"/>')
    sizes = []
    for txt, rel, wt in lines:
        s = base if rel == 1.0 else min(base * rel, _wfit(txt, wt, w - 2 * padx))
        sizes.append(s)
    stack = sum(s * LEAD for s in sizes)
    baseline = y + h / 2 - stack / 2
    for (txt, rel, wt), s in zip(lines, sizes):
        baseline += s * LEAD
        text(x + w / 2, baseline - s * 0.30, txt, s, wt)


PANEL_TS = None      # set once, shared by all three panel titles


DASH = "10 7"        # one definition: panel() and callout() must not drift apart


def panel(ident, x, y, w, h, title):
    emit(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="13" '
         f'fill="none" stroke="{INK}" stroke-width="{SW_BOX}" stroke-dasharray="{DASH}"/>')
    text(x + w / 2, y + 12 + PANEL_TS * 0.78, title, PANEL_TS, "b")


def callout(ident, x, y, w, h):
    emit(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="13" '
         f'fill="none" stroke="{INK}" stroke-width="{SW_BOX}" stroke-dasharray="{DASH}"/>')


def arrow(ident, x1, y1, x2, y2, color=INK, sw=SW_ARROW, marker="ahInk"):
    emit(f'  <path id="{ident}" d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" '
         f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="butt" '
         f'marker-end="url(#{marker})"/>')


# ------------------------------------------------------------------ layout --
P1 = (30, 48, 540, 462)
P2 = (598, 48, 320, 462)
P3 = (946, 48, 400, 462)
ROW = (158, 292, 426)

emit(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     f'width="{W}" height="{H}" version="1.1">')
emit('  <defs>')
for mid, col in (("ahInk", INK), ("ahRoyal", ROYAL), ("ahBrick", BRICK),
                 ("ahGray", GRAY)):
    emit(f'    <marker id="{mid}" viewBox="0 0 10 10" refX="10" refY="5" '
         f'markerWidth="4.0" markerHeight="4.0" markerUnits="strokeWidth" '
         f'orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" '
         f'fill="{col}"/></marker>')
emit('  </defs>')
emit(f'  <rect width="{W}" height="{H}" fill="#ffffff"/>')

PANEL_TS = min(23.0, min((p[2] - 34) / measure(t, "b") for p, t in (
    (P1, "Offline Analysis"), (P2, "Load-Time Architectural Fork"),
    (P3, "Inference Execution Graphs"))))

panel("panel-offline", *P1, "Offline Analysis")
panel("panel-fork",    *P2, "Load-Time Architectural Fork")
panel("panel-exec",    *P3, "Inference Execution Graphs")

SUB = 0.80          # nominal sub-line size relative to the main line

# ------------------------------------------------------- panel 1 content ----
L_WSNR  = [("Weight SQNR", 1.0, "n"), ("(per-format round-trip)", SUB, "n")]
L_PAL   = [("Adaptive Weight Palette", 1.0, "n"),
           ("argmax-SQNR format search", SUB, "n"),
           ("W8: {INT8, MXINT8}", SUB, "n"),
           ("W4: {INT4, NF4, FP4, MX4}", SUB, "n")]
L_TSTEP = [("Timestep t", 1.0, "n")]
L_STATS = [("Activation Statistics", 1.0, "n"),
           ("(Kurtosis κ, per layer × bucket)", SUB, "n")]
# SNR(t) needs only alphas_cumprod -- no activations, no forward pass (Sec. 1.1),
# so it gets its own box fed straight from the timestep.
L_SNR   = [("Diffusion SNR(t)", 1.0, "n"),
           ("(no forward pass)", SUB, "n")]
L_LUT   = [("Per-(Layer × Timestep-Bucket)", SUB, "n"),
           ("Activation Format LUT", 1.0, "n"),
           ("{INT8, FP8, MX}", SUB, "n")]

G_WSNR  = (40, 112, 210, 92)
G_PAL   = (290, 94, 270, 128)
G_TSTEP = (40, 264, 170, 60)
G_STATS = (250, 252, 310, 84)
G_SNR   = (40, 396, 170, 62)
G_LUT   = (250, 374, 310, 106)

b1 = group_base([(G_WSNR[2], G_WSNR[3], 15, 9, L_WSNR),
                 (G_PAL[2],  G_PAL[3],  15, 9, L_PAL),
                 (G_TSTEP[2],G_TSTEP[3],15, 9, L_TSTEP),
                 (G_SNR[2],  G_SNR[3],  12, 7, L_SNR),
                 (G_STATS[2],G_STATS[3],15, 9, L_STATS),
                 (G_LUT[2],  G_LUT[3],  15, 9, L_LUT)])
box("box-weight-snr", *G_WSNR,  FILL["stat"],     L_WSNR,  b1)
box("box-palette",    *G_PAL,   FILL["palette"],  L_PAL,   b1)
box("box-timestep",   *G_TSTEP, FILL["timestep"], L_TSTEP, b1)
box("box-diff-snr",   *G_SNR,   FILL["timestep"], L_SNR,   b1, padx=12)
box("box-act-stats",  *G_STATS, FILL["stat"],     L_STATS, b1)
box("box-act-lut",    *G_LUT,   FILL["lut"],      L_LUT,   b1)

# ------------------------------------------------------- panel 2: forks -----
FORKS = [("box-unet",    "unet",    "UNet Diffusion",       "(Q-Diffusion style)",       ROW[0]),
         ("box-fewstep", "fewstep", "Few-Step Distilled",    "(SDXL-Turbo, MixDQ style)", ROW[1]),
         ("box-dit",     "dit",     "Diffusion Transformer", "(Q-DiT style)",             ROW[2])]
FW, FH = 284, 100
b2 = group_base([(FW, FH, 15, 9, [(m, 1.0, "n"), (sb, SUB, "n")]) for _, _, m, sb, _ in FORKS])
for ident, key, main, sub, cy in FORKS:
    box(ident, 616, cy - FH // 2, FW, FH, FILL[key], [(main, 1.0, "n"), (sub, SUB, "n")], b2)

# ---------------------------------------------------- panel 3: dispatch -----
# UNet: weights are chosen per output channel over FORMATS only -- the joint
# (g,f) search is the DiT's change (Sec. 4.6(i)); and the activation scales are
# read from the calibrated LUT, not computed at inference (Sec. 4.6, UNet path).
# The UNet cell carries a third detail line rather than one long one, so the
# MXFP8 shortcut fact survives without dragging the whole group's type down.
EXEC = [("box-exec-unet", ["W: per-output-channel format",
                           "A: per-bucket format, static scales",
                           "+ MXFP8 shortcut layers"], ROW[0]),
        ("box-exec-fewstep", ["W: per-channel (8b), (g,f) (4b)",
                              "A: MixDQ static scales (1-step)"], ROW[1]),
        ("box-exec-dit", ["W: input-aware (g,f) search",
                          "A: macro-route + dynamic per-sample scale"], ROW[2])]
EW, EH = 368, 104


def _exec_lines(tail):
    return [("Fake-Quant Dispatch", 1.0, "n")] + [(t, 0.86, "n") for t in tail]


b3 = group_base([(EW, EH, 14, 9, _exec_lines(tail)) for _, tail, _ in EXEC])
for ident, tail, cy in EXEC:
    box(ident, 962, cy - EH // 2, EW, EH, FILL["dispatch"], _exec_lines(tail), b3, padx=14)

# ----------------------------------------------------------- offline edges --
arrow("edge-wsnr-palette", 250, 158, 290, 158)
arrow("edge-tstep-stats",  210, 294, 250, 294)
arrow("edge-tstep-snr",    125, 324, 125, 396)
arrow("edge-snr-lut",      210, 427, 250, 427)
arrow("edge-stats-lut",    405, 336, 405, 374)

# --------------------------- weight palette -> ALL THREE forks (black) ------
arrow("edge-palette-unet",    560, 140, 616, ROW[0] - 10, color=ROYAL, marker="ahRoyal")
arrow("edge-palette-fewstep", 560, 168, 616, ROW[1] - 14, color=ROYAL, marker="ahRoyal")
arrow("edge-palette-dit",     560, 200, 616, ROW[2] - 18, color=ROYAL, marker="ahRoyal")

# --------- activation LUT -> UNet and DiT ONLY (purple, format dispatch) ---
# NOTE: there is deliberately NO edge-lut-fewstep. Do not add one.
arrow("edge-lut-unet", 560, 396, 616, ROW[0] + 16, color=BRICK, marker="ahBrick")
arrow("edge-lut-dit",  560, 454, 616, ROW[2] + 14, color=BRICK, marker="ahBrick")

# ------------------------------------------------------- fork -> dispatch ---
for i, name in enumerate(("unet", "fewstep", "dit")):
    arrow(f"edge-{name}-exec", 900, ROW[i], 962, ROW[i])

# ------------------------------------------------------------- callout A ----
CA = (716, 526, 630, 228)
callout("callout-fewstep", *CA)
ts = min(24.0, (CA[2] - 60) / measure("Few-Step Exception", "b"))
text(CA[0] + CA[2] / 2, CA[1] + 14 + ts * 0.78, "Few-Step Exception", ts, "b")

# SVG collapses a leading space in a <text>, so the gap after the bold head has
# to come from the x-offset, not from the string.
leg = [(ROYAL, "ahRoyal", "weights:", "adaptive palette feeds all three arms"),
       (BRICK, "ahBrick", "activations:", "format LUT feeds the multi-step arms only")]
ls = min(18.0, (CA[2] - 130) / max(measure(h + " " + t, "n") for _, _, h, t in leg))
for i, (col, mk, head, tail) in enumerate(leg):
    y = CA[1] + 62 + i * (ls * LEAD + 8)
    emit(f'  <path d="M {CA[0]+26} {y-ls*0.32:.1f} L {CA[0]+80} {y-ls*0.32:.1f}" '
         f'fill="none" stroke="{col}" stroke-width="{SW_ARROW}" marker-end="url(#{mk})"/>')
    text(CA[0] + 94, y, head, ls, "b", anchor="start")
    text(CA[0] + 94 + measure(head + " ", "b") * ls, y, tail, ls, "n", anchor="start")

body = ["A single sampler step leaves no timestep axis to route",
        "over, so the few-step arm inherits MixDQ's calibrated",
        "static activation scales unchanged. Chameleon supplies",
        "only its weights."]
bs = min(19.0, (CA[2] - 52) / max(measure(t, "n") for t in body))
y0 = CA[1] + 132
for i, ln in enumerate(body):
    text(CA[0] + 26, y0 + i * bs * LEAD, ln, bs, "n", anchor="start")

# ------------------------------------------------------------- callout B ----
CB = (30, 526, 658, 228)
callout("callout-routing", *CB)
ts = min(24.0, (CB[2] - 60) / measure("Routing Rule", "b"))
text(CB[0] + CB[2] / 2, CB[1] + 14 + ts * 0.78, "Routing Rule", ts, "b")
ss = min(18.0, (CB[2] - 80) / measure("Applied per (layer, timestep bucket).", "i"))
text(CB[0] + CB[2] / 2, CB[1] + 62, "Applied per (layer, timestep bucket).", ss, "i")

rules = [("shortcut layer", "MXFP8 E4M3"),
         ("κ > 5 or SNR < 0.2", "FP8 E5M2"),
         ("κ ≤ 3 and SNR > 2", "INT8-asym"),
         ("otherwise", "FP8 E4M3")]
BX, BW = CB[0] + 22, CB[2] - 44
c1 = max(measure(a, "m") for a, _ in rules)
c2 = max(measure("→ " + b, "m") for _, b in rules)
ms = min(18.0, (BW - 56) / (c1 + c2 + 1.2))
bh = 4 * ms * LEAD + 22
emit(f'  <rect x="{BX}" y="{CB[1]+76}" width="{BW}" height="{bh:.1f}" rx="7" '
     f'fill="#F2F2F2" stroke="#CCCCCC" stroke-width="1"/>')
for i, (cond, fmt) in enumerate(rules):
    y = CB[1] + 76 + 11 + (i + 1) * ms * LEAD - ms * 0.30
    text(BX + 20, y, cond, ms, "n", anchor="start", family=MONO)
    text(BX + 28 + (c1 + 0.8) * ms, y, "→ " + fmt, ms, "n", anchor="start", family=MONO)
foot = "Shortcut rule is checked first, before any statistic."
fs = min(17.0, (CB[2] - 60) / measure(foot, "n"))
foot_y = CB[1] + CB[3] - 16                      # anchored to the panel bottom
assert foot_y - fs > CB[1] + 76 + bh, "routing block collides with the footnote"
text(CB[0] + CB[2] / 2, foot_y, foot, fs, "n", fill="#333333")

# Zoom-indicator lines tying the Routing Rule panel to the box it expands:
# each top corner of the callout to the corresponding bottom corner of the LUT.
# Same dashed style as every panel perimeter, and no arrowheads -- these mark a
# correspondence, not a dataflow.
# Both boxes have rounded corners, so the mathematical corner is not on the
# drawn border -- the nearest point along the diagonal is inset by r(1-1/sqrt2).
# Aiming at the sharp corner left a visible gap at each end.
def _inset(r):
    return r * (1 - 2 ** -0.5)


LUT_X, LUT_W, LUT_BOT = G_LUT[0], G_LUT[2], G_LUT[1] + G_LUT[3]
kb, kc = _inset(10), _inset(13)          # box rx=10, callout rx=13
for nm, sgn, lx in (("left", +1, LUT_X), ("right", -1, LUT_X + LUT_W)):
    x1, y1 = CB[0] + sgn * kc if nm == "left" else CB[0] + CB[2] - kc, CB[1] + kc
    x2, y2 = lx + sgn * kb, LUT_BOT - kb
    emit(f'  <path id="zoom-{nm}" d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" fill="none" '
         f'stroke="{ZOOM}" stroke-width="{SW_BOX}" stroke-dasharray="{DASH}"/>')


emit('</svg>')

path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "images/chameleon_schematic.svg")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote", path)
