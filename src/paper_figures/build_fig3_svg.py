#!/usr/bin/env python3
"""Figure 3 -- SDXL UNet execution path, drawn as clean editable SVG.

U-shape: encoder steps down-right (DownBlock2D -> CrossAttnDown 640 -> 1280), the
cross-attention mid block sits at the base, the decoder climbs back up. At every
level the concat (c) takes BOTH inputs diffusers concatenates -- the encoder skip
from the left and the decoder stream from below -- and feeds the up-block ResNets'
conv1 (x3 per level, 9 in total: the layers pinned to MXFP8).

Type sizes follow Figure 4 (both figures are 1376 px wide and set at \\textwidth).
Activation table = measured shares from the shipped SDXL LUTs (see SHARE below).

Regenerate: ./src/paper_figures/build_figs.sh fig3_unet_schematic
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import Fig, C, INK, GRAY, SW_BOX, group_base, wfit, measure, MONO, code_w, chip

W, H = 1376, 1080
f = Fig(W, H)
SUB = 0.80
LBL = "#555555"                                   # italic wire labels, as in Figure 4
STRIP = ["#E7C5C6", "#F0B9B6", "#FDD6B3", "#F9F2C1", "#D4E3BC", "#C2DFE9", "#C4BCDD", "#DFCCE5"]
BRICK, CAT = "#B15A47", "#EAF2FC"
ROYAL, BUS_OP, BUS_SW = "#4169E1", 0.75, 1.8   # bus colours at 75% opacity
SWEEP = ["#F1B695", "#F8C796", "#FADA90", "#F5E39B", "#BBD1AA",   # Figure 4's bucket sweep
         "#B7CDA8", "#98C4C7", "#8EB1CE", "#A9ADD2", "#AC99C7"]
FMT_CELL = "#D4D4D4"
f.defs.append('    <linearGradient id="schedGrad" x1="0" y1="0" x2="0" y2="1">' +
              "".join(f'<stop offset="{i/9:.3f}" stop-color="{c}"/>'
                      for i, c in enumerate(SWEEP)) + '</linearGradient>')

def rbox(ident, r, fill, lines, base, padx=10, pady=6):
    f.box(ident, r[0], r[1], r[2] - r[0], r[3] - r[1], fill, lines, base, padx=padx, pady=pady)
def wire(ident, pts, sw=1.8, color=GRAY, op=1.0):  # bus trunk: no arrowhead
    d = f"M {pts[0][0]} {pts[0][1]}" + "".join(f" L {x} {y}" for x, y in pts[1:])
    f.emit(f'  <path id="{ident}" d="{d}" fill="none" stroke="{color}" stroke-opacity="{op}" '
           f'stroke-width="{sw}" stroke-linejoin="round"/>')
def dot(x, y, color=GRAY, op=1.0): f.emit(f'  <circle cx="{x}" cy="{y}" r="3.2" fill="{color}" fill-opacity="{op}"/>')
# translucent arrowheads: the marker's BASE sits on the path end (refX=0) and the
# line stops one head-length short, so line and head never overlap into a darker spot
for _mid, _col in (("ahBrickT", BRICK), ("ahRoyalT", ROYAL)):
    f.defs.append(f'    <marker id="{_mid}" viewBox="0 0 10 10" refX="0" refY="5" markerWidth="4" '
                  f'markerHeight="4" markerUnits="strokeWidth" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" '
                  f'fill="{_col}" fill-opacity="{BUS_OP}"/></marker>')
def busarrow(ident, pts, color, marker, sw=BUS_SW, op=BUS_OP):
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** .5; hl = 4 * sw
    end = (x1 - (x1 - x0) * hl / L, y1 - (y1 - y0) * hl / L)
    q = pts[:-1] + [end]
    d = f"M {q[0][0]:.1f} {q[0][1]:.1f}" + "".join(f" L {x:.1f} {y:.1f}" for x, y in q[1:])
    f.emit(f'  <path id="{ident}" d="{d}" fill="none" stroke="{color}" stroke-opacity="{op}" '
           f'stroke-width="{sw}" stroke-linejoin="round" marker-end="url(#{marker})"/>')

# ---- rich text: `backticks` = code (monospace on a grey chip); wrap + justify ----
LEAD2, PGAP = 1.45, 6
RAGGED = []                                        # justified lines that fell back
def _pieces(word):
    if word == "→": return [("→", "arrow")]
    return [(seg[1:-1], True) if seg.startswith("`") else (seg, False)
            for seg in re.split(r"(`[^`]*`)", word) if seg]
def _pw(t, code, size):
    if code == "arrow": return size * 1.3
    return code_w(t, size) if code else measure(t, "n") * size
def _frags(chunk, size):
    """Words -> fragments (pieces, width, spaced_before). Plain hyphenated words may
    break after an existing hyphen ('peak-to-median' -> 'peak-' 'to-' 'median');
    words containing code never split."""
    out = []
    for w in chunk.split():
        pcs = _pieces(w)
        parts = [w] if any(c for _, c in pcs) else [q for q in re.split(r"(?<=-)(?=[^-])", w) if q]
        for k, q in enumerate(parts):
            qp = pcs if len(parts) == 1 else [(q, False)]
            out.append((qp, sum(_pw(t, c, size) for t, c in qp), k == 0))
    return out
def _lw(F, i, j, sp):
    return sum(F[k][1] for k in range(i, j)) + sp * sum(1 for k in range(i + 1, j) if F[k][2])
def _breaks(F, sp, width):
    """Minimum-raggedness breaking: min sum of squared slack; last line free;
    one-fragment lines (other than the last) penalised."""
    n = len(F); INF = float("inf"); best = [0.0] + [INF] * n; prev = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            w = _lw(F, i, j, sp)
            if w > width and j - i > 1: break
            cost = 0.0 if j == n else (width - w) ** 2 + (4000.0 if j - i == 1 else 0.0)
            if best[i] + cost < best[j]: best[j], prev[j] = best[i] + cost, i
    cuts, j = [], n
    while j > 0: cuts.append((prev[j], j)); j = prev[j]
    return cuts[::-1]
def para(x, y, width, text, size, align="justify", draw=True, fill="#000000"):
    """Wrap to `width` (balanced breaks, may break after hyphens), justify all but each
    paragraph's last line; a line needing gaps wider than 3 spaces falls back to
    left-aligned. '\\n' forces a break. Returns the line count."""
    sp = measure(" ", "n") * size
    lines = []                                      # (fragments, is_last_of_paragraph)
    for chunk in text.split("\n"):
        F = _frags(chunk, size)
        assert all(fr[1] <= width + 0.5 for fr in F), f"a word is wider than {width}px: {chunk[:40]}"
        cuts = _breaks(F, sp, width)
        for k, (i, j) in enumerate(cuts): lines.append((F[i:j], k == len(cuts) - 1))
    if draw:
        for li, (L, last) in enumerate(lines):
            tot = sum(fr[1] for fr in L)
            ngaps = sum(1 for fr in L[1:] if fr[2])
            gap = (width - tot) / ngaps if ngaps else 0
            if align == "justify" and not last and ngaps and gap > 3 * sp:
                RAGGED.append(" ".join("".join(t for t, _ in fr[0]) for fr in L))
            if not (align == "justify" and not last and ngaps and gap <= 3 * sp):
                gap = sp; lw = tot + sp * ngaps
                cx = x + (width - lw) / 2 if align == "center" else x
            else:
                cx = x
            by = y + li * size * LEAD2
            for k, (pcs, wd, spaced) in enumerate(L):
                if k and spaced: cx += gap
                for t, code in pcs:
                    pw = _pw(t, code, size)
                    if code == "arrow":
                        f.arrow(f"e-glyph{len(f.out)}", [(cx + 1, by - size * 0.32), (cx + pw - 1, by - size * 0.32)], sw=1.6)
                        cx += pw; continue
                    if code:
                        chip(f, cx, by, t, size, fill)
                    else:
                        f.text(cx, by, t, size, "n", anchor="start", fill=fill)
                    cx += pw
    return len(lines)

# ================================================================ geometry ===
BW, BH, S = 210, 58, 90                           # block size, staircase step
L1, L2, L3, LM = 480, 580, 680, 780               # level rows; mid-block row
X0 = 150
def blk(i, row, x=None): x = X0 + i * S if x is None else x; return (x, row - BH / 2, x + BW, row + BH / 2)
D320, D640, D1280 = blk(0, L1), blk(1, L2), blk(2, L3)
CX, CR = D1280[2] + 46, 14                        # concat column
KX0, KW = CX + CR + 10, 84
ROWS = {1: L1, 2: L2, 3: L3}
KBOX = {i: (KX0, y - 14, KX0 + KW, y + 14) for i, y in ROWS.items()}
UX = KBOX[3][2] + 38
U1280, U640, U320 = blk(0, L3, UX), blk(0, L2, UX + S), blk(0, L1, UX + 2 * S)
U320 = (U320[0], U320[1], U320[0] + 170, U320[3])       # narrower: longer conv_out -> pred run
MID = (D1280[2] - 34, LM - BH / 2, U1280[0] + 16, LM + BH / 2)

CONVIN  = (26, L1 - 24, 122, L1 + 24)
LATENT  = (14, L1 - 150, 178, L1 - 80)
CONVOUT = (U320[2] + 28, L1 - 24, U320[2] + 124, L1 + 24)
PRED    = (1250, L1 - 45, 1362, L1 + 45)
TSTEP, SIZE, TEMB = (20, 28, 180, 72), (20, 88, 180, 132), (230, 28, 460, 132)
CLIP, BRK = (34, 172, 196, 254), (22, 160, 208, 290)

# ================================================================== blocks ===
BLOCKS = [("box-down1", D320, C["down"], "DownBlock2D", "(320 ch)"),
          ("box-down2", D640, C["attn"], "CrossAttnDownBlock2D", "(640 ch)"),
          ("box-down3", D1280, C["deep"], "CrossAttnDownBlock2D", "(1280 ch)"),
          ("box-up3", U1280, C["deep"], "CrossAttnUpBlock2D", "(1280 ch)"),
          ("box-up2", U640, C["attn"], "CrossAttnUpBlock2D", "(640 ch)"),
          ("box-up1", U320, C["down"], "UpBlock2D", "(320 ch)")]
b_blk = group_base([(BW, BH, 10, 6, [(m, 1.0, "n"), (s, SUB, "n")]) for *_, m, s in BLOCKS], cap=17.0)
for ident, r, fill, m, s in BLOCKS:
    rbox(ident, r, fill, [(m, 1.0, "n"), (s, SUB, "n")], b_blk)
rbox("box-mid", MID, C["mid"], [("Mid block", 1.0, "n"), ("cross-attn, 1280 ch", SUB, "n")], b_blk)

IO = [("box-latent", LATENT, C["latent"], [("Noisy VAE latent,", 1.0, "n"), ("4 × 128 × 128", SUB, "n")]),
      ("box-convin", CONVIN, C["conv"], [("conv_in", 1.0, "n")]),
      ("box-convout", CONVOUT, C["conv"], [("conv_out", 1.0, "n")]),
      ("box-prednoise", PRED, C["latent"], [("Predicted", 1.0, "n"), ("noise", 1.0, "n"), ("(epsilon-hat)", SUB, "n")])]
b_io = group_base([(r[2] - r[0], r[3] - r[1], 9, 5, l) for _, r, _, l in IO], cap=17.6)
for ident, r, fill, lines in IO: rbox(ident, r, fill, lines, b_io, padx=9, pady=5)
f.text((CONVIN[0] + CONVIN[2]) / 2, CONVIN[3] + 22, "FP16", 14)
f.text((CONVOUT[0] + CONVOUT[2]) / 2, CONVOUT[3] + 22, "FP16", 14)

TOPB = [("box-tstep", TSTEP, C["timestep"], [("Timestep t", 1.0, "n")]),
        ("box-size", SIZE, C["timestep"], [("Size / crop ids", 1.0, "n")]),
        ("box-temb", TEMB, C["latent"], [("Time + added", 1.0, "n"), ("embedding", 1.0, "n"),
                                         ("(t, pooled text, size/crop)", SUB, "n")]),
        ("box-clip", CLIP, C["latent"], [("CLIP text", 1.0, "n"), ("embedding", 1.0, "n"), ("(2048-d)", SUB, "n")])]
b_top = group_base([(r[2] - r[0], r[3] - r[1], 10, 6, l) for _, r, _, l in TOPB], cap=18.0)
for ident, r, fill, lines in TOPB: rbox(ident, r, fill, lines, b_top)

# text encoders stay FP16: bracketed exactly like Figure 4's adaln_single
f.emit(f'  <rect id="bracket-text" x="{BRK[0]}" y="{BRK[1]}" width="{BRK[2] - BRK[0]}" height="{BRK[3] - BRK[1]}" '
       f'rx="10" fill="none" stroke="#7A7A7A" stroke-width="1.5" stroke-dasharray="6 4"/>')
f.text((BRK[0] + BRK[2]) / 2, BRK[3] - 11, "text encoders  (FP16)", 13, "i", fill=LBL)

# ============================================================ concat + conv1 ==
for i, cy in ROWS.items():
    f.circle(f"cat{i}", CX, cy, CR, CAT, "c", 16)
    x0, y0, x1, y1 = KBOX[i]
    f.emit(f'  <rect id="box-conv1-{i}" x="{x0}" y="{y0}" width="{KW}" height="{y1 - y0}" rx="6" '
           f'fill="{BRICK}" stroke="{INK}" stroke-width="{SW_BOX}"/>')
    f.text((x0 + x1) / 2, cy + 5.2, "conv1 ×3", min(15, wfit("conv1 ×3", "n", KW - 10)), "n", fill="#FFFFFF")

# ================================================================ dataflow ===
DATA = []                                          # (ident, pts) of every black dataflow edge
def A(ident, pts, color=INK, sw=None):
    f.arrow(ident, pts, color=color) if sw is None else f.arrow(ident, pts, color=color, sw=sw)
    if color == INK: DATA.append((ident, pts))
A("e-latent-convin", [(74, LATENT[3]), (74, CONVIN[1])])
A("e-convin-down1", [(CONVIN[2], L1), (D320[0], L1)])
A("e-up1-convout", [(U320[2], L1), (CONVOUT[0], L1)])
A("e-convout-pred", [(CONVOUT[2], L1), (PRED[0], L1)])
A("e-down1-down2", [(300, D320[3]), (300, D640[1])])
A("e-down2-down3", [(395, D640[3]), (395, D1280[1])])
A("e-down3-mid", [(420, D1280[3]), (420, LM), (MID[0], LM)])
for i, (src, dst) in {1: (D320, U320), 2: (D640, U640), 3: (D1280, U1280)}.items():
    cy = ROWS[i]
    A(f"e-skip{i}-cat", [(src[2], cy), (CX - CR, cy)])                        # encoder skip
    f.emit(f'  <path id="e-cat{i}-conv1" d="M {CX + CR} {cy} L {KX0} {cy}" fill="none" stroke="{INK}" stroke-width="3.4"/>')
    DATA.append((f"e-cat{i}-conv1", [(CX + CR, cy), (KX0, cy)]))
    A(f"e-conv1{i}-dec", [(KBOX[i][2], cy), (dst[0], cy)])
# conv_in's output is a skip too: the third one at the top level (up_blocks.2.resnets.2
# reads 320 = 320 stream + 320 skip). Tapped off conv_in -> DownBlock2D, run above DownBlock2D.
JX, SK0_Y = CONVIN[2] + 14, D320[1] - 11
SKIP0 = [(JX, L1), (JX, SK0_Y), (CX, SK0_Y), (CX, L1 - CR)]
f.arrow("e-convin-cat1", SKIP0)
f.emit(f'  <circle cx="{JX}" cy="{L1}" r="3.6" fill="{INK}"/>')
DATA.append(("e-convin-cat1", [(CONVIN[2], L1), (CX, L1 - CR)]))
# decoder stream into each concat, from below (torch.cat([hidden_states, res_hidden_states]))
RISERS = {"e-mid-cat3": [(CX, MID[1]), (CX, L3 + CR)],
          "e-up3-cat2": [(747, U1280[1]), (747, 630), (CX, 630), (CX, L2 + CR)],
          "e-up2-cat1": [(837, U640[1]), (837, 530), (CX, 530), (CX, L1 + CR)]}
for k, pts in RISERS.items(): A(k, pts)

# ======================================= timestep-conditioning bus (Fig 4 style) =
TB_Y = 300
TEMB_T = {"box-down1": (340, D320[1]), "box-down2": (430, D640[1]), "box-down3": (520, D1280[1]),
          "box-mid": (556, MID[1]), "box-up3": (800, U1280[1]), "box-up2": (890, U640[1]), "box-up1": (940, U320[1])}
T_LAST = max(x for x, _ in TEMB_T.values())
wire("e-temb-trunk", [(340, TEMB[3]), (340, TB_Y), (T_LAST, TB_Y)], color=BRICK, op=BUS_OP)
for k, (x, y) in TEMB_T.items():
    if x != T_LAST: dot(x, TB_Y, BRICK, BUS_OP)
    busarrow(f"e-cond-{k}", [(x, TB_Y), (x, y)], BRICK, "ahBrickT")
f.text(633, TB_Y - 8, "timestep conditioning", 13, "i", fill=LBL)
A("e-tstep-temb", [(TSTEP[2], 50), (TEMB[0], 50)])
A("e-size-temb", [(SIZE[2], 110), (TEMB[0], 110)])
A("e-pooled-temb", [(BRK[2], 185), (250, 185), (250, TEMB[3])], color=GRAY, sw=1.8)
f.text(256, 160, "pooled text", 13, "i", anchor="start", fill=LBL)

# ============================================ cross-attention K/V bus (Fig 4 style)
KV_Y = 220
KV_T = {"box-down2": (385, D640[1]), "box-down3": (475, D1280[1]), "box-mid": (713, MID[1]),
        "box-up3": (770, U1280[1]), "box-up2": (860, U640[1])}
K_LAST = max(x for x, _ in KV_T.values())
wire("e-kv-trunk", [(BRK[2], KV_Y), (K_LAST, KV_Y)], color=ROYAL, op=BUS_OP)
for k, (x, y) in KV_T.items():
    if x != K_LAST: dot(x, KV_Y, ROYAL, BUS_OP)
    busarrow(f"e-kv-{k}", [(x, KV_Y), (x, y)], ROYAL, "ahRoyalT")
f.text(BRK[2] + 10, KV_Y - 8, "K/V", 13, "i", anchor="start", fill=LBL)

# ======================================================= activation callout ==
# Shares of the 372 layers per format and bucket, from the shipped SDXL LUTs
# (ablations/_calibration_cache/sdxl_w{4,8}a8_lut.pt; W4A8 and W8A8 agree to 0.3 pt).
# Bucket 0 first (dynamic_fake_quant.py:107, T1 = low t). The missing 2.4% per
# bucket is MXFP8: the nine shortcut conv1 layers pinned by topology.
SHARE = [(10, 50, 37), (12, 48, 37), (12, 49, 37), (0, 61, 36), (0, 62, 35),
         (0, 62, 36), (0, 0, 98), (0, 0, 98), (0, 0, 98), (0, 0, 98)]
AX, AY, AW, AH = 958, 20, 404, 400
f.panel("callout-act", AX, AY, AW, AH)
f.text(AX + AW / 2, AY + 32, "A₈ Activation Quantization (Runtime)",
       min(17, wfit("A₈ Activation Quantization (Runtime)", "b", AW - 20)), "b")
f.text(AX + 20, AY + 62, "Macro-routing, static scales", 16, "b", anchor="start")
f.text(AX + 20, AY + 82, "share of the 372 layers per format, per bucket",
       min(13.5, wfit("share of the 372 layers per format, per bucket", "n", AW - 40)), anchor="start")
TY, RH = AY + 140, 19.0
GBX, GBW, BKX, BKW, NX = AX + 42, 14, AX + 66, 52, AX + 160
NW = (AX + AW - 12 - NX) / 3
HX = [NX + NW * (j + 0.5) for j in range(3)]
XM = (BKX + NX + 3 * NW) / 2                      # centre of the table
TH1, TH2 = "Timestep Bucket", "Format Code"
f.text(XM - 14, AY + 110, TH1, 14.5, "b", anchor="end")
f.arrow("e-hdr", [(XM - 9, AY + 105), (XM + 9, AY + 105)], sw=2.2)
f.text(XM + 14, AY + 110, TH2, 14.5, "b", anchor="start")
assert XM - 14 - measure(TH1, "b") * 14.5 >= AX + 12 and XM + 14 + measure(TH2, "b") * 14.5 <= AX + AW - 12, "header overflows"
f.text(BKX + BKW / 2, TY - 8, "Bucket", 13, "b")
for x, t in zip(HX, ("INT8", "E4M3", "E5M2")): f.text(x, TY - 8, t, 13, "b")
rs = min(15.0, RH / 1.26)
for i, row in enumerate(SHARE):
    y = TY + i * RH
    f.rect(BKX, y, BKW, RH, SWEEP[i], INK, 1.0)
    f.text(BKX + BKW / 2, y + RH * 0.72, f"T{i + 1}", rs, "b")
    f.arrow(f"e-row{i}", [(BKX + BKW + 4, y + RH / 2), (NX - 4, y + RH / 2)], sw=2.2)
    for j, v in enumerate(row):
        f.rect(NX + j * NW, y, NW, RH, FMT_CELL, INK, 1.0)
        f.text(HX[j], y + RH * 0.72, f"{v}%" if v else "–", rs, "b" if v == max(row) else "n")
BOT = TY + RH * 10
f.rect(GBX, TY + 15, GBW, BOT - TY - 15, "url(#schedGrad)", INK, 1.0, rx=2)   # arrow UP: denoising
f.emit(f'  <polygon points="{GBX - 4},{TY + 16} {GBX + GBW + 4},{TY + 16} {GBX + GBW / 2},{TY}" '
       f'fill="{SWEEP[0]}" stroke="{INK}" stroke-width="1.0"/>')
f.text(AX + 14, TY + RH * 2.4, "clean signal", 13, rotate=-90)
f.text(AX + 29, TY + RH * 2.4, "(low t)", 12, rotate=-90, fill="#444444")
f.text(AX + 14, TY + RH * 7.6, "pure noise", 13, rotate=-90)
f.text(AX + 29, TY + RH * 7.6, "(high t)", 12, rotate=-90, fill="#444444")
foot = ["MXFP8: the 9 shortcut `conv1` (2.4%) in every bucket",
        "per-bucket scales, calibrated ahead of time",
        "and static at inference (Q-Diffusion style)"]
fs = 13.0
while any(para(AX + 12, 0, AW - 24, t, fs, draw=False) > 1 for t in foot): fs -= 0.25
assert fs >= 12, f"activation footer shrank to {fs:.1f}"
for i, t in enumerate(foot): para(AX + 12, BOT + 22 + i * 17, AW - 24, t, fs, align="center", fill="#333333")
assert BOT + 22 + 2 * 17 + 5 < AY + AH - 4, "activation-callout footer overflows"

# ======================================================= topology callout ===
TX0, TY0, TW0 = 958, 625, 404
TP0 = "`up_blocks.*.resnets.*.conv1`\n→ MXFP8 E4M3, unconditionally, bypassing the κ/SNR check; ×3 per level, one per up-block ResNet."
TP = [TP0,
      "The concatenated input spans a peak-to-median channel range of 58× at the median "
      "layer and up to 163× (~1.4× an ordinary conv)."]
ts = 14.0
tl = sum(para(TX0 + 13, 0, TW0 - 26, t, ts, draw=False) for t in TP)
TH0 = 62 + (tl - 1) * ts * LEAD2 + (len(TP) - 1) * PGAP + 18
f.panel("callout-topology", TX0, TY0, TW0, TH0)
assert TY0 + TH0 + 12 <= 840, "topology panel runs into the bottom band"
f.emit(f'  <rect x="{TX0 + 14}" y="{TY0 + 14}" width="54" height="22" rx="5" fill="{BRICK}" stroke="{INK}" stroke-width="1"/>')
f.text(TX0 + 41, TY0 + 30, "conv1", 13, fill="#FFFFFF")
tt = "Shortcut Topology Rule"
f.text(TX0 + 78, TY0 + 31, tt, min(17, wfit(tt, "b", TW0 - 90)), "b", anchor="start")
yy = TY0 + 62
for t in TP: yy += para(TX0 + 13, yy, TW0 - 26, t, ts) * ts * LEAD2 + PGAP

# ==================================================== bottom band: callouts ==
BY, BHH = 840, 226
CP1 = "Channel-wise `torch.cat` of the decoder stream (from below) with one encoder skip, before each of the three up-block ResNets."
CP2 = "Per level, the skips are the two ResNet outputs of the same-level encoder block and the downsampled output of the level above (`conv_in`'s output at the top)."
CP3 = "Both `conv1` (pinned to MXFP8) and the parallel 1×1 `conv_shortcut` (routed by κ/SNR) read its output."
CP = [CP1, CP2, CP3]
_CW = 1362 - (14 + 560 + 18 + 460 + 18) - 26
_need = 62 + (sum(para(0, 0, _CW, t, 14.0, draw=False) for t in CP) - 1) * 14.0 * LEAD2 + (len(CP) - 1) * PGAP + 18
BHH = max(BHH, int(_need) + 1)
WX, WW = 14, 560
f.panel("callout-weight", WX, BY, WW, BHH)
f.text(WX + WW / 2, BY + 32, "W₄/W₈ Weight Quantization (Offline)", 18, "b")
f.text(WX + WW / 2, BY + 56, "Per-output-channel format selection.", 15)
SX, SY, SWD, BH_ = WX + 52, BY + 80, 80, 12
f.text(SX + SWD / 2, SY - 7, "input channel", 13)
f.text(WX + 34, SY + 4 * BH_, "output channel", 13, rotate=-90)
for i, col in enumerate(STRIP): f.rect(SX, SY + i * BH_, SWD, BH_, col, INK, 0.8)
yb0, yb1, xb, ym = SY, SY + 8 * BH_, SX + SWD + 10, SY + 4 * BH_
f.emit(f'  <path d="M {xb} {yb0} Q {xb + 8} {yb0} {xb + 8} {yb0 + 11} L {xb + 8} {ym - 9} Q {xb + 8} {ym} {xb + 16} {ym} '
       f'Q {xb + 8} {ym} {xb + 8} {ym + 9} L {xb + 8} {yb1 - 11} Q {xb + 8} {yb1} {xb} {yb1}" '
       f'fill="none" stroke="{INK}" stroke-width="1.6"/>')
PX = xb + 28; PW = WX + WW - 16 - PX
w8, w4a, w4b = "W8: {INT8-sym, INT8-asym, MXINT8}", "W4: {INT4-sym, INT4-asym, NF4,", "FP4 E2M1, MXINT4, MXFP4 E2M1}"
ps = min(14.0, wfit(w8, "n", PW - 16), wfit(w4b, "n", PW - 16))
f.rect(PX, SY - 4, PW, 32, "#EDEDED", "#C8C8C8", 1, rx=6); f.text(PX + PW / 2, SY + 17, w8, ps)
f.rect(PX, SY + 38, PW, 54, "#EDEDED", "#C8C8C8", 1, rx=6)
f.text(PX + PW / 2, SY + 60, w4a, ps); f.text(PX + PW / 2, SY + 80, w4b, ps)
WF = "372 quantized conv and linear layers; attention Q/K/V and `conv_in`/`conv_out` stay FP16."
wfs = 13.0
nwf = para(WX + 12, 0, WW - 24, WF, wfs, align="center", draw=False)
assert nwf <= 2, f"weight footer needs {nwf} lines"
para(WX + 12, yb1 + 22, WW - 24, WF, wfs, align="center", fill="#333333")
assert yb1 + 22 + (nwf - 1) * wfs * LEAD2 + 5 < BY + BHH - 4, "weight-callout footer overflows"

FX0 = WX + WW + 18; FW0 = 460
f.panel("callout-fewstep", FX0, BY, FW0, BHH)
f.text(FX0 + FW0 / 2, BY + 32, "Few-Step Variant (SDXL-Turbo)", 18, "b")
BUL5 = "concat: 9 up-block `conv_shortcut` layers stay FP16; no MXFP8 pin"
BUL = ["same UNet topology; 1 denoising step at 512 × 512",
       "weights: Chameleon adaptive palette",
       "activations: MixDQ static scales + mixed precision",
       "MixDQ's BOS-aware bypass dropped",
       BUL5]
bs = 15.0
while any(para(FX0 + 22, 0, FW0 - 40, "• " + t, bs, align="left", draw=False) > 1 for t in BUL): bs -= 0.25
assert bs >= 13, f"few-step bullets shrank to {bs:.1f}"
for i, t in enumerate(BUL): para(FX0 + 22, BY + 70 + i * 31, FW0 - 40, "• " + t, bs, align="left")
assert BY + 70 + 4 * 31 + 8 < BY + BHH - 4, "few-step bullets overflow"
CX0 = FX0 + FW0 + 18; CW0 = 1362 - CX0
f.panel("callout-concat", CX0, BY, CW0, BHH)
f.circle("legend-cat", CX0 + 28, BY + 27, CR, CAT, "c", 16)
ct = "Skip Concatenation"
f.text(CX0 + 52, BY + 33, ct, min(17, wfit(ct, "b", CW0 - 64)), "b", anchor="start")
cs = 14.0
yy = BY + 62
for t in CP: yy += para(CX0 + 13, yy, CW0 - 26, t, cs) * cs * LEAD2 + PGAP
assert yy - cs * LEAD2 - PGAP + 6 < BY + BHH - 4, "concat text overflows"
f.h = max(H, int(BY + BHH + 14))                 # canvas grows with the bottom band

# =================================================================== checks ==
BOXES = {"box-down1": D320, "box-down2": D640, "box-down3": D1280, "box-up1": U320, "box-up2": U640,
         "box-up3": U1280, "box-mid": MID, "box-latent": LATENT, "box-convin": CONVIN, "box-convout": CONVOUT,
         "box-prednoise": PRED, "box-tstep": TSTEP, "box-size": SIZE, "box-temb": TEMB, "box-clip": CLIP,
         **{f"cat{i}": (CX - CR, y - CR, CX + CR, y + CR) for i, y in ROWS.items()},
         **{f"conv1-{i}": b for i, b in KBOX.items()},
         "callout-act": (AX, AY, AX + AW, AY + AH), "callout-topology": (TX0, TY0, TX0 + TW0, TY0 + TH0),
         "callout-weight": (WX, BY, WX + WW, BY + BHH), "callout-fewstep": (FX0, BY, FX0 + FW0, BY + BHH),
         "callout-concat": (CX0, BY, CX0 + CW0, BY + BHH)}
def seg_rect(p, q, r, m=4):
    x0, y0, x1, y1 = r[0] - m, r[1] - m, r[2] + m, r[3] + m
    t0, t1, dx, dy = 0.0, 1.0, q[0] - p[0], q[1] - p[1]
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0: return False
        else:
            t = qq / pp
            if pp < 0: t0 = max(t0, t)
            else: t1 = min(t1, t)
            if t0 > t1: return False
    return True
def shrink(p, q, d=10):
    L = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** .5 or 1
    return ((p[0] + (q[0] - p[0]) * d / L, p[1] + (q[1] - p[1]) * d / L),
            (q[0] - (q[0] - p[0]) * d / L, q[1] - (q[1] - p[1]) * d / L))
bad = []
def check_path(name, pts, allowed):
    for p, q in zip(pts, pts[1:]):
        a, b = shrink(p, q)
        for k, r in BOXES.items():
            if k not in allowed and seg_rect(a, b, r): bad.append(f"{name} crosses {k}")
for k, (x, y) in TEMB_T.items(): check_path(f"cond->{k}", [(x, TB_Y), (x, y)], {k})
for k, (x, y) in KV_T.items():   check_path(f"kv->{k}", [(x, KV_Y), (x, y)], {k})
check_path("temb trunk", [(340, TEMB[3]), (340, TB_Y), (T_LAST, TB_Y)], {"box-temb"})
check_path("kv trunk", [(BRK[2], KV_Y), (K_LAST, KV_Y)], {"box-clip"})
check_path("pooled", [(BRK[2], 185), (250, 185), (250, TEMB[3])], {"box-clip", "box-temb"})
check_path("e-mid-cat3", RISERS["e-mid-cat3"], {"box-mid", "cat3"})
check_path("e-convin-cat1", SKIP0, {"box-convin", "cat1"})
check_path("e-up3-cat2", RISERS["e-up3-cat2"], {"box-up3", "cat2"})
check_path("e-up2-cat1", RISERS["e-up2-cat1"], {"box-up2", "cat1"})
check_path("e-down3-mid", [(420, D1280[3]), (420, LM), (MID[0], LM)], {"box-down3", "box-mid"})
keys = list(BOXES)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = BOXES[keys[i]], BOXES[keys[j]]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]: bad.append(f"overlap {keys[i]} / {keys[j]}")
for k, r in BOXES.items():
    if k != "box-clip" and r[0] < BRK[2] and BRK[0] < r[2] and r[1] < BRK[3] and BRK[1] < r[3]:
        bad.append(f"text-encoder bracket overlaps {k}")
for name, gap in (("convin->down1", D320[0] - CONVIN[2]), ("down3->cat3", CX - CR - D1280[2]),
                  ("conv1->up3", U1280[0] - KBOX[3][2]), ("up1->convout", CONVOUT[0] - U320[2]),
                  ("convout->pred", PRED[0] - CONVOUT[2])):
    if gap < 18: bad.append(f"arrow {name} only {gap}px")
NODES = {k: r for k, r in BOXES.items() if k.startswith(("box-", "cat", "conv1-"))}
def at(p):
    hit = [k for k, r in NODES.items() if r[0] - 2 <= p[0] <= r[2] + 2 and r[1] - 2 <= p[1] <= r[3] + 2]
    assert len(hit) == 1, f"edge endpoint {p} touches {hit}"
    return hit[0]
EDGES = {(at(pts[0]), at(pts[-1])) for _, pts in DATA}
FORBID = {("box-mid", "box-up2"), ("box-up2", "box-up3"), ("box-up3", "box-up1")}
for e in EDGES & FORBID: bad.append(f"forbidden edge {e[0]} -> {e[1]}")
def reach(a, b, seen=None):
    seen = seen or set()
    if a == b: return True
    seen.add(a)
    return any(reach(y, b, seen) for x, y in EDGES if x == a and y not in seen)
for a, b in (("box-down1", "box-down2"), ("box-down2", "box-down3"), ("box-down3", "box-mid"),
             ("box-mid", "box-up3"), ("box-up3", "box-up2"), ("box-up2", "box-up1"), ("box-up1", "box-prednoise")):
    if not reach(a, b): bad.append(f"no path {a} => {b}")
for a, b in (("box-up2", "box-up3"), ("box-up1", "box-up2"), ("box-up3", "box-mid")):
    if reach(a, b): bad.append(f"backward path {a} => {b}")
bad += [f"ragged justified line: {r}" for r in RAGGED]
assert not bad, "\n".join(bad)
print(f"{len(EDGES)} dataflow edges; decoder mid->up3->up2->up1 via c/conv1; forbidden edges absent")
print(f"block text {b_blk:.1f}px, io {b_io:.1f}px, top {b_top:.1f}px, topology {ts:.1f}px, bullets {bs:.1f}px")
f.save("images/fig3_unet_schematic.svg")
