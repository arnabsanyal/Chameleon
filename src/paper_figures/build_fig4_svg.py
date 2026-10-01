#!/usr/bin/env python3
"""Figure 4 -- PixArt-alpha DiT execution path, drawn as clean editable SVG.

Wiring is literal: tokens enter the block at the top, run down the residual
spine, and leave from the last (+) into Final LayerNorm.

Verified against diffusers (pixart_transformer_2d.py / attention.py) and
PixArt-alpha (2310.00426) / Q-DiT (2406.17343):
  * adaln_single runs ONCE before `for block in self.transformer_blocks`, so it
    sits outside the x28 container. Its 6*dim output is summed with each block's
    scale_shift_table and chunked into shift/scale/gate -- it feeds the two
    modulated LayerNorms and the two output gates, NOT the residual stream.
  * per block: norm1 -> modulate -> attn1 -> x gate_msa -> (+)
               norm2 -> modulate -> ff    -> x gate_mlp -> (+)
  * skips carry the sublayer INPUT (before its norm) to the (+).
  * bucket order follows dynamic_fake_quant.py:107 -- bucket 0 (T1) is the
    cleanest signal and takes INT8; bucket B-1 (T10) is pure noise -> FP8 E5M2.

Regenerate: python3 build_fig4_svg.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import Fig, C, INK, GRAY, group_base, wfit, rich

ROYAL, DOTS = "#4169E1", "1 6"

W, H = 1376, 960
f = Fig(W, H)
SUB, MLP = 0.80, "#B9C8E8"
SWEEP = ["#F1B695", "#F8C796", "#FADA90", "#F5E39B", "#BBD1AA",
         "#B7CDA8", "#98C4C7", "#8EB1CE", "#A9ADD2", "#AC99C7"]
FMT_CELL = "#D4D4D4"
f.defs.append(f'    <marker id="ahRoyal" viewBox="0 0 10 10" refX="10" refY="5" '
              f'markerWidth="4.2" markerHeight="4.2" markerUnits="strokeWidth" '
              f'orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ROYAL}"/></marker>')
f.defs.append('    <linearGradient id="schedGrad" x1="0" y1="0" x2="0" y2="1">' +
              "".join(f'<stop offset="{i/9:.3f}" stop-color="{c}"/>'
                      for i, c in enumerate(SWEEP)) + '</linearGradient>')

# ---------------------------------------------------------- DiT container ---
DX, DW, DY, DH = 520, 250, 150, 686
SPINE, IX, IW, R = DX + DW / 2, DX + 34, DW - 68, 13
SKIP_X, BUS_X, RAIL_X = 534, 756, 810          # three clear vertical corridors
# attn_output = gate * attn_output ; hidden = attn_output + hidden
# -> the gate is a MULTIPLY on the sublayer output, upstream of the single
# residual add. Three adds and two mults per block, as in the code.
NORM1, ATT1 = (200, 30), (268, 58)
GATE1_Y, ADD1 = 362, 411
ATT2, ADD2 = (452, 58), 546
NORM2, MLP_ = (588, 30), (656, 58)
GATE2_Y, ADD3 = 750, 799
ATT2_MID = ATT2[0] + ATT2[1] // 2      # centre of Cross-Attention's left edge

# two offset cards behind the container: the x28 label alone read as an
# orphan, the stack makes the repetition visible
CARDS = 2
for k in range(CARDS, 0, -1):
    f.emit(f'  <rect x="{DX + 7 * k}" y="{DY - 7 * k}" width="{DW}" height="{DH}" rx="14" '
           f'fill="#EFEFEF" stroke="{INK}" stroke-width="1.7"/>')
STACK_R = DX + 7 * CARDS + DW          # outer right edge of the card stack
# The input arrow lands on the FRONT card's top edge, so its solid head would
# otherwise print on top of the back cards' top borders. Break those borders
# where the arrow passes, the way a line-hop does.
f.emit(f'  <rect x="{SPINE - 8}" y="{DY - 7 * CARDS + 1}" width="16" '
       f'height="{7 * CARDS - 2}" fill="#EFEFEF"/>')
f.emit(f'  <rect id="dit-block" x="{DX}" y="{DY}" width="{DW}" height="{DH}" rx="14" '
       f'fill="#EFEFEF" stroke="{INK}" stroke-width="1.7"/>')
f.text(DX + DW - 14, DY + 24, "DiT Block", 19, "b", anchor="end")
# tucked against the card stack, in the gap between Timestep embedding
# (ends y=140) and AdaLN-single(t) (starts y=193)
f.text(DX + DW + 20, DY + 25, "× 28", 22, "b", anchor="start")
assert DX + DW + 20 > STACK_R, "x28 overlaps the card stack"

SUBS = [("box-norm1", "LayerNorm + modulate", None, "#DDE5F0", NORM1),
        ("box-attn1", "Self-Attention", "(attn1)", C["attn"], ATT1),
        ("box-attn2", "Cross-Attention", "(attn2)", C["down"], ATT2),
        ("box-norm2", "LayerNorm + modulate", None, "#DDE5F0", NORM2),
        ("box-mlp", "Feed-Forward", "(MLP)", MLP, MLP_)]
b_in = group_base([(IW, h, 10, 5,
                    [(m, 1.0, "n")] if s is None else [(m, 1.0, "n"), (s, SUB, "n")])
                   for _, m, s, _, (y, h) in SUBS], cap=18)
for ident, main, sub, fill, (y, h) in SUBS:
    ls = [(main, 1.0, "n")] if sub is None else [(main, 1.0, "n"), (sub, SUB, "n")]
    f.box(ident, IX, y, IW, h, fill, ls, b_in, padx=10)
for nm, cy in (("add1", ADD1), ("add2", ADD2), ("add3", ADD3)):
    f.circle(nm, SPINE, cy, R, "#ffffff", "+", 16)

for a, b in [(DY, NORM1[0]), (NORM1[0] + NORM1[1], ATT1[0]),
             (ATT1[0] + ATT1[1], GATE1_Y - R), (GATE1_Y + R, ADD1 - R),
             (ADD1 + R, ATT2[0]), (ATT2[0] + ATT2[1], ADD2 - R),
             (ADD2 + R, NORM2[0]), (NORM2[0] + NORM2[1], MLP_[0]),
             (MLP_[0] + MLP_[1], GATE2_Y - R), (GATE2_Y + R, ADD3 - R),
             (ADD3 + R, DY + DH)]:
    assert b - a >= 18, f"spine segment {a}->{b} too short"
    f.arrow(f"e-spine-{a}", [(SPINE, a), (SPINE, b)], sw=2.8)


def bracket(ident, y0, y1, x_turn, x_end, rad=11, sw=2.2, color=INK):
    """Half rounded-rectangle: out, along, back in. Never bulges over a box."""
    d = (f"M {SPINE} {y0} L {x_turn + rad} {y0} Q {x_turn} {y0} {x_turn} {y0 + rad} "
         f"L {x_turn} {y1 - rad} Q {x_turn} {y1} {x_turn + rad} {y1} L {x_end} {y1}")
    f.emit(f'  <path id="{ident}" d="{d}" fill="none" stroke="{color}" '
           f'stroke-width="{sw}" stroke-linejoin="round" marker-end="url(#ahInk)"/>')


# skips branch off the spine BEFORE each sublayer's norm; the corridor at
# SKIP_X sits left of every box (boxes start at IX), so nothing is jumped over
assert SKIP_X < IX and SKIP_X > DX, "skip corridor is not inside the container"
# each skip branches off the (longer) input arrow feeding its sublayer, well
# behind that arrow's head
for i, (y0, cy) in enumerate(((DY + 20, ADD1), (ADD1 + R + 8, ADD2),
                              (ADD2 + R + 8, ADD3)), start=1):
    bracket(f"e-skip{i}", y0, cy, SKIP_X, SPINE - R)

# ------------------------------- adaLN-single output: where it actually goes --
# One bus down the right corridor feeding the two modulated norms and the two
# output gates. BUS_X sits right of every box and left of the dispatch rail.
assert IX + IW < BUS_X < RAIL_X, "modulation bus collides with boxes or rail"
# The output runs straight left into the first LayerNorm+modulate, with a
# vertical offshoot dropping to the other three consumers. No arrowhead on the
# drop itself -- it is a bus, not a signal into empty space.
MODY, BUS_TOP, BRANCH_X = NORM1[0] + 15, 249, 505
f.emit(f'  <path id="e-mod-feed" d="M 248 324 L {BRANCH_X} 324 L {BRANCH_X} {MODY}" '
       f'fill="none" stroke="{GRAY}" stroke-width="1.8" stroke-linejoin="round"/>')
f.arrow("e-mod-norm1", [(BRANCH_X, MODY), (IX, MODY)], color=GRAY, sw=1.8)
f.emit(f'  <path id="e-mod-branch" d="M {BRANCH_X} {BUS_TOP} L {BUS_X} {BUS_TOP}" '
       f'fill="none" stroke="{GRAY}" stroke-width="1.8"/>')
f.emit(f'  <circle cx="{BRANCH_X}" cy="{BUS_TOP}" r="3.2" fill="{GRAY}"/>')
f.emit(f'  <path id="e-mod-bus" d="M {BUS_X} {BUS_TOP} L {BUS_X} {ADD3}" fill="none" '
       f'stroke="{GRAY}" stroke-width="1.8"/>')
f.arrow("e-mod-norm2", [(BUS_X, NORM2[0] + 15), (IX + IW, NORM2[0] + 15)], color=GRAY, sw=1.8)
# The multiply sits ON the spine: it takes the sublayer output from above and
# the gate from the modulation bus, and feeds the residual add below.
for nm, cy, lbl in (("gate1", GATE1_Y, "× `gate_msa`"), ("gate2", GATE2_Y, "× `gate_mlp`")):
    f.circle(f"op-{nm}", SPINE, cy, R, "#ffffff", "×", 17)
    f.arrow(f"e-mod-{nm}", [(BUS_X, cy), (SPINE + R, cy)], color=GRAY, sw=1.8)
    rich(f, SPINE + 26, cy + 17, lbl, 12.5, "i", anchor="start", fill="#444444")

# ------------------------------------------------------------ token input ---
IN = [("box-latent", 20, 160, [("Noisy VAE latent,", 1.0, "n"), ("4 × 128 × 128", SUB, "n")]),
      ("box-patchify", 208, 114, [("Patchify", 1.0, "n"), ("(patch size 2)", SUB, "n")]),
      ("box-seq", 350, 164, [("Sequence:", 1.0, "n"), ("4096 tokens", 1.0, "n"),
                             ("× 1152 dim", SUB, "n")])]
b_in2 = group_base([(w, 74, 11, 6, l) for _, _, w, l in IN], cap=18)
for ident, x, w, ls in IN:
    f.box(ident, x, 28, w, 74, C["latent"], ls, b_in2, padx=11)
f.arrow("e-lat-pat", [(180, 65), (208, 65)])
f.arrow("e-pat-seq", [(322, 65), (350, 65)])
f.arrow("e-seq-block", [(514, 65), (SPINE, 65), (SPINE, DY)])

# ------------------------------------------------- adaLN-single, OUTSIDE ----
# diffusers' AdaLayerNormSingle IS the embedder + SiLU + Linear, so both boxes
# are one module -- and DEFAULT_SKIP_PATTERNS matches the whole thing. Bracket
# them so "adaln_single stays FP16" clearly covers both.
f.emit(f'  <rect id="bracket-adaln" x="42" y="206" width="206" height="186" rx="10" '
       f'fill="none" stroke="#7A7A7A" stroke-width="1.5" stroke-dasharray="6 4"/>')
rich(f, 145, 380, "`adaln_single`  (FP16)", 13, "i", anchor="middle", fill="#555555")
f.box("box-tstep", 70, 140, 150, 40, C["timestep"], [("Timestep t", 1.0, "n")], 18)
f.box("box-temb", 54, 216, 182, 44, C["latent"], [("Timestep embedding", 1.0, "n")], 15.5)
# the 1024-MS checkpoint has sample_size 128, so diffusers turns on
# use_additional_conditions: resolution and aspect ratio are embedded too
f.box("box-adaln", 54, 296, 182, 56, C["deep"],
      [("AdaLN-single", 1.0, "n"), ("(t, resolution, aspect ratio)", SUB, "n")], 16)
for a, b in ((60, 96), (140, 176)):
    assert b - a >= 30, "conditioning arrow too short"
f.arrow("e-t-temb", [(145, 180), (145, 216)])
f.arrow("e-temb-adaln", [(145, 260), (145, 296)])
f.text(42, 414, "one global MLP, shared by", 12.5, "n", anchor="start", fill="#444444")
f.text(42, 430, "all 28 blocks; summed with", 12.5, "n", anchor="start", fill="#444444")
rich(f, 42, 446, "each block's `scale_shift_table`", 12.5, "n", anchor="start", fill="#444444")

# ------------------------------------------------------------- text side ----
f.box("box-caption", 60, 640, 220, 46, C["conv"], [("Caption projection", 1.0, "n")], 17)
f.box("box-t5", 40, 716, 260, 60, C["latent"],
      [("T5 text embedding", 1.0, "n"), ("(4096-d, 120 tokens)", SUB, "n")], 17)
f.arrow("e-t5-caption", [(170, 716), (170, 686)])
f.arrow("e-caption-attn2", [(280, 663), (430, 663), (430, ATT2_MID), (IX, ATT2_MID)])
f.text(438, ATT2_MID - 11, "K/V", 15, anchor="start")

# ------------------------------------------------------------ token output --
# The output chain runs RIGHT TO LEFT: Final LayerNorm sits under the spine and
# the chain walks back towards the page edge, which frees the space to its right
# for the modulation wire reaching norm_out.
OUT = [("box-lnorm", 505, 230, [("Final LayerNorm", 1.0, "n"), ("+ modulate", 1.0, "n")], "#DDE5F0"),
       ("box-projout", 361, 116, [("Linear", 1.0, "n"), ("(proj_out)", SUB, "n")], C["timestep"]),
       ("box-unpatch", 223, 110, [("Unpatchify", 1.0, "n")], C["latent"]),
       ("box-pred", 79, 116, [("Predicted", 1.0, "n"), ("noise", 1.0, "n"),
                              ("(epsilon-hat)", SUB, "n")], C["latent"])]
b_out = group_base([(w, 74, 11, 6, l) for _, _, w, l, _ in OUT], cap=18)
for ident, x, w, ls, fill in OUT:
    f.box(ident, x, 870, w, 74, fill, ls, b_out, padx=11)
f.arrow("e-add3-out", [(SPINE, DY + DH), (SPINE, 870)], sw=2.8)
assert 870 - (DY + DH) >= 25, "block output arrowhead sits flush with the container"
f.arrow("e-lnorm-proj", [(505, 907), (477, 907)])
f.arrow("e-proj-unpatch", [(361, 907), (333, 907)])
f.arrow("e-unpatch-pred", [(223, 907), (195, 907)])
# norm_out carries its own scale_shift_table (2, dim) and is shifted/scaled by
# adaLN-single before proj_out, so the modulation bus continues down to it
f.emit(f'  <path id="e-mod-out" d="M {BUS_X} {ADD3} L {BUS_X} 907" fill="none" '
       f'stroke="{GRAY}" stroke-width="1.8"/>')
f.arrow("e-mod-lnorm", [(BUS_X, 907), (735, 907)], color=GRAY, sw=1.8)
assert BUS_X > 505 + 230, "modulation wire starts inside the Final LayerNorm box"

# ------------------- ONE callout: activation over weights, no inner borders --
CX, CW, CY, CH = 990, 370, 20, 830
f.emit(f'  <rect id="callout" x="{CX}" y="{CY}" width="{CW}" height="{CH}" rx="13" '
       f'fill="none" stroke="{ROYAL}" stroke-width="3.0" stroke-dasharray="{DOTS}" '
       f'stroke-linecap="round"/>')

f.text(CX + CW / 2, CY + 32, "A₈ Activation Quantization (Runtime)", 17, "b")
f.text(CX + 20, CY + 62, "Macro-routing", 16, "b", anchor="start")
f.text(CX + 20, CY + 82, "format code per (layer, timestep bucket)", 13.5, "n", anchor="start")
TY, RH = CY + 120, 19.0
BKX, BKW, FMX, FMW = CX + 72, 110, CX + 208, 146
AGAP = (BKX + BKW + FMX) / 2
f.text(BKX + BKW, CY + 108, "Timestep Bucket", 14.5, "b", anchor="end")
f.text(FMX, CY + 108, "Format Code", 14.5, "b", anchor="start")
f.arrow("e-hdr", [(AGAP - 9, CY + 104), (AGAP + 9, CY + 104)], sw=2.2)
# bucket 0 first, matching dynamic_fake_quant.py:107
# Measured from the shipped configs (coco_eval_w4a8_10b_iaw.json and the W8A8
# twin): FP8 E5M2 wins 97-100% of the 282 linears in EVERY bucket and INT8-asym
# is never selected. The UNet's three-phase progression does not occur here --
# DiT activations stay heavy-tailed across the whole schedule.
ROWS = [("T1", "FP8 E5M2  98%"), ("T2", "FP8 E5M2  98%"),
        ("T3", "FP8 E5M2  97%"), ("T4", "FP8 E5M2  97%"),
        ("T5", "FP8 E5M2 100%"), ("T6", "FP8 E5M2 100%"),
        ("T7", "FP8 E5M2 100%"), ("T8", "FP8 E5M2 100%"),
        ("T9", "FP8 E5M2 100%"), ("T10", "FP8 E5M2 100%")]
rs = min(15.0, RH / 1.26, wfit("FP8 E5M2 100%", "n", FMW - 14))
GBX, GBW = CX + 42, 14
f.rect(GBX, TY + 15, GBW, RH * 10 - 15, "url(#schedGrad)", INK, 1.0, rx=2)
f.emit(f'  <polygon points="{GBX-4},{TY+16} {GBX+GBW+4},{TY+16} '
       f'{GBX+GBW/2},{TY}" fill="{SWEEP[0]}" stroke="{INK}" stroke-width="1.0"/>')
f.text(CX + 14, TY + RH * 2.4, "clean signal", 13, "n", rotate=-90)
f.text(CX + 29, TY + RH * 2.4, "(low t)", 12, "n", rotate=-90, fill="#444444")
f.text(CX + 14, TY + RH * 7.6, "pure noise", 13, "n", rotate=-90)
f.text(CX + 29, TY + RH * 7.6, "(high t)", 12, "n", rotate=-90, fill="#444444")
for i, (bk, fmt) in enumerate(ROWS):
    y = TY + i * RH
    f.rect(BKX, y, BKW, RH, SWEEP[i], INK, 1.0)
    f.rect(FMX, y, FMW, RH, FMT_CELL, INK, 1.0)
    f.text(BKX + BKW / 2, y + RH * 0.72, bk, rs, "b")
    f.text(FMX + FMW / 2, y + RH * 0.72, fmt, rs, "b")
    f.arrow(f"e-row{i}", [(AGAP - 9, y + RH / 2), (AGAP + 9, y + RH / 2)], sw=2.2)
f.text(CX + CW / 2, TY + RH * 10 + 20, "one LUT read per layer per step", 13, "n", fill="#333333")
f.text(CX + CW / 2, TY + RH * 10 + 38,
       "INT8-asym never selected: unlike the UNet, DiT activations", 12, "n", fill="#333333")
f.text(CX + CW / 2, TY + RH * 10 + 54,
       "stay heavy-tailed at every timestep", 12, "n", fill="#333333")


def rule(y):
    f.emit(f'  <path d="M {CX+22} {y} L {CX+CW-22} {y}" stroke="#CCCCCC" stroke-width="1.1"/>')


rule(TY + RH * 10 + 68)
f.text(CX + 20, TY + RH * 10 + 92, "Micro-scaling", 16, "b", anchor="start")
for i, ln in enumerate(["scale computed dynamically",
                        "from the current tensor's min/max at inference"]):
    f.text(CX + 20, TY + RH * 10 + 114 + i * 18, ln, 13.5, "n", anchor="start")

rule(470)
f.text(CX + CW / 2, 500, "W₄ Weight Quantization (Offline)", 18, "b")
f.text(CX + CW / 2, 522, "Per-linear joint (g, f) search.", 15, "n")
GS, FMTS = ["288", "192", "128", "64", "32"], ["INT4-asym", "NF4", "FP4 E2M1"]
CWD, CHT = 44, 25
GX, GY = CX + 66, 548
ARG = (4, 0)          # g = 32, INT4-asym -- the measured winner
HEAT = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [3, 1, 1]]
assert HEAT[ARG[0]][ARG[1]] == 3, "argmax cell is not the row/column crossing"
SHADE = {0: "#DCE6F2", 1: "#FBE9C4", 3: "#E8922E"}
for r in range(5):
    for c in range(3):
        f.rect(GX + c * CWD, GY + r * CHT, CWD, CHT, SHADE[HEAT[r][c]], INK, 1.0)
    f.text(GX - 8, GY + r * CHT + CHT * 0.68, GS[r], 14, anchor="end")
for c, nm in enumerate(FMTS):
    f.text(GX + c * CWD + CWD / 2, GY + 5 * CHT + 22, nm, 12.5, "n", rotate=-38)
f.text(GX - 40, GY + 2.5 * CHT, "Group size (g)", 13.5, "n", rotate=-90)
f.text(GX + 1.5 * CWD, GY + 5 * CHT + 54, "Format (f)", 14)
f.text(CX + CW / 2, GY + 5 * CHT + 90, "g = 32 for all 282 layers;", 13, "n", fill="#333333")
f.text(CX + CW / 2, GY + 5 * CHT + 108, "INT4-asym 213,  NF4 69,  FP4 E2M1 0", 13, "n", fill="#333333")
AGX, ACY = GX + 3 * CWD, GY + ARG[0] * CHT + CHT / 2
f.arrow("e-argmax", [(AGX + 22, ACY), (GX + (ARG[1] + 1) * CWD + 3, ACY)], sw=2.2)
f.text(AGX + 34, ACY, "argmax", 13, anchor="middle", rotate=-90)
BX, BY = CX + 244, GY - 2
assert AGX + 34 + 8 < BX, "argmax label runs into the grouping strip"
PAL = ["#4670B6", "#70548C", "#9AB68C", "#EEC470", "#E0864E", "#5BA8A0"]
for r in range(8):
    for c in range(8):
        f.rect(BX + c * 14, BY + r * 15, 14, 15, PAL[(r * 5 + c * 3) % 6], "#FFFFFF", 0.6)
for g in range(4):
    f.emit(f'  <rect x="{BX + g * 28:.0f}" y="{BY - 3:.0f}" width="28" height="{8*15+6}" '
           f'rx="3" fill="none" stroke="{INK}" stroke-width="2.0"/>')
f.text(BX + 56, BY + 8 * 15 + 24, "Input-channel", 13.5, "n")
f.text(BX + 56, BY + 8 * 15 + 40, "grouping (along C_in)", 12.5, "n", fill="#444444")
f.text(CX + CW / 2, CY + CH - 46,
       "282 quantized linears (28 × 10 + caption projection);", 13, "n", fill="#333333")
rich(f, CX + CW / 2, CY + CH - 28, "`proj_out` and `adaln_single` stay FP16.", 13, "n", anchor="middle", fill="#333333")
assert CY + CH - 28 + 6 < CY + CH, "callout footer overflows"

# --------------------------------- one leader for the whole callout ---------
LEADY = CY + CH / 2
# stop on the outer edge of the card stack rather than running over it
f.emit(f'  <path id="e-callout-block" d="M {CX} {LEADY} L {STACK_R} {LEADY}" fill="none" '
       f'stroke="{ROYAL}" stroke-width="3.0" stroke-dasharray="{DOTS}" '
       f'stroke-linecap="round" marker-end="url(#ahRoyal)"/>')

f.save("images/fig4_dit_schematic.svg")
