#!/usr/bin/env python3
"""Repair fig4 (PixArt-alpha DiT execution-path schematic) onto a copy."""
import re, shutil, sys, xml.etree.ElementTree as ET

SRC = "fig4/Chameleon DiT Execution Path Schematic-image-7.svg"
DST = "figs-fixed/fig4_dit_schematic.svg"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace('', NS)
ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
S = '{%s}' % NS


def num(s):
    m = re.search(r'[-+]?\d*\.?\d+', s or '0')
    return float(m.group()) if m else 0.0


EDITS = [
    ('Cross Attention', 'Cross-Attention', None),        # hyphen dropped
    ('pura noise', 'pure noise', None),
    ('TT0', 'T10', None),
    ('Per-linear joint (g. f) search.', 'Per-linear joint (g, f) search.', None),
]

# NOTE: the exporter emits the rotated axis labels ("clean signal / (low t)",
# "pure noise / (high t)", "Group size (g)") as TINY horizontal <text> at
# 4.4-10px, stacked to fake rotation. Enlarging them to a legible size makes
# them collide with each other and with the row labels, because the fake
# rotation gives them no vertical room. Left at source size deliberately;
# fixing this properly means redrawing those labels as real rotated text.
ENLARGE = {}

BOLD = {'Activation Callout (Runtime)', 'Weight Callout (Offline)', 'DiT Block'}

shutil.copy(SRC, DST)
tree = ET.parse(DST)
root = tree.getroot()
parent = {c: p for p in root.iter() for c in p}

done, missing = [], []
for old, new, y in EDITS:
    hit = None
    for e in root.iter(S + 'text'):
        if ''.join(e.itertext()) != old:
            continue
        if y is not None and abs(num(e.get('y')) - y) > 3:
            continue
        hit = e; break
    if hit is None:
        missing.append(old); continue
    for sub in list(hit):
        hit.remove(sub)
    hit.text = new
    done.append((old, new))

for e in root.iter(S + 'text'):
    t = ''.join(e.itertext())
    if t in ENLARGE:
        old = num(e.get('font-size'))
        if old < ENLARGE[t]:
            e.set('font-size', f'{ENLARGE[t]}px')
            done.append((f'{t!r} font-size', f'{old:.1f} -> {ENLARGE[t]}'))
    if t in BOLD:
        e.set('font-weight', 'bold')
        done.append((t, '<bold>'))

DARK = {'#848775', '#547056', '#845f3f'}
killed = 0
for e in [e for e in root.iter(S + 'path')
          if e.get('fill') in DARK and len(e.get('d', '')) > 900]:
    parent[e].remove(e); killed += 1
done.append((f'{killed} corrupt shading paths', '<deleted>'))

tree.write(DST, encoding='utf-8', xml_declaration=True)
for a, b in done:
    print(f"  OK   {a!r} -> {b!r}")
for m in missing:
    print(f"  MISS {m!r}")
print(f"\n{len(done)} applied, {len(missing)} not found -> {DST}")
sys.exit(1 if missing else 0)
