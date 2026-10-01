#!/usr/bin/env python3
"""Repair fig3 (SDXL UNet execution-path schematic) in place on a copy.

All edits are <text> content swaps -- the geometry is untouched.
"""
import re, shutil, sys, xml.etree.ElementTree as ET

SRC = "fig3/Chameleon DiT Execution Path Schematic-image-9.svg"
DST = "figs-fixed/fig3_unet_schematic.svg"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace('', NS)
ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
S = '{%s}' % NS


def num(s):
    m = re.search(r'[-+]?\d*\.?\d+', s or '0')
    return float(m.group()) if m else 0.0


# (exact current text, replacement, approx y to disambiguate or None)
EDITS = [
    ('(10 buckets, T1T10)', '(10 buckets, T1–T10)', None),   # missing en dash
    ('14', 'T4', 168.5),                                          # bucket row label
    ('FP6-E4M3', 'FP8-E4M3', None),                               # T6 row
    ('FP8-ESM2', 'FP8-E5M2', None),                               # T8 row: letter S for 5
    ('held static at inference (Q-Oiffusion style)',
     'held static at inference (Q-Diffusion style)', None),
    ('Measured peak-to-median channel magnitude 48x (up to',
     'Measured peak-to-median channel magnitude 58x (up to', None),
    ('118x), ~1.3x wider than an ordinary conv.',
     '163x), ~1.4x wider than an ordinary conv.', None),
    ('W4: (INT4-sym, INT4-asym, NF4,',
     'W4: {INT4-sym, INT4-asym, NF4,', None),                     # parens -> braces
    ('FP4-E2M1, MXINT4, MXFP4-E2M1)',
     'FP4-E2M1, MXINT4, MXFP4-E2M1}', None),
]
DELETE = [('output-channel', 579.7)]      # duplicate of the rotated axis label

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
        hit = e
        break
    if hit is None:
        missing.append(old)
        continue
    for sub in list(hit):
        hit.remove(sub)
    hit.text = new
    done.append((old, new))

for txt, y in DELETE:
    for e in list(root.iter(S + 'text')):
        if ''.join(e.itertext()) == txt and abs(num(e.get('y')) - y) <= 3:
            parent[e].remove(e)
            done.append((txt, '<deleted>'))
            break

# --- remove the corrupt shading layer -------------------------------------
# The exporter emits three large single-subpath "shading" fills (#848775,
# #547056, #845f3f) that its own PNG renderer collapses to hairlines but that
# spec-compliant renderers (verified with cairosvg AND resvg) fill solid,
# blacking out the T7-T10 format cells. They carry no information; drop them.
DARK = {'#848775', '#547056', '#845f3f'}
killed = 0
for e in [e for e in root.iter(S + 'path')
          if e.get('fill') in DARK and len(e.get('d', '')) > 900]:
    parent[e].remove(e); killed += 1
done.append((f'{killed} corrupt shading paths', '<deleted>'))

# --- restore bold on the callout headings ---------------------------------
BOLD = {'Activation Callout (Runtime)', 'Weight Callout (Offline)',
        'Few-step variant (SDXL-Turbo)'}
for e in root.iter(S + 'text'):
    if ''.join(e.itertext()) in BOLD:
        e.set('font-weight', 'bold')
        done.append((''.join(e.itertext()), '<bold>'))

tree.write(DST, encoding='utf-8', xml_declaration=True)
for a, b in done:
    print(f"  OK   {a!r} -> {b!r}")
for m in missing:
    print(f"  MISS {m!r}")
print(f"\n{len(done)} applied, {len(missing)} not found -> {DST}")
sys.exit(1 if missing else 0)
