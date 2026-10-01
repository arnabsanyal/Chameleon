#!/usr/bin/env python3
"""Structural check on images/chameleon_schematic.svg.

Asserts the semantics that Figure 2 exists to convey, so a later edit that
quietly re-adds the forbidden edge fails loudly instead of shipping.
"""
import re, sys, xml.etree.ElementTree as ET

NS = '{http://www.w3.org/2000/svg}'
SVG = 'images/chameleon_schematic.svg'
root = ET.parse(SVG).getroot()

boxes, edges, sizes = {}, [], []
for e in root.iter():
    i = e.get('id') or ''
    if e.tag == NS + 'rect' and i.startswith('box-'):
        x, y, w, h = (float(e.get(k)) for k in ('x', 'y', 'width', 'height'))
        boxes[i] = (x, y, x + w, y + h)
    if e.tag == NS + 'path' and i.startswith('edge-'):
        n = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+', e.get('d'))]
        edges.append((i, (n[0], n[1]), (n[2], n[3])))
    if e.tag == NS + 'text':
        sizes.append(float(e.get('font-size')))


def owner(p, tol=4):
    for b, (x0, y0, x1, y1) in boxes.items():
        if x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol:
            return b
    return '?'


E = {(owner(a), owner(b)) for _, a, b in edges}
fail = []

if ('box-act-lut', 'box-fewstep') in E:
    fail.append('FORBIDDEN edge act-lut -> fewstep is present')
for arm in ('box-unet', 'box-dit'):
    if ('box-act-lut', arm) not in E:
        fail.append(f'missing act-lut -> {arm}')
for arm in ('box-unet', 'box-fewstep', 'box-dit'):
    if ('box-palette', arm) not in E:
        fail.append(f'missing palette -> {arm}')
for a, b in (('box-timestep', 'box-act-stats'), ('box-act-stats', 'box-act-lut')):
    if (a, b) not in E:
        fail.append(f'broken activation chain: {a} -> {b}')
if '?' in {o for e in E for o in e}:
    fail.append('an edge endpoint does not land on any box')

# overlap check
ids = list(boxes)
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = boxes[ids[i]], boxes[ids[j]]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            fail.append(f'boxes overlap: {ids[i]} / {ids[j]}')

print(f'{len(boxes)} boxes, {len(edges)} edges, {len(sizes)} text runs')
print(f'font sizes: min {min(sizes):.1f}  max {max(sizes):.1f}')
for a, b in sorted(E):
    print(f'  {a:17} -> {b}')
if fail:
    print('\nFAIL:'); [print('  -', f) for f in fail]; sys.exit(1)
print('\nOK: all structural assertions hold')
