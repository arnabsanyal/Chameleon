#!/usr/bin/env python3
"""Score one 24k-image dir for paper Tables 1/2/3.

Computes:
- FID via cleanfid (against COCO val2014)
- CLIP-score via the existing src/test/clip_score.py logic, capped to the
  number of PNGs actually present (so we don't iterate over all 40,504 captions
  when the dir only has 24,576 images).

Writes <img_dir>/coco_score.json.

Usage:
    CUDA_VISIBLE_DEVICES=3 python score_24k.py <img_dir> [--label LABEL]
"""

from __future__ import annotations
import os

import argparse, json, os, sys, time
from pathlib import Path

# Lazy-import the heavy stuff so help is fast.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("img_dir")
    p.add_argument("--label", default=None)
    p.add_argument("--ref",  default=os.path.join(os.environ.get("CHAMELEON_DATA_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "data")), "coco/val2014"))
    p.add_argument("--captions",
                   default=os.path.join(os.environ.get("CHAMELEON_DATA_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "data")), "coco/annotations/captions_val2014.json"))
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="Score only the first N images (sorted by filename). "
                        "Use to sub-score the first 5k of a 24k baseline folder "
                        "for a paired comparison with 5k ablation arms. 0 = all.")
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        sys.exit(f"not a directory: {img_dir}")

    pngs = sorted(p for p in img_dir.iterdir() if p.suffix == ".png")
    n_pngs = len(pngs)
    if n_pngs == 0:
        sys.exit(f"no PNGs in {img_dir}")

    # ── Optional --limit: FID is sample-size-biased, so to compare against N-image
    # ablation arms we score exactly the first N here (symlinked into a temp dir
    # for cleanfid, which takes a directory). The first N PNGs are byte-identical
    # to a fresh N-image run (deterministic per-index seeds).
    import tempfile, shutil
    fid_dir = str(img_dir)
    tmp_dir = None
    if args.limit and args.limit < n_pngs:
        tmp_dir = tempfile.mkdtemp(prefix="score_limit_")
        for src in pngs[:args.limit]:
            os.symlink(src.resolve(), os.path.join(tmp_dir, src.name))
        fid_dir = tmp_dir
        n_pngs = args.limit
        pngs = pngs[:args.limit]

    label = args.label or img_dir.name
    print(f"[score24k] {label}: scoring {n_pngs} PNGs"
          f"{' (first %d of full dir)' % args.limit if tmp_dir else ''} in {img_dir}")

    # ── FID ─────────────────────────────────────────────────────────────────
    from cleanfid import fid as cleanfid_fid
    t0 = time.perf_counter()
    fid_val = cleanfid_fid.compute_fid(fid_dir, args.ref)
    t_fid = time.perf_counter() - t0
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[score24k] {label}: FID={fid_val:.3f} ({t_fid:.1f}s)")

    # ── CLIP ────────────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test"))
    from clip_score import CLIPScoreCalculator, load_coco_captions  # noqa: E402

    prompts = load_coco_captions(args.captions)
    end_ind = min(n_pngs, len(prompts))
    t0 = time.perf_counter()
    calc = CLIPScoreCalculator()
    clip_results = calc.calculate_scores(
        images_dir=str(img_dir),
        prompts=prompts,
        start_ind=0,
        end_ind=end_ind,
        batch_size=args.clip_batch_size,
    )
    t_clip = time.perf_counter() - t0
    stats = clip_results["statistics"]
    print(f"[score24k] {label}: CLIP={stats['mean']:.3f} ± {stats['std']:.3f} "
          f"(n={stats['num_images']}, {t_clip:.1f}s)")

    # ── Persist ─────────────────────────────────────────────────────────────
    result = {
        "label":        label,
        "fid":          float(fid_val),
        "clip_mean":    stats["mean"],
        "clip_std":     stats["std"],
        "clip_median":  stats["median"],
        "n_images":     stats["num_images"],
        "n_missing":    stats["num_missing"],
        "image_dir":    str(img_dir),
        "ref_dir":      args.ref,
        "captions":     args.captions,
        "fid_time_s":   round(t_fid, 1),
        "clip_time_s":  round(t_clip, 1),
    }
    # Limited (sub-scored) runs write to a separate file so the full-set
    # coco_score.json is never clobbered.
    out = img_dir / (f"coco_score_{args.limit}.json" if (args.limit and tmp_dir)
                     else "coco_score.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[score24k] {label}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
