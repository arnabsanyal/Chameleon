"""
Lightweight progress watchdog for an ablation row.

Two modes:

  Polling (default):
      Loops, every --poll-seconds:
        - counts images in --row
        - if a 20/40/60/80/100% milestone has been crossed since last poll,
          samples a few generated images, computes pixel statistics, decides
          a DIRECTION verdict, and writes the row's progress.json
        - if no new images have appeared in --stall-minutes, marks STALLED
      Exits when the 100% milestone has been logged or it receives SIGTERM.

  Single-shot (--once):
      Computes one final checkpoint (uses the highest milestone the current
      image count satisfies) and writes progress.json. Used by
      verifier_wrap.sh as the EXIT trap so the final 100% milestone always
      gets recorded even if the watchdog gets killed before its next poll.

The watchdog never aborts the parent generation — it only logs. Use
DIRECTION verdicts in progress.json to decide whether to manually kill a
row, but the most common signal you'll act on is `stalled=true` (means the
generation crashed and you should look at the row log).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

CHECKPOINTS = (0.20, 0.40, 0.60, 0.80, 1.00)
SAMPLE_SIZE = 16             # images sampled per milestone for stats
STD_DEGENERATE = 4.0         # px-std below this counts as degenerate
DEGEN_FRACTION_BAD = 0.30    # >30% degenerate in sample → DEGENERATE verdict
STD_REGRESSION_DROP = 0.50   # mean_std drops by half → REGRESSION verdict


def scan_images(row: Path) -> list[Path]:
    if not row.exists():
        return []
    out = []
    for ext in ("png", "jpg", "jpeg", "webp"):
        out.extend(row.rglob(f"*.{ext}"))
    return sorted(out)


def image_stats(path: Path) -> dict | None:
    try:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    return {
        "std":   float(arr.std()),
        "mean":  float(arr.mean()),
        "max":   float(arr.max()),
        "min":   float(arr.min()),
    }


def sample_stats(images: list[Path], k: int = SAMPLE_SIZE) -> dict:
    rng = np.random.default_rng(0xCAFE)
    if len(images) > k:
        idx = rng.choice(len(images), size=k, replace=False)
        sub = [images[i] for i in idx]
    else:
        sub = images
    stats = [s for s in (image_stats(p) for p in sub) if s is not None]
    if not stats:
        return {
            "sampled": len(sub), "readable": 0, "degenerate_count": len(sub),
            "mean_std": 0.0, "min_std": 0.0, "mean_pixel_mean": 0.0,
        }
    stds  = [s["std"]  for s in stats]
    means = [s["mean"] for s in stats]
    return {
        "sampled":          len(sub),
        "readable":         len(stats),
        "degenerate_count": sum(1 for s in stds if s < STD_DEGENERATE),
        "mean_std":         float(np.mean(stds)),
        "min_std":          float(np.min(stds)),
        "mean_pixel_mean":  float(np.mean(means)),
    }


def decide_direction(cur: dict, prev: dict | None) -> str:
    if cur["readable"] == 0:
        return "DEGENERATE"
    degen_frac = cur["degenerate_count"] / cur["sampled"]
    if degen_frac > DEGEN_FRACTION_BAD:
        return "DEGENERATE"
    if prev is not None and prev.get("mean_std", 0) > 0:
        if cur["mean_std"] < prev["mean_std"] * STD_REGRESSION_DROP:
            return "REGRESSION"
    return "OK"


def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"row": None, "checkpoints": {}, "events": []}


def save_progress(path: Path, prog: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(prog, indent=2))
    tmp.replace(path)


def label_for(frac: float) -> str:
    return f"{int(round(frac * 100))}%"


def highest_milestone_reached(frac: float) -> float | None:
    crossed = [c for c in CHECKPOINTS if frac >= c]
    return max(crossed) if crossed else None


def emit_milestone(row: Path, total: int, prog: dict, milestone: float, count: int):
    images = scan_images(row)
    stats = sample_stats(images)
    label = label_for(milestone)

    # Find previous milestone's stats for direction comparison.
    prev_stats = None
    for c in CHECKPOINTS:
        if c >= milestone:
            break
        if label_for(c) in prog["checkpoints"]:
            prev_stats = prog["checkpoints"][label_for(c)]

    direction = decide_direction(stats, prev_stats)
    chk = {
        "fraction":  milestone,
        "count":     count,
        "total":     total,
        "timestamp": time.time(),
        "direction": direction,
        **stats,
    }
    prog["checkpoints"][label] = chk
    msg = (f"[verify] {row.name} {label} count={count}/{total} "
           f"mean_std={stats['mean_std']:.2f} degen={stats['degenerate_count']}"
           f"/{stats['sampled']} → {direction}")
    prog["events"].append({"t": time.time(), "msg": msg})
    print(msg, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--total", type=int, required=True)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--stall-minutes", type=int, default=20)
    p.add_argument("--once", action="store_true",
                   help="Single synchronous checkpoint at current count, then exit.")
    args = p.parse_args()

    row = Path(args.row)
    progress_path = row / "progress.json"
    row.mkdir(parents=True, exist_ok=True)
    prog = load_progress(progress_path)
    prog["row"] = str(row)

    if args.once:
        images = scan_images(row)
        cur = len(images)
        frac = cur / max(args.total, 1)
        m = highest_milestone_reached(frac)
        if m is not None and label_for(m) not in prog["checkpoints"]:
            emit_milestone(row, args.total, prog, m, cur)
        # If we already passed 100% completion (sometimes count > total), record final.
        if frac >= 1.0 and "100%" not in prog["checkpoints"]:
            emit_milestone(row, args.total, prog, 1.00, cur)
        prog["final_count"] = cur
        prog["final_fraction"] = frac
        save_progress(progress_path, prog)
        return

    # Polling mode.
    last_count = 0
    last_change_t = time.time()
    next_idx = 0
    # If a previous run already logged some milestones, advance past them.
    while next_idx < len(CHECKPOINTS) and label_for(CHECKPOINTS[next_idx]) in prog["checkpoints"]:
        next_idx += 1

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT,  lambda *_: stop.update(flag=True))

    while next_idx < len(CHECKPOINTS) and not stop["flag"]:
        time.sleep(args.poll_seconds)
        images = scan_images(row)
        cur = len(images)
        frac = cur / max(args.total, 1)

        if cur > last_count:
            last_count = cur
            last_change_t = time.time()
        elif (time.time() - last_change_t) > args.stall_minutes * 60 and cur > 0:
            stall = {
                "stalled":         True,
                "stall_at_count":  cur,
                "stall_minutes":   args.stall_minutes,
                "timestamp":       time.time(),
            }
            prog.setdefault("events", []).append(
                {"t": time.time(),
                 "msg": f"[verify] {row.name} STALLED at {cur}/{args.total} "
                        f"(no new images in {args.stall_minutes}m)"})
            prog.update(stall)
            save_progress(progress_path, prog)
            print(f"[verify] {row.name} STALLED at {cur}/{args.total}",
                  file=sys.stderr, flush=True)
            # Reset stall detector so we don't spam.
            last_change_t = time.time()

        # Cross any new milestones since last poll.
        while next_idx < len(CHECKPOINTS) and frac >= CHECKPOINTS[next_idx]:
            emit_milestone(row, args.total, prog, CHECKPOINTS[next_idx], cur)
            save_progress(progress_path, prog)
            next_idx += 1


if __name__ == "__main__":
    main()
