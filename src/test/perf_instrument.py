"""Simulated-memory + latency instrumentation for the generation scripts.

Drop this into a generation loop with three lines:

    from perf_instrument import PerfTracker
    tracker = PerfTracker(label="sdxl_fp16", device=self.device)
    tracker.reset()        # call AFTER the model is on the GPU but BEFORE the loop
    ...                    # existing for-batch-in-prompts loop
    tracker.add(n_in_batch)# inside the loop, after pipe(...) returns
    ...
    tracker.finish()       # after the loop
    tracker.dump_json(f"{output_dir}/perf.json")

What gets measured
- avg_latency_s_per_image = wall(start..end) / n_images   (covers all batches)
- peak_mem_gb             = torch.cuda.max_memory_allocated() / 2**30
                            (reset just before the loop, so model weights are
                            already resident — this is the *inference-only*
                            peak, not the load peak.  Toggle with
                            include_model_load=True to measure both.)

The numbers are deliberately wall-clock end-to-end including I/O (PNG save),
because that matches what the user sees.  For an "engine-only" latency, set
exclude_save=True and the saver loop will be skipped in the timer (caller is
responsible for moving image.save out of the timed region).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    _HAS_CUDA = False


class PerfTracker:
    def __init__(self, label: str = "run", device: str = "cuda"):
        self.label = label
        self.device = str(device)
        self.n_images = 0
        self.batch_times: list[float] = []
        self._t_batch_start: float | None = None
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._included_model_load = False

    # ── timer boundary ────────────────────────────────────────────────────
    def reset(self, include_model_load: bool = False) -> None:
        """Anchor the start of the measurement window.

        Call this AFTER ``pipe.to(device)``/quantization is finished (so model
        load isn't billed to latency) but BEFORE the generation loop.
        """
        if _HAS_CUDA:
            torch.cuda.synchronize()
            if not include_model_load:
                torch.cuda.reset_peak_memory_stats()
        self._included_model_load = include_model_load
        self._t_start = time.perf_counter()

    def batch_start(self) -> None:
        """Optional: mark the start of a single batch for per-batch latency."""
        if _HAS_CUDA:
            torch.cuda.synchronize()
        self._t_batch_start = time.perf_counter()

    def batch_end(self, n: int) -> None:
        """Optional: pair with batch_start() to record per-batch wall time."""
        if _HAS_CUDA:
            torch.cuda.synchronize()
        if self._t_batch_start is not None:
            self.batch_times.append(time.perf_counter() - self._t_batch_start)
            self._t_batch_start = None
        self.n_images += n

    def add(self, n: int) -> None:
        """Lightweight: bump the image counter without per-batch latency."""
        self.n_images += n

    def finish(self) -> None:
        if _HAS_CUDA:
            torch.cuda.synchronize()
        self._t_end = time.perf_counter()

    # ── summary / serialisation ───────────────────────────────────────────
    def summary(self) -> dict:
        if self._t_start is None or self._t_end is None:
            raise RuntimeError("PerfTracker.summary called before reset()/finish()")
        elapsed = self._t_end - self._t_start
        n = max(self.n_images, 1)
        peak_bytes = (
            torch.cuda.max_memory_allocated() if _HAS_CUDA else 0
        )
        peak_gb = peak_bytes / (1024 ** 3)
        out = {
            "label": self.label,
            "device": self.device,
            "n_images": self.n_images,
            "total_wall_s": round(elapsed, 3),
            "avg_latency_s_per_image": round(elapsed / n, 3),
            "peak_mem_gb": round(peak_gb, 3),
            "included_model_load": self._included_model_load,
        }
        if self.batch_times:
            bt = sorted(self.batch_times)
            mid = bt[len(bt) // 2]
            out["batch_count"] = len(bt)
            out["batch_latency_median_s"] = round(mid, 3)
            out["batch_latency_mean_s"] = round(sum(bt) / len(bt), 3)
        return out

    def dump_json(self, path) -> dict:
        """Persist summary() to ``path`` and return the dict.  Parent dir
        is created if needed."""
        s = self.summary()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=2))
        print(
            f"[perf] {s['label']}: n={s['n_images']}  "
            f"avg_latency={s['avg_latency_s_per_image']:.3f}s/img  "
            f"peak_mem={s['peak_mem_gb']:.3f} GB  "
            f"-> {p}"
        )
        return s
