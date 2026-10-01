"""Instrumented calibration runner for Table 8 (calibration cost).

Wraps the existing per-model calibration command (--num-samples 0 --save-...)
as a subprocess; times it, polls nvidia-smi for that PID's peak GPU memory,
reads the saved cache size after exit, and writes a JSON to the cache dir.

Usage:
    CUDA_VISIBLE_DEVICES=4 python calib_cost.py --model sdxl
    CUDA_VISIBLE_DEVICES=6 python calib_cost.py --model lcm
    CUDA_VISIBLE_DEVICES=3 python calib_cost.py --model dit

The three models are independent; run them on different GPUs in parallel.
"""

from __future__ import annotations
import os

import argparse, json, os, subprocess, sys, time
from pathlib import Path


REPO_ROOT  = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
CACHE_DIR  = Path(os.path.join(os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "outputs")), "ablations/_calibration_cache"))
NUM_BUCKETS       = 10
NUM_CALIB_SAMPLES = 128

MODELS = {
    "sdxl": {
        "cwd":   REPO_ROOT / "src/sdxl-chameleon/generation",
        "cmd":   ["chameleon_generation.py",
                  "--weight-bits", "4",
                  "--num-buckets", str(NUM_BUCKETS),
                  "--num-calibration-samples", str(NUM_CALIB_SAMPLES),
                  "--num-samples", "0",
                  "--save-weights",        str(CACHE_DIR / "sdxl_w4a8_weights.pt"),
                  "--save-activation-lut", str(CACHE_DIR / "sdxl_w4a8_lut.pt")],
        "files": [CACHE_DIR / "sdxl_w4a8_weights.pt",
                  CACHE_DIR / "sdxl_w4a8_lut.pt"],
    },
    # "lcm" retired: Chameleon-on-MixDQ has no calibration step to cost-measure
    # (it reuses MixDQ's shipped activation scales; the weight SNR search runs
    # inline at injection, not as a separate cached artifact).
    "dit": {
        "cwd":   REPO_ROOT / "src/chameleon-dit/generation",
        "cmd":   ["chameleon_dit_generation.py",
                  "--weight-bits", "4",
                  "--num-samples", "0",
                  "--save-weights", str(CACHE_DIR / "dit_w4a8_weights.pt")],
        "files": [CACHE_DIR / "dit_w4a8_weights.pt"],
    },
}


def query_proc_mem_mb(pid: int) -> int:
    """Return MiB used by `pid` on the active GPU(s); 0 if no longer running."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    total = 0
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            if int(parts[0]) == pid:
                total += int(parts[1])
        except (ValueError, IndexError):
            continue
    return total


def run_one(model: str) -> dict:
    spec = MODELS[model]
    cwd  = spec["cwd"]
    argv = ["python"] + spec["cmd"]

    print(f"[calib_cost] {model}: cwd={cwd}")
    print(f"[calib_cost] {model}: argv={' '.join(argv)}")
    print(f"[calib_cost] {model}: CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    t0 = time.perf_counter()
    proc = subprocess.Popen(argv, cwd=str(cwd))

    peak_mb = 0
    poll_interval = 5.0
    while proc.poll() is None:
        try:
            mb = query_proc_mem_mb(proc.pid)
            if mb > peak_mb:
                peak_mb = mb
        except Exception as e:
            print(f"[calib_cost] poll warn: {e}", file=sys.stderr)
        time.sleep(poll_interval)

    rc = proc.returncode
    elapsed = time.perf_counter() - t0
    print(f"[calib_cost] {model}: subprocess rc={rc}  elapsed={elapsed:.1f}s")

    if rc != 0:
        raise SystemExit(f"calibration subprocess failed rc={rc}")

    file_bytes = 0
    file_sizes = {}
    for f in spec["files"]:
        if f.exists():
            sz = f.stat().st_size
            file_sizes[f.name] = sz
            file_bytes += sz

    cost = {
        "model":           model,
        "calib_time_s":    round(elapsed, 2),
        "calib_time_human": f"{int(elapsed//60)}m {int(elapsed%60)}s",
        "peak_mem_gb":     round(peak_mb / 1024, 2),
        "peak_mem_mb":     peak_mb,
        "lut_bytes":       file_bytes,
        "lut_kb":          round(file_bytes / 1024, 2),
        "files":           file_sizes,
        "num_buckets":     NUM_BUCKETS,
        "num_calib_samples": NUM_CALIB_SAMPLES,
    }
    out_path = CACHE_DIR / f"{model}_w4a8_calib_cost.json"
    out_path.write_text(json.dumps(cost, indent=2))
    print(f"[calib_cost] wrote {out_path}")
    print(json.dumps(cost, indent=2))
    return cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    args = p.parse_args()
    run_one(args.model)


if __name__ == "__main__":
    main()
