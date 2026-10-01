# Chameleon Ablations

Four merged ablations covering the headline claims of the paper. Designed to fit
in **~9–12 GPU-hours on a single A100** (or one overnight run on 4× A100) by
sub-sampling COCO to 1k prompts and reusing one calibration pass per
(model, bit-width).

> **Caveat for table captions.** FID values produced by these scripts are
> computed on a 1,000-prompt COCO subset and are internally consistent across
> ablation rows, but should NOT be cross-compared with the 5k main-table
> numbers. Note this in any table caption that uses these results.

## The 4 Ablations

| # | Folder | Model(s) | Rows | What it isolates |
|---|---|---|---|---|
| 1 | `ablation_1_routing/`   | SDXL W4A8         | 4 + 1 perturb addendum | κ vs SNR vs both vs single-format — the headline routing claim |
| 2 | `ablation_2_palette/`   | SDXL W4A8         | 6 | Single-format baselines + drop-one (covers shortcut/MXFP8 claim) |
| 3 | `ablation_3_arch_fork/` | LCM, DiT          | 2 + 3 = 5 | BAQ on LCM; macro+micro on DiT |
| 4 | `ablation_4_buckets/`   | SDXL W4A8         | 4 | Bucket-count sweep B ∈ {1, 5, 10, 20} |

> **Note — Chameleon-LCM is now Chameleon-on-MixDQ.** The few-step LCM target
> keeps MixDQ's calibrated static activations and only swaps in Chameleon's
> adaptive *weight* palette (`--chameleon-weights` on
> `src/chameleon-lcm/generation/chameleon_lcm_generation.py`). Consequences for
> ablations: (a) there is **no LCM activation routing**, so the routing
> ablations (1) and activation-format palette drops (2) are **SDXL/DiT-only** —
> they don't apply to lcm; (b) there is **no LCM calibration step**, so the LCM
> calibration-cache/cost jobs are retired; (c) **BAQ on/off** (ablation 3) maps
> to MixDQ's `--no-bos` (see `patches/lcm_disable_baq.py`); (d) LCM **weight-
> palette** drops still work (`palette_drop_format --weight-drop`).

## Run order (stop-early friendly, 4× A100 wall-clock)

When run sequentially via `run_all.sh 1`, `... 2`, etc., the order is:

```
1. Ablation 1  (~2 h on 4 GPUs, 5 SDXL rows)        ← headline κ-vs-SNR
2. Ablation 2  (~2 h on 4 GPUs, 6 SDXL rows)        ← palette + shortcut
3. Ablation 4  (~2 h on 4 GPUs, 4 SDXL rows)        ← bucket sweep
4. Ablation 3  (~40 min on 4 GPUs, 2 LCM + 3 DiT)   ← architectural forks
```

But `run_all.sh all` is faster: it pours all 20 rows into one shared queue, so
the slow SDXL rows and the cheap LCM/DiT rows interleave naturally and no GPU
sits idle waiting for a barrier — total ~9 h on 4× A100.

If a deadline forces a cut, the natural drops are: row 5 of Ablation 1
(threshold perturbation), and the `drop_NF4` / `drop_MXFP4` rows of Ablation 2.

## Parallel execution (4× A100)

`run_all.sh all` opens **one shared 4-GPU dispatcher** and submits all 20
generation rows from all four ablations into the same queue. Each in-flight
job runs with `CUDA_VISIBLE_DEVICES` pinned to its assigned GPU id; the moment
any row finishes, the next queued row starts on the freed GPU. There is no
per-ablation barrier — the tail of ablation 1 does NOT wait before ablation 2
starts.

### Picking which GPUs to use

```bash
GPU_IDS="4,5,6,7" bash run_all.sh all     # explicit list (any ids)
NUM_GPUS=8        bash run_all.sh all     # ids 0..N-1
bash run_all.sh all                        # default: 0..3
```

`GPU_IDS` takes precedence over `NUM_GPUS`. The dispatcher seeds the GPU pool
with exactly the ids you list; each in-flight job gets `CUDA_VISIBLE_DEVICES`
pinned to its assigned id. On a machine with GPUs 0–7, setting
`GPU_IDS="4,5,6,7"` leaves 0–3 untouched for other tenants.

### 20% checkpoint verifier

Every generation row spawns a background `verify_progress.py` watchdog that:

- polls the row's output directory every 60 s (configurable: `VERIFIER_POLL`)
- at each **20 / 40 / 60 / 80 / 100%** milestone, samples 16 generated
  images, computes pixel statistics (mean/std/min/max), and writes them to
  `<row>/progress.json` along with a **DIRECTION** verdict:
    - `OK` — stats stable or improving
    - `REGRESSION` — `mean_std` dropped >50% from previous milestone
      (model collapsed mid-run)
    - `DEGENERATE` — >30% of the sampled images have `std < 4` (all-black,
      all-noise, NaN propagation)
- detects **stalled generation**: if the image count hasn't changed in
  `VERIFIER_STALL_MIN` minutes (default 20), marks `stalled=true` in
  `progress.json` and logs to stderr — useful for catching crashes that
  leave Python hanging without dying.

The verifier is observation-only — it never aborts the parent generation.
You read `progress.json` to decide whether to manually kill a row.

To check progress across all in-flight rows:

```bash
for f in $OUT_ROOT/ablation_*/*/progress.json; do
    jq -r '"\(.row | split("/")[-1])\t\(.checkpoints | to_entries | map("\(.key):\(.value.direction)") | join(" "))"' "$f"
done
```

Set `VERIFIER_DISABLE=1` to skip watchdog spawning entirely.

- Calibration phase (3 jobs: SDXL/LCM/DiT W4A8) runs in parallel on 3 of the 4
  GPUs as a one-time prelude, ~30 minutes wall-clock.
- Generation phase (20 rows) drains the shared queue, ~9 hours wall-clock on
  4× A100 (vs 36 GPU-hours sequential).
- Per-row stdout/stderr lives at `$OUT_ROOT/_logs/<timestamp>/<row>.log` so
  concurrent jobs don't interleave output.

To run individual ablations on 4 GPUs (each inits its own dispatcher):

```bash
bash run_all.sh 1            # ablation 1 only, 4 GPUs
bash run_all.sh 1 2          # ablations 1+2, shared dispatcher
NUM_GPUS=8 bash run_all.sh all   # override pool size
```

For 4 separate nodes (rather than 4 GPUs in one node) see the multi-node
note in `common/dispatch.sh` — replace the `bash -c` line with `ssh node$gpu
bash -c` and ensure $OUT_ROOT is on shared storage.

## How a row is run

1. **Calibration is shared.** `common/cache_calibration.sh` runs ONCE per
   (model, bit-width), dumps the per-(layer, bucket) statistics, and saves both
   weight quantization and activation LUT to disk. Every ablation row then
   loads those cached files via `--load-weights` / `--load-activation-lut`.
   Routing variants (κ-only, SNR-only, drop-MXFP8, …) are applied at LUT-
   construction time inside the patch shims in `patches/` — they do NOT
   require re-running calibration.

2. **Generation.** Each row calls the model's existing CLI
   (`chameleon_generation.py`, `chameleon_lcm_generation.py`,
   `chameleon_dit_generation.py`) wrapped through a patch shim from
   `patches/` that monkey-patches `route_format_by_kurtosis_snr` (or the
   per-architecture equivalent) before the CLI's `main()` runs.

3. **Scoring.** `common/score.sh` invokes clean-FID against the cached COCO 1k
   reference Inception statistics, plus CLIP score on the generated images.

## Top-level driver

```bash
bash run_all.sh           # interactive menu
bash run_all.sh 1         # run ablation 1 only
bash run_all.sh all       # run 1 → 2 → 4 → 3 sequentially
```

Output lands at `$CHAMELEON_OUTPUT_ROOT/ablations/<ablation>/<row>/`.

## GPU-hour budget (single A100, 1k subset)

| Component                                         | Cost   |
|---------------------------------------------------|--------|
| Cached calibration (SDXL + LCM + PixArt)          | ~1.0 h |
| Ablation 1: 5 SDXL rows × 1.95 h                  | 9.75 h |
| Ablation 2: 6 SDXL rows × 1.95 h                  | 11.70 h |
| Ablation 3: 2 LCM (~0.20 h) + 3 PixArt (~0.64 h)  | 2.32 h |
| Ablation 4: 4 SDXL rows × 1.95 h                  | 7.80 h |
| FID/CLIP scoring overhead                         | ~0.5 h |
| Buffer (10%)                                      | ~3.3 h |
| **Total**                                         | **~36 h** |

With the two natural cuts above: **~30 h**.
With 4× A100 parallel-over-rows: **~9 h overnight**.
