# FP16 SDXL baseline

Generates the unquantized FP16 reference for the SDXL rows of Table 1
([stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
50 steps, CFG 7.5, 1024×1024).

## Usage

From the repository root, with the `chameleon` environment active:

```bash
python src/baseline/generation/baseline_generation.py \
    --coco-captions data/coco/annotations/captions_val2014.json \
    --num-samples 24576 --batch-size 16 \
    --output-dir outputs/sdxl_fp16

python src/paper_figures/score_24k.py outputs/sdxl_fp16/images --label sdxl_fp16
```

Or run `bash src/baseline/generation/run_baseline.sh` for an interactive menu.

Images are written as `images/{index:05d}.png`. Captions are the COCO val2014 annotations shuffled with seed 42, and
each image's latent seed is its global index, so every method in the paper sees the same prompt and seed at the same
index. Use `--start-ind` to resume an interrupted run.

| Argument | Default | Description |
|---|---|---|
| `--coco-captions` | none | COCO captions JSON; without it, `--prompt-file` or built-in sample prompts are used |
| `--prompt-file` | none | text file with one prompt per line |
| `--num-samples` | 100 | number of images |
| `--start-ind` | 0 | first index to generate (for resuming) |
| `--batch-size` | 4 | images per forward pass |
| `--num-inference-steps` | 50 | denoising steps |
| `--guidance-scale` | 7.5 | classifier-free guidance scale |
| `--output-dir` | `$CHAMELEON_OUTPUT_ROOT` | output root; images go to `<output-dir>/images/` |
| `--calculate-fid` / `--reference-dir` | off | compute clean-FID against a reference folder after generation |

`example_usage.py` shows the same generator used from Python.
