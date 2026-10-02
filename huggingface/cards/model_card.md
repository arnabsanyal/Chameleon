---
license: mit
base_model:
  - stabilityai/stable-diffusion-xl-base-1.0
  - stabilityai/sdxl-turbo
  - PixArt-alpha/PixArt-XL-2-1024-MS
pipeline_tag: text-to-image
tags:
  - quantization
  - post-training-quantization
  - fp8
  - microscaling
  - diffusion
  - text-to-image
---

# Chameleon: calibration artifacts

Calibration outputs for **Chameleon: Dynamic Format Adapter for Efficient Diffusion**
([arXiv:2609.33496](https://arxiv.org/abs/2609.33496), under review at ICLR 2027).
Code: [github.com/arnabsanyal/Chameleon](https://github.com/arnabsanyal/Chameleon) ·
Project page: [arnabsanyal.github.io/iclr2027/chameleon.html](https://arnabsanyal.github.io/iclr2027/chameleon.html) ·
Generated samples: [datasets/arnabsanyal/chameleon-coco-samples](https://huggingface.co/datasets/arnabsanyal/chameleon-coco-samples)

Chameleon is a post-training quantizer that chooses a number format, out of INT8, FP8 E4M3/E5M2 and MX formats,
per layer and per timestep bucket, routed by activation kurtosis and the diffusion signal-to-noise ratio.
It ships **no new model weights**: it quantizes the public checkpoints at load time. These files record what
calibration chose, so you can reproduce the Table 1 rows without recalibrating.

## Files

| Path | Backbone | Setting | Contents |
|---|---|---|---|
| `sdxl/w8a8/chameleon_sdxl_w8a8.json` | SDXL base 1.0 | W8A8 | per-channel weight formats + activation LUT (372 layers × 10 buckets) |
| `sdxl/w4a8/chameleon_sdxl_w4a8.json` | SDXL base 1.0 | W4A8 | per-channel weight formats + activation LUT (372 layers × 10 buckets) |
| `pixart-alpha/w8a8/chameleon_pixart_w8a8.json` | PixArt-α XL/2 1024-MS | W8A8 | per-layer (group size, format) + activation LUT (282 layers × 10 buckets) |
| `pixart-alpha/w4a8/chameleon_pixart_w4a8.json` | PixArt-α XL/2 1024-MS | W4A8 | input-aware (group size, format) + activation LUT (282 layers × 10 buckets) |
| `results/table1.json` | all | all | FID and CLIP for every Table 1 row (24,576 COCO-2014 images each) |
| `SHA256SUMS` | | | checksums of the files above |

**SDXL-Turbo has no file here.** On the few-step path, Chameleon's weight formats are folded deterministically at
load time on top of MixDQ's calibrated activations. MixDQ's configs are fetched from MixDQ's repository by
`third_party/mixdq/fetch_configs.sh` rather than redistributed, so the code repository alone reproduces those rows.

## Usage

Set up the code repository first (see its README), then:

```bash
hf download arnabsanyal/chameleon --local-dir artifacts
CAPS=data/coco/annotations/captions_val2014.json

# SDXL, W4A8: no calibration pass, the LUT is embedded in the file
python src/sdxl-chameleon/generation/chameleon_generation.py --weight-bits 4 \
    --load-weights artifacts/sdxl/w4a8/chameleon_sdxl_w4a8.json \
    --coco-captions $CAPS --num-samples 24576 --num-inference-steps 50 --guidance-scale 7.5

# PixArt-alpha, W4A8 (input-aware weight selection)
python src/chameleon-dit/generation/chameleon_dit_generation.py --weight-bits 4 --input-aware-weights \
    --load-weights artifacts/pixart-alpha/w4a8/chameleon_pixart_w4a8.json \
    --coco-captions $CAPS --num-samples 24576 --num-inference-steps 20 --guidance-scale 4.5
```

Use `--weight-bits 8` with the `w8a8` files (and no `--input-aware-weights`) for the W8A8 rows.
The base checkpoints are downloaded from their own repositories and remain under their own licences.

## Results (Table 1)

clean-FID against COCO-2014 val, CLIP score against the generating captions; 24,576 images per row.

| Backbone | Method | Bits | FID ↓ | CLIP ↑ |
|---|---|---|---:|---:|
| SDXL (50 steps, CFG 7.5, 1024²) | FP16 | W16A16 | 16.16 | 26.88 |
| | Chameleon | W8A8 | 14.23 | 26.70 |
| | Chameleon | W4A8 | 14.43 | 26.64 |
| SDXL-Turbo (1 step, no CFG, 512²) | FP16 | W16A16 | 21.63 | 26.63 |
| | Chameleon | W8A8 | 20.93 | 26.62 |
| | Chameleon | W4A8 | 21.29 | 26.85 |
| PixArt-α (20 steps, CFG 4.5, 1024²) | FP16 | W16A16 | 27.63 | 25.99 |
| | Chameleon | W8A8 | 24.33 | 25.81 |
| | Chameleon | W4A8 | 22.24 | 26.09 |

`results/table1.json` also holds the Q-Diffusion, PTQ4DM, MixDQ and Q-DiT rows.

## Citation

```bibtex
@misc{sanyal2026chameleon,
  title         = {Chameleon: Dynamic Format Adapter for Efficient Diffusion},
  author        = {Sanyal, Arnab and Chinchali, Sandeep},
  year          = {2026},
  eprint        = {2609.33496},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.33496},
  howpublished  = {Under review at the International Conference on Learning Representations (ICLR) 2027}
}
```
