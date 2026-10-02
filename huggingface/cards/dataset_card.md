---
license: other
license_name: per-split-model-licences
pretty_name: Chameleon COCO-2014 samples
task_categories:
  - text-to-image
size_categories:
  - 100K<n<1M
tags:
  - quantization
  - diffusion
  - fid
  - coco
configs:
  - config_name: default
    data_files:
{{DATA_FILES}}
---

# Chameleon: COCO-2014 samples behind Table 1

The generated images that produce the FID and CLIP scores in Table 1 of **Chameleon: Dynamic Format Adapter for
Efficient Diffusion** ([arXiv:2609.33496](https://arxiv.org/abs/2609.33496)).
Code: [github.com/arnabsanyal/Chameleon](https://github.com/arnabsanyal/Chameleon) ·
Calibration artifacts: [arnabsanyal/chameleon](https://huggingface.co/arnabsanyal/chameleon)

## Contents

One split per Table 1 row, each holding 24,576 images as WebDataset `.tar` shards of about 1 GB. The PNGs are the
exact files that were scored. Each sample is `NNNNN.png` with a sidecar `NNNNN.json`:

| Field | Meaning |
|---|---|
| `index` | position in the evaluation list, 0 to 24,575 (also the file name) |
| `seed` | generator seed (`torch.Generator().manual_seed(seed)`), equal to `index` |
| `caption` | the COCO-2014 val caption used as the prompt |
| `coco_image_id`, `coco_caption_id` | ids in `captions_val2014.json` |
| `clip_score` | CLIP score of this image against `caption` |

The evaluation list is the first 24,576 captions of `captions_val2014.json` after `random.seed(42);
random.shuffle(captions)`, identical across all rows, so image `i` of every split answers the same prompt.

| Split prefix | Backbone | Resolution | Sampling |
|---|---|---|---|
| `sdxl_*` | [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 1024² | 50 steps, CFG 7.5 |
| `turbo_*` | [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) | 512² | 1 step, no CFG |
| `pixart_*` | [PixArt-α XL/2 1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | 1024² | 20 steps, CFG 4.5 |

`*_fp16` splits are the unquantized references; `*_chameleon_w8a8` / `*_chameleon_w4a8` are Chameleon.
Scores per split are in [`results/table1.json`](https://huggingface.co/arnabsanyal/chameleon/blob/main/results/table1.json).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("arnabsanyal/chameleon-coco-samples", split="sdxl_chameleon_w4a8", streaming=True)
sample = next(iter(ds))          # {"png": PIL.Image, "json": {...}, "__key__": "00000", ...}
```

To recompute FID, download one split and point clean-fid at the extracted PNGs:

```bash
hf download arnabsanyal/chameleon-coco-samples --repo-type dataset --include "sdxl_chameleon_w4a8/*" --local-dir samples
mkdir -p imgs && for t in samples/sdxl_chameleon_w4a8/*.tar; do tar -xf "$t" -C imgs --wildcards '*.png'; done
python src/paper_figures/score_24k.py imgs --label sdxl_chameleon_w4a8     # from the code repository
```

## Licence

Images are outputs of third-party models and follow those models' licences: CreativeML Open RAIL++-M for SDXL base
1.0 and PixArt-α, and the Stability AI licence of SDXL-Turbo for the `turbo_*` splits (non-commercial research use).
Captions come from [COCO](https://cocodataset.org/#termsofuse) (CC BY 4.0). Use the images for research and
evaluation.

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
