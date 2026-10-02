# Releasing artifacts to the Hugging Face Hub

The tooling that builds the two Hub repositories linked from the main README. You need the original runs
(`CHAMELEON_OUTPUT_ROOT`), COCO captions (`CHAMELEON_DATA_ROOT`) and `huggingface_hub` (`hf auth login`).

| Hub repo | Built by | Contents | Size |
|---|---|---|---|
| [`arnabsanyal/chameleon`](https://huggingface.co/arnabsanyal/chameleon) (model) | `stage_model_repo.py` | calibration artifacts for SDXL and PixArt-α (W8A8, W4A8), Table 1 scores, card | ~36 MB |
| [`arnabsanyal/chameleon-coco-samples`](https://huggingface.co/datasets/arnabsanyal/chameleon-coco-samples) (dataset) | `pack_samples.py` | the 24,576 scored COCO images per Table 1 row, as WebDataset shards | ~235 GB (Chameleon + FP16) |

`manifest.py` lists every file and image folder that goes out; the cards are in `cards/`.

## Steps

```bash
export CHAMELEON_OUTPUT_ROOT=/path/to/runs CHAMELEON_DATA_ROOT=/path/to/data
cd huggingface

# 1. Model repo: copy and check the artifacts, collect the scores, write README.md and SHA256SUMS
python stage_model_repo.py --out ../hf_model

# 2. Confirm each artifact reproduces its released images (GPU, a few minutes each)
for f in ../hf_model/{sdxl,pixart-alpha}/w?a8/*.json; do bash verify_artifacts.sh "$f" 8; done

# 3. Dataset repo: pack the Chameleon and FP16 rows (~1 GB shards; rerun to resume)
python pack_samples.py --out ../hf_samples --rows chameleon fp16      # add `baselines` for all 17 rows (~490 GB)

# 4. Upload (both repos are created private; flip them to public on the Hub after a look)
bash upload.sh model   ../hf_model
bash upload.sh dataset ../hf_samples
```

What stays out on purpose:
- **No SDXL-Turbo artifact.** Chameleon's weights are folded at load time over MixDQ's activation configs, which are
  fetched from MixDQ and not redistributed.
- **No base-model weights or COCO images.** They are downloaded from their own sources under their own licences.
- **Only the configs behind Table 1.** Exploratory calibration variants and bucket-count ablation LUTs are left out.
- **Only the Chameleon and FP16 sample rows by default.** The baseline-method rows are regenerable with
  `src/q-baselines`, `src/mixdq-lcm` and `src/q-dit`, and would double the size.

The SDXL artifacts were saved by a later calibration run than the one that generated the SDXL Table 1 images, so
step 2 is what ties them to the table. If an artifact reports `MISMATCH`, regenerate that row from the artifact and
re-score it before uploading, or upload it without claiming it reproduces the table.
