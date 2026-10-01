# MixDQ integration

The SDXL-Turbo rows of Table 1 (the MixDQ baseline and Chameleon's few-step path) run on top of
[MixDQ](https://github.com/A-suozhang/MixDQ) (Zhao et al., ECCV 2024), which ships its quantized SDXL-Turbo as the
Hugging Face custom pipeline [`nics-efc/MixDQ`](https://huggingface.co/nics-efc/MixDQ).

| File | What it does |
|---|---|
| `mixdq_pipeline.patch` | Our changes to MixDQ's `pipeline.py`: a W4A8 / W8A8 fake-quant path that consumes MixDQ's shipped scales and per-layer configs, a fallback when MixDQ's CUDA kernels are unavailable, and the opt-in Chameleon weight fold (`MIXDQ_CHAMELEON_WEIGHTS=1`, set by `src/chameleon-lcm`). |
| `apply_patch.sh` | Downloads `pipeline.py` at the pinned revision `74e2a7c97d080189633c66b68e3f41cb789d28c6`, applies the patch, and installs the result in the Hugging Face cache. Safe to re-run. |
| `fetch_configs.sh` | Downloads MixDQ's mixed-precision configs (`weight_4.00.yaml`, `act_8.00.yaml`) from MixDQ's GitHub at commit `4f6b32ad20d980494bcfecae50bfde5d17ed805c` and verifies their SHA-256 against the files used for the paper. |

Run both once after installing:

```bash
bash third_party/mixdq/apply_patch.sh
bash third_party/mixdq/fetch_configs.sh
```

## Licenses

The `nics-efc/MixDQ` model repository, which contains the `pipeline.py` this patch modifies, is released under the
MIT License. MixDQ's GitHub repository does not state a license, so its configs are downloaded from the original
source rather than redistributed here. MixDQ's code and configs remain the property of their authors.
