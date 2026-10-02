"""What goes to the Hugging Face Hub, and where it comes from.

Source paths are relative to CHAMELEON_OUTPUT_ROOT (the folder the generation scripts write to), so the same
manifest works on any machine that holds the original runs.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTPUT_ROOT = os.environ.get("CHAMELEON_OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs"))

MODEL_REPO = os.environ.get("CHAMELEON_HF_MODEL_REPO", "arnabsanyal/chameleon")
DATASET_REPO = os.environ.get("CHAMELEON_HF_DATASET_REPO", "arnabsanyal/chameleon-coco-samples")

# Calibration artifacts, keyed by their path in the model repo. Each file is the complete input to --load-weights:
# the SDXL and PixArt-alpha files embed the activation LUT next to the per-layer weight formats.
# SDXL-Turbo has no entry on purpose: its Chameleon weights are folded deterministically at load time on top of
# MixDQ's activation configs, which are fetched from MixDQ (third_party/mixdq/fetch_configs.sh), not redistributed.
ARTIFACTS = {
    "sdxl/w8a8/chameleon_sdxl_w8a8.json": "ablations/_calibration_cache/sdxl_w8a8_weights.pt",
    "sdxl/w4a8/chameleon_sdxl_w4a8.json": "ablations/_calibration_cache/sdxl_w4a8_weights.pt",
    "pixart-alpha/w8a8/chameleon_pixart_w8a8.json": "chameleon-dit/saved_configs/coco_eval_w8a8_10b.json",
    "pixart-alpha/w4a8/chameleon_pixart_w4a8.json": "chameleon-dit/saved_configs/coco_eval_w4a8_10b_iaw.json",
}

# Table 1 rows: (split name, image folder, backbone, method, bits). Image folders hold 00000.png ... 24575.png,
# where image i uses the i-th caption of the seed-42 shuffle of captions_val2014.json and generator seed i.
CHAMELEON_ROWS = [
    ("sdxl_chameleon_w8a8", "chameleon/coco_eval/w8a8", "SDXL", "Chameleon", "W8A8"),
    ("sdxl_chameleon_w4a8", "chameleon/coco_eval/w4a8", "SDXL", "Chameleon", "W4A8"),
    ("turbo_chameleon_w8a8", "chameleon-lcm/coco_eval_w8a8", "SDXL-Turbo", "Chameleon", "W8A8"),
    ("turbo_chameleon_w4a8", "chameleon-lcm/coco_eval_w4a8", "SDXL-Turbo", "Chameleon", "W4A8"),
    ("pixart_chameleon_w8a8", "chameleon-dit/coco_eval/w8a8_chameleon_dit_10b", "PixArt-alpha", "Chameleon", "W8A8"),
    ("pixart_chameleon_w4a8", "chameleon-dit/coco_eval/w4a8_chameleon_dit_10b_iaw", "PixArt-alpha", "Chameleon", "W4A8"),
]
FP16_ROWS = [
    ("sdxl_fp16", "coco_baseline/images", "SDXL", "FP16", "W16A16"),
    ("turbo_fp16", "lcm/coco_eval", "SDXL-Turbo", "FP16", "W16A16"),
    ("pixart_fp16", "pixart-alpha-dit/coco_eval", "PixArt-alpha", "FP16", "W16A16"),
]
BASELINE_ROWS = [
    ("sdxl_qdiffusion_w8a8", "quantized/coco_eval/qdiffusion/w8a8/images", "SDXL", "Q-Diffusion", "W8A8"),
    ("sdxl_qdiffusion_w4a8", "quantized/coco_eval/qdiffusion/w4a8/images", "SDXL", "Q-Diffusion", "W4A8"),
    ("sdxl_ptq4dm_w8a8", "quantized/coco_eval/ptq4dm/w8a8/images", "SDXL", "PTQ4DM", "W8A8"),
    ("sdxl_ptq4dm_w4a8", "quantized/coco_eval/ptq4dm/w4a8/images", "SDXL", "PTQ4DM", "W4A8"),
    ("turbo_mixdq_w8a8", "mixdq-lcm/coco_eval_w8a8", "SDXL-Turbo", "MixDQ", "W8A8"),
    ("turbo_mixdq_w4a8", "mixdq-lcm/coco_eval_w4a8", "SDXL-Turbo", "MixDQ", "W4A8"),
    ("pixart_qdit_w8a8", "q-dit/coco_eval/w8a8_g128", "PixArt-alpha", "Q-DiT", "W8A8"),
    ("pixart_qdit_w4a8", "q-dit/coco_eval/w4a8_g128", "PixArt-alpha", "Q-DiT", "W4A8"),
]
ROW_SETS = {
    "chameleon": CHAMELEON_ROWS,
    "fp16": FP16_ROWS,
    "baselines": BASELINE_ROWS,
}

NUM_IMAGES = 24576


def score_file(image_dir):
    """coco_score.json sits next to the images (some runs keep both in an images/ subfolder)."""
    for d in (image_dir, os.path.dirname(image_dir)):
        p = os.path.join(OUTPUT_ROOT, d, "coco_score.json")
        if os.path.isfile(p):
            return p
    return None


def clip_file(image_dir):
    for d in (image_dir, os.path.dirname(image_dir)):
        p = os.path.join(OUTPUT_ROOT, d, "clip_fixed.json")
        if os.path.isfile(p):
            return p
    return None


def rows(names):
    out = []
    for n in names:
        out += ROW_SETS[n]
    return out
