"""Assemble the Hugging Face model repo (calibration artifacts + Table 1 scores) in a local folder.

    CHAMELEON_OUTPUT_ROOT=/path/to/runs python huggingface/stage_model_repo.py --out hf_model

The folder is ready for `bash huggingface/upload.sh model hf_model`. Nothing is uploaded here.
"""

import argparse
import hashlib
import json
import os
import shutil

from manifest import ARTIFACTS, NUM_IMAGES, OUTPUT_ROOT, ROW_SETS, score_file

HERE = os.path.dirname(os.path.abspath(__file__))

EXPECTED_MODEL = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "pixart-alpha": "PixArt-alpha/PixArt-XL-2-1024-MS",
}

# Only the paper's metrics leave the machine; run paths and scorer bookkeeping stay behind.
SCORE_KEYS = ("fid", "clip_mean", "clip_std", "n_images")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_artifacts(out):
    for dst, src in ARTIFACTS.items():
        src = os.path.join(OUTPUT_ROOT, src)
        with open(src) as f:
            cfg = json.load(f)
        backbone = dst.split("/")[0]
        bits = int(dst.split("/")[1][1])
        assert cfg["model_id"] == EXPECTED_MODEL[backbone], (src, cfg["model_id"])
        assert cfg["weight_bits"] == bits, (src, cfg["weight_bits"])
        assert cfg["num_buckets"] == 10, (src, cfg["num_buckets"])
        assert cfg.get("activation_lut"), f"{src} has no embedded activation LUT"
        os.makedirs(os.path.join(out, os.path.dirname(dst)), exist_ok=True)
        shutil.copyfile(src, os.path.join(out, dst))
        print(f"  {dst:48s} {len(cfg['layers']):4d} layers  <- {os.path.relpath(src, OUTPUT_ROOT)}")


def stage_scores(out):
    table = []
    for split, image_dir, backbone, method, bits in [r for rs in ROW_SETS.values() for r in rs]:
        path = score_file(image_dir)
        assert path, f"no coco_score.json for {image_dir}"
        with open(path) as f:
            s = json.load(f)
        assert s["n_images"] == NUM_IMAGES and s.get("n_missing", 0) == 0, (path, s["n_images"])
        assert "clip_fixed_date" in s, f"{path} predates the caption-alignment fix"
        row = {"split": split, "backbone": backbone, "method": method, "bits": bits}
        row.update({k: s[k] for k in SCORE_KEYS})
        table.append(row)
        print(f"  {split:24s} FID {s['fid']:6.2f}  CLIP {s['clip_mean']:5.2f}")
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    with open(os.path.join(out, "results", "table1.json"), "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging folder (created; must be empty or absent)")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} is not empty")
    os.makedirs(args.out, exist_ok=True)

    print(f"Calibration artifacts (from {OUTPUT_ROOT}):")
    stage_artifacts(args.out)
    print("Table 1 scores:")
    stage_scores(args.out)
    shutil.copyfile(os.path.join(HERE, "cards", "model_card.md"), os.path.join(args.out, "README.md"))

    sums = []
    for root, _, files in os.walk(args.out):
        for name in sorted(files):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, args.out)
            if rel not in ("README.md", "SHA256SUMS"):
                sums.append(f"{sha256(p)}  {rel}")
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sorted(sums, key=lambda l: l.split("  ")[1])) + "\n")
    print(f"Staged {len(sums)} files + README.md + SHA256SUMS in {args.out}")


if __name__ == "__main__":
    main()
