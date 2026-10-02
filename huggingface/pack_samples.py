"""Pack the Table 1 image folders into WebDataset shards for the Hugging Face dataset repo.

    CHAMELEON_OUTPUT_ROOT=/path/to/runs CHAMELEON_DATA_ROOT=/path/to/data \
        python huggingface/pack_samples.py --out hf_samples --rows chameleon fp16

Each split becomes <out>/<split>/<split>-NNNNN.tar. A sample is the original PNG (bit-exact, so FID is unchanged)
plus a JSON sidecar: {"index", "seed", "caption", "coco_image_id", "coco_caption_id", "clip_score"}.
Re-running skips finished shards, so an interrupted pack can be resumed.
"""

import argparse
import io
import json
import os
import random
import tarfile

from manifest import NUM_IMAGES, OUTPUT_ROOT, REPO_ROOT, clip_file, rows

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("CHAMELEON_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def coco_order(captions_json):
    """The generators' caption order: the seed-42 shuffle of all val2014 captions.

    random.shuffle's permutation depends only on the list length, so shuffling the annotation records reproduces
    the order the generators applied to the bare caption strings, and keeps the COCO ids with them.
    """
    with open(captions_json) as f:
        anns = json.load(f)["annotations"]
    random.seed(42)
    random.shuffle(anns)
    return anns[:NUM_IMAGES]


def clip_scores(image_dir, anns):
    path = clip_file(image_dir)
    if path is None:
        return None
    with open(path) as f:
        scores = {s["index"]: s for s in json.load(f)["scores"]}
    for i in (0, 1, NUM_IMAGES // 2, NUM_IMAGES - 1):
        assert scores[i]["prompt"] == anns[i]["caption"].strip(), f"{path}: caption order mismatch at {i}"
    return {i: s["score"] for i, s in scores.items()}


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def pack_split(split, image_dir, anns, out, shard_bytes):
    src = os.path.join(OUTPUT_ROOT, image_dir)
    names = [f"{i:05d}.png" for i in range(NUM_IMAGES)]
    missing = [n for n in names if not os.path.isfile(os.path.join(src, n))]
    assert not missing, f"{src}: {len(missing)} images missing, e.g. {missing[:3]}"
    clip = clip_scores(image_dir, anns)

    # Fix shard boundaries from file sizes first, so a resumed run produces the same shards.
    shards, cur, size = [], [], 0
    for i, n in enumerate(names):
        s = os.path.getsize(os.path.join(src, n))
        if cur and size + s > shard_bytes:
            shards.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += s
    shards.append(cur)

    os.makedirs(os.path.join(out, split), exist_ok=True)
    for k, idxs in enumerate(shards):
        dst = os.path.join(out, split, f"{split}-{k:05d}.tar")
        if os.path.isfile(dst):
            continue
        with tarfile.open(dst + ".tmp", "w", format=tarfile.USTAR_FORMAT) as tar:
            for i in idxs:
                with open(os.path.join(src, names[i]), "rb") as f:
                    add(tar, f"{i:05d}.png", f.read())
                meta = {
                    "index": i,
                    "seed": i,
                    "caption": anns[i]["caption"].strip(),
                    "coco_image_id": anns[i]["image_id"],
                    "coco_caption_id": anns[i]["id"],
                    "clip_score": None if clip is None else round(clip[i], 4),
                }
                add(tar, f"{i:05d}.json", json.dumps(meta).encode())
        os.replace(dst + ".tmp", dst)
        print(f"  {split}: shard {k + 1}/{len(shards)} ({len(idxs)} images)", flush=True)
    return len(shards)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", nargs="+", default=["chameleon", "fp16"], choices=["chameleon", "fp16", "baselines"])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these split names")
    ap.add_argument("--shard-gb", type=float, default=1.0)
    ap.add_argument("--captions", default=os.path.join(DATA_ROOT, "coco", "annotations", "captions_val2014.json"))
    args = ap.parse_args()

    anns = coco_order(args.captions)
    for split, image_dir, *_ in rows(args.rows):
        if args.only and split not in args.only:
            continue
        print(f"{split} <- {image_dir}", flush=True)
        n = pack_split(split, image_dir, anns, args.out, int(args.shard_gb * (1 << 30)))
        print(f"  {split}: {n} shards done", flush=True)

    # The card's front matter lists every split packed so far, so the Hub viewer and `datasets` see all of them.
    splits = sorted(d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)))
    files = "\n".join(f"      - split: {s}\n        path: {s}/*.tar" for s in splits)
    with open(os.path.join(HERE, "cards", "dataset_card.md")) as f:
        card = f.read().replace("{{DATA_FILES}}", files)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(card)


if __name__ == "__main__":
    main()
