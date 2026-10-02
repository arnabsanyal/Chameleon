#!/usr/bin/env bash
# Upload a staged folder to the Hugging Face Hub (run `hf auth login` first).
#
#   bash huggingface/upload.sh model   hf_model      # from stage_model_repo.py
#   bash huggingface/upload.sh dataset hf_samples    # from pack_samples.py
#
# Repo names default to arnabsanyal/chameleon and arnabsanyal/chameleon-coco-samples; override them with
# CHAMELEON_HF_MODEL_REPO / CHAMELEON_HF_DATASET_REPO. Repos are created private; make them public on the Hub
# once you have checked them.
set -euo pipefail

kind="${1:?usage: upload.sh model|dataset <folder>}"
folder="${2:?usage: upload.sh model|dataset <folder>}"
PYTHON="${PYTHON:-python}"

case "$kind" in
  model)   repo="${CHAMELEON_HF_MODEL_REPO:-arnabsanyal/chameleon}" ;;
  dataset) repo="${CHAMELEON_HF_DATASET_REPO:-arnabsanyal/chameleon-coco-samples}" ;;
  *) echo "unknown kind: $kind" >&2; exit 1 ;;
esac

[ -f "$folder/README.md" ] || { echo "$folder/README.md missing; stage the folder first" >&2; exit 1; }
find "$folder" -name '*.tmp' | grep -q . && { echo "$folder has unfinished *.tmp shards" >&2; exit 1; }

"$PYTHON" - "$kind" "$repo" "$folder" <<'EOF'
import sys
from huggingface_hub import HfApi

kind, repo, folder = sys.argv[1:]
api = HfApi()
api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
if kind == "model":
    api.upload_folder(repo_id=repo, repo_type=kind, folder_path=folder,
                      commit_message="Chameleon calibration artifacts and Table 1 scores (v1.0-iclr2027)")
else:
    # Resumable, multi-commit upload for the ~150-250 GB of shards.
    api.upload_large_folder(repo_id=repo, repo_type=kind, folder_path=folder)
print(f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")
EOF
