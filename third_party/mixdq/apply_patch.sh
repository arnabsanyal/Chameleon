#!/usr/bin/env bash
# Patch MixDQ's Hugging Face custom pipeline (nics-efc/MixDQ) for Chameleon.
#
# The published pipeline.py only supports W8A8 with MixDQ's CUDA kernels. This
# patch adds (1) a W4A8 / W8A8 fake-quant path that consumes MixDQ's shipped
# scales and per-layer YAML configs, (2) a stub for missing mixdq_extension
# kernels, and (3) the opt-in Chameleon weight fold (MIXDQ_CHAMELEON_WEIGHTS=1)
# used by src/chameleon-lcm. Both SDXL-Turbo rows of Table 1 need it.
#
# Run once after installing, and again after clearing the Hugging Face cache.
# Safe to re-run: files that are already patched are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV="74e2a7c97d080189633c66b68e3f41cb789d28c6"   # pinned: the patch targets this revision

PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
"$PYTHON" - "$HERE/mixdq_pipeline.patch" "$REV" <<'PY'
import glob, os, shutil, subprocess, sys, tempfile
from huggingface_hub import hf_hub_download
patch, rev = sys.argv[1], sys.argv[2]
MARK = "MIXDQ_CHAMELEON_WEIGHTS"

src = hf_hub_download("nics-efc/MixDQ", "pipeline.py", revision=rev)   # snapshot path (often a symlink)
with tempfile.TemporaryDirectory() as td:
    work = os.path.join(td, "pipeline.py")
    shutil.copyfile(os.path.realpath(src), work)
    if MARK not in open(work).read():
        subprocess.run(["patch", "--quiet", work, patch], check=True)
    patched = open(work).read()

modules = os.environ.get("HF_MODULES_CACHE", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "modules"))
targets = [src] + glob.glob(os.path.join(modules, "diffusers_modules", "**", "nics-efc--MixDQ", rev, "pipeline.py"), recursive=True)
for t in targets:
    if os.path.exists(t) and MARK in open(t).read():
        print(f"already patched: {t}"); continue
    if os.path.islink(t):
        os.unlink(t)                      # replace the snapshot symlink, never the shared blob
    with open(t, "w") as f:
        f.write(patched)
    print(f"patched: {t}")
PY
