#!/usr/bin/env bash
# Download MixDQ's released mixed-precision configs for SDXL-Turbo.
#
# The MixDQ W4A8 baseline in Table 1 uses MixDQ's per-layer weight / activation
# bit-width assignment (weight_4.00.yaml averages ~4 bits/weight over a W2/W4/W8
# mix; act_8.00.yaml keeps activations at 8 bits). These files belong to MixDQ
# and are not redistributed here: this script fetches them from MixDQ's GitHub
# repository at a pinned commit and checks that they match the files used for
# the paper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT="4f6b32ad20d980494bcfecae50bfde5d17ed805c"
BASE="https://raw.githubusercontent.com/A-suozhang/MixDQ/${COMMIT}/kernels/cfgs"

declare -A SHA256=(
  [weight_4.00.yaml]="f5c5b75d3d4fe8feeef609f646318bf7bbb5a69cbfb1a04f65c0d578152e45e7"
  [act_8.00.yaml]="13a348c7cde35e389f34eba8b5efec751a057ccfa44d773bbe14126638b000c8"
)
declare -A SUBDIR=([weight_4.00.yaml]="weight" [act_8.00.yaml]="act")

for dest in "$ROOT/src/mixdq-lcm/generation/configs" "$ROOT/src/chameleon-lcm/generation/configs"; do
  mkdir -p "$dest"
  for f in "${!SHA256[@]}"; do
    curl -fsSL "${BASE}/${SUBDIR[$f]}/${f}" -o "$dest/$f"
    echo "${SHA256[$f]}  $dest/$f" | sha256sum --check --quiet
    echo "ok  ${dest#$ROOT/}/$f"
  done
done
