#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: run_frozen_7b.sh OUTPUT_ROOT SEED CONTAINER_IMAGE [CONFIG]}"
seed="${2:-17}"
container_image="${3:?usage: run_frozen_7b.sh OUTPUT_ROOT SEED CONTAINER_IMAGE [CONFIG]}"
config="${4:-configs/experiments/frozen_24gb.yaml}"
cd "$repo_root"
python -m jag_tree run "$config" --output-root "$output_root" --seed "$seed" --backend transformers --sandbox container --container-image "$container_image"
