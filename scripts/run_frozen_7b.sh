#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: run_frozen_7b.sh OUTPUT_ROOT SEED CONTAINER_IMAGE_OR_trusted-local [CONFIG]}"
seed="${2:-17}"
container_image="${3:?usage: run_frozen_7b.sh OUTPUT_ROOT SEED CONTAINER_IMAGE_OR_trusted-local [CONFIG]}"
config="${4:-configs/experiments/frozen_24gb.yaml}"
cd "$repo_root"
python_bin="${PYTHON:-python3}"
if [[ "$container_image" == "trusted-local" ]]; then
    "$python_bin" -m jag_tree run "$config" --output-root "$output_root" --seed "$seed" --backend transformers --sandbox trusted-local --allow-pilot-predictor
else
    "$python_bin" -m jag_tree run "$config" --output-root "$output_root" --seed "$seed" --backend transformers --sandbox container --container-image "$container_image"
fi
