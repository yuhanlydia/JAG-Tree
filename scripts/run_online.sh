#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: run_online.sh OUTPUT_ROOT SEED CONTAINER_IMAGE FORMAL_CONFIG}"
seed="${2:?usage: run_online.sh OUTPUT_ROOT SEED CONTAINER_IMAGE FORMAL_CONFIG}"
container_image="${3:?usage: run_online.sh OUTPUT_ROOT SEED CONTAINER_IMAGE FORMAL_CONFIG}"
config="${4:?usage: run_online.sh OUTPUT_ROOT SEED CONTAINER_IMAGE FORMAL_CONFIG}"
cd "$repo_root"
python -m jag_tree run "$config" --output-root "$output_root" --seed "$seed" --backend transformers --sandbox container --container-image "$container_image"
