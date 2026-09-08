#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: run_smoke.sh OUTPUT_ROOT [SEED]}"
seed="${2:-7}"
cd "$repo_root"
python -m jag_tree run configs/experiments/smoke.yaml --output-root "$output_root" --seed "$seed" --backend fake --sandbox fake
