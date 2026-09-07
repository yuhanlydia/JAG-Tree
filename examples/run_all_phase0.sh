#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PARENT="${1:-${REPO_ROOT}/runs}"
SEED="${2:-101}"

cd "${REPO_ROOT}"
python -m coding_opsd run-all --profile smoke --seed "${SEED}" --output "${OUTPUT_PARENT}"
