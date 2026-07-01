#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

echo "DATASET_ROOT=${DATASET_ROOT}"
echo "Generating MVTec AD metadata..."
python src/generate_dataset_json/mvtec.py

echo "Generating VisA metadata..."
python src/generate_dataset_json/visa.py

echo "Done. meta.json files were generated under DATASET_ROOT."
