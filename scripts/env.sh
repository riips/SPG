export DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES=${DEVICE}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export SCRIPT_DIR
export REPO_ROOT
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/../datasets}"
export CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/features}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
