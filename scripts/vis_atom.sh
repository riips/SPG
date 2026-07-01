#!/usr/bin/env bash

# Simple helper script to run SAE atom overlays.
# Adjust the variables below to match your environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

PYTHON_MODULE="${PYTHON_MODULE:-src.analysis_sae.sae_atom_overlays}"
ATOM_IDS="${ATOM_IDS:-3351}"
CKPT="${CKPT:-}"
DATASET="${DATASET:-mvtec}"
HIDDEN_DIM="${HIDDEN_DIM:-4096}"
TOPK="${TOPK:-32}"
INPUT_NORM="${INPUT_NORM:-none}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SAE_DATASET_ROOT="${SAE_DATASET_ROOT:-}"

if [[ -z "${CKPT}" ]]; then
    echo "Error: CKPT is required." >&2
    echo "Usage: CKPT=outputs/YYYY-MM-DD/HH-MM-SS_none_facebook_dinov3-vitl16/checkpoints/epoch_50.pth scripts/vis_atom.sh" >&2
    exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "Error: CKPT does not exist: ${CKPT}" >&2
    echo "Usage: CKPT=outputs/YYYY-MM-DD/HH-MM-SS_none_facebook_dinov3-vitl16/checkpoints/epoch_50.pth scripts/vis_atom.sh" >&2
    exit 1
fi

read -r -a ATOMS <<< "${ATOM_IDS}"

cmd=(
    python -m "${PYTHON_MODULE}" individual
    --atoms "${ATOMS[@]}"
    --ckpt "${CKPT}"
    --dataset "${DATASET}"
    --hidden-dim "${HIDDEN_DIM}"
    --topk "${TOPK}"
    --input-norm "${INPUT_NORM}"
)

if [[ -n "${CACHE_DIR:-}" ]]; then
    cmd+=(--cache-dir "${CACHE_DIR}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${SAE_DATASET_ROOT}" ]]; then
    cmd+=(--dataset-root "${SAE_DATASET_ROOT}")
fi

"${cmd[@]}"
