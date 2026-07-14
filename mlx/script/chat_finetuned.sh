#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../llm_models/model_extended"

if [[ ! -d "${MODEL_DIR}" ]]; then
	echo "Error: model directory not found: ${MODEL_DIR}" >&2
	exit 1
fi

mlx_lm.chat --model "${MODEL_DIR}"
