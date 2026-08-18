#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/structured_tdca_qwen_tuning50_frozen.yaml}"

bash scripts/run_qwen_experiment.sh --config "$config" --method dense_rag
bash scripts/run_qwen_experiment.sh --config "$config" --method hybrid_rag
