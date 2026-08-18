#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/structured_tdca_qwen_tuning50_frozen.yaml}"

# Key mechanism ablations on exactly the same frozen 50 examples.
bash scripts/run_qwen_experiment.sh --config "$config" --memory_mode none
bash scripts/run_qwen_experiment.sh --config "$config" --scheduler greedy
bash scripts/run_qwen_experiment.sh --config "$config" --scheduler diffusion

# Oracle gap decomposition for separating retrieval and planning ceilings.
bash scripts/run_qwen_experiment.sh --config "$config" --oracle_evidence
bash scripts/run_qwen_experiment.sh --config "$config" --oracle_decomposition
bash scripts/run_qwen_experiment.sh --config "$config" --oracle_evidence --oracle_decomposition
