#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/structured_tdca_qwen_validation200_frozen.yaml}"

python -m pytest tests_research/test_frozen_manifests.py tests_research/test_compare.py -q

mkdir -p research_outputs/validation200_launcher_logs
started="$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_qwen_experiment.sh --config "$config" \
  2>&1 | tee "research_outputs/validation200_launcher_logs/${started}_structured.log"
bash scripts/run_qwen_experiment.sh --config "$config" --method ircot --retriever bm25 \
  2>&1 | tee "research_outputs/validation200_launcher_logs/${started}_ircot.log"
