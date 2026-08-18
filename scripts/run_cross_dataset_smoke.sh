#!/usr/bin/env bash
set -euo pipefail

# Pre-registered order: main Dense track then controlled IRCoT on identical IDs.
# Runs are sequential so API contention does not confound latency measurements.
configs=(
  configs/structured_tdca_qwen_hotpot_smoke20_frozen.yaml
  configs/structured_tdca_qwen_2wiki_smoke20_frozen.yaml
)

bash scripts/test_all.sh
python scripts/verify_hotpot_scorer_parity.py >/dev/null
python scripts/verify_2wiki_scorer_parity.py >/dev/null
python scripts/audit_research_datasets.py --output research_outputs/dataset_audit.json >/dev/null

for config in "${configs[@]}"; do
  bash scripts/run_qwen_experiment.sh --config "$config"
  bash scripts/run_qwen_experiment.sh --config "$config" --method ircot --retriever bm25
done
