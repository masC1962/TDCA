#!/usr/bin/env bash
set -euo pipefail

stage="${1:-tests}"

case "$stage" in
  tests)
    python -m pytest -q tests_research
    ;;
  smoke20)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_qwen_smoke20.yaml
    ;;
  development50)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_qwen_development50.yaml
    ;;
  controls50)
    bash scripts/run_qwen_experiment.sh --config configs/baseline_structured_qwen_dynamic_development50.yaml
    bash scripts/run_qwen_experiment.sh --config configs/baseline_ircot_qwen_dynamic_development50.yaml
    ;;
  ablations50)
    for ablation in A1 A2 A3 A4 A5 A6; do
      bash scripts/run_qwen_experiment.sh \
        --config configs/dynamic_hypergraph_qwen_development50.yaml \
        --dynamic_ablation "$ablation"
    done
    ;;
  budget_curve50)
    for point in "8 8000 4" "12 12000 6" "16 16000 8" "24 24000 12"; do
      read -r calls tokens retrievals <<<"$point"
      bash scripts/run_qwen_experiment.sh \
        --config configs/dynamic_hypergraph_qwen_development50.yaml \
        --max_llm_calls "$calls" --max_total_tokens "$tokens" \
        --max_retrieval_calls "$retrievals"
    done
    ;;
  heldout200)
    python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("configs/dynamic_heldout_gate.json").read_text(encoding="utf-8"))
if gate.get("status") != "open":
    raise SystemExit(f"Dynamic heldout gate is closed: {gate.get('reason', 'development is incomplete')}")
PY
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_qwen_heldout200.yaml
    ;;
  *)
    echo "usage: $0 {tests|smoke20|development50|controls50|ablations50|budget_curve50|heldout200}" >&2
    exit 2
    ;;
esac
