#!/usr/bin/env bash
set -euo pipefail

stage="${1:-tests}"

case "$stage" in
  tests)
    PYTHONPATH="${PYTHONPATH:-}:src:." python -m pytest -q tests_research
    ;;
  smoke20)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_smoke20.yaml
    ;;
  development50)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_development50.yaml
    ;;
  budget_curve50)
    for point in "8 8000 4" "12 12000 6" "16 16000 8" "24 24000 12"; do
      read -r calls tokens retrievals <<<"$point"
      bash scripts/run_qwen_experiment.sh \
        --config configs/dynamic_hypergraph_v2_qwen_development50.yaml \
        --max_llm_calls "$calls" --max_total_tokens "$tokens" \
        --max_retrieval_calls "$retrievals"
    done
    ;;
  heldout200)
    python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("configs/dynamic_v2_hard_gate.json").read_text(encoding="utf-8"))
if gate.get("status") != "open":
    raise SystemExit("Dynamic v2 heldout gate is closed; every machine-verified hard gate must pass first")
PY
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_heldout200.yaml
    ;;
  cross_smoke20)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_hotpot_smoke20.yaml
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_2wiki_smoke20.yaml
    ;;
  cross_heldout200)
    python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("configs/dynamic_v2_hard_gate.json").read_text(encoding="utf-8"))
if gate.get("status") != "open":
    raise SystemExit("Dynamic v2 heldout gate is closed; cross-dataset heldout runs are forbidden")
PY
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_hotpot_heldout200.yaml
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_2wiki_heldout200.yaml
    ;;
  *)
    echo "usage: $0 {tests|smoke20|development50|budget_curve50|heldout200|cross_smoke20|cross_heldout200}" >&2
    exit 2
    ;;
esac
