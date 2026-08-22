#!/usr/bin/env bash
set -euo pipefail

stage="${1:-tests}"

load_qwen_env() {
  local env_file="${TDCA_ENV_FILE:-/root/.config/tdca/qwen.env}"
  if [[ ! -r "$env_file" ]]; then
    echo "environment file is not readable: $env_file" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  if [[ -z "${DASHSCOPE_API_KEY:-${LLM_API_KEY:-}}" ]]; then
    echo "neither DASHSCOPE_API_KEY nor LLM_API_KEY was loaded" >&2
    exit 2
  fi
}

require_gate_report() {
  local report="${TDCA_V2_GATE_REPORT:-}"
  if [[ -z "$report" ]]; then
    echo "TDCA_V2_GATE_REPORT must name the machine-verified passing gate report" >&2
    exit 2
  fi
  python - "$report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("schema_version") != "dynamic-hypergraph-v2-gate-evaluation-v2" or not report.get("passed"):
    raise SystemExit("Dynamic v2 heldout gate is closed: supplied report is absent, stale, or failing")
PY
}

case "$stage" in
  tests)
    PYTHONPATH="${PYTHONPATH:-}:src:." python -m pytest -q tests_research tests
    ;;
  smoke20)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_smoke20.yaml
    ;;
  development50)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_development50.yaml
    ;;
  matched_allocators50)
    for mode in adaptive_evc uniform fixed_order; do
      bash scripts/run_qwen_experiment.sh \
        --config configs/dynamic_hypergraph_v2_qwen_development50.yaml \
        --allocator_mode "$mode"
    done
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
    require_gate_report
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_heldout200.yaml
    ;;
  cross_smoke20)
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_hotpot_smoke20.yaml
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_2wiki_smoke20.yaml
    ;;
  cross_heldout200)
    require_gate_report
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_hotpot_heldout200.yaml
    bash scripts/run_qwen_experiment.sh --config configs/dynamic_hypergraph_v2_qwen_2wiki_heldout200.yaml
    ;;
  revision_development)
    load_qwen_env
    python scripts/evaluate_revision_suite.py predict \
      --split development \
      --output "${REVISION_OUTPUT:-research_outputs/dynamic_v2_revision_development_predictions.json}"
    ;;
  revision_evaluation)
    load_qwen_env
    python scripts/evaluate_revision_suite.py predict \
      --split evaluation \
      --output "${REVISION_OUTPUT:-research_outputs/dynamic_v2_revision_evaluation_predictions.json}"
    ;;
  score_revision)
    : "${REVISION_PREDICTIONS:?REVISION_PREDICTIONS is required}"
    python scripts/evaluate_revision_suite.py score \
      --predictions "$REVISION_PREDICTIONS" \
      --output "${REVISION_SCORE_OUTPUT:-research_outputs/dynamic_v2_revision_score.json}"
    ;;
  hard_gate)
    : "${V1_RUN:?V1_RUN is required}"
    : "${V2_RUN:?V2_RUN is required}"
    : "${UNIFORM_RUN:?UNIFORM_RUN is required}"
    : "${FIXED_RUN:?FIXED_RUN is required}"
    : "${REVISION_EVAL:?REVISION_EVAL is required}"
    : "${CAMPAIGN_LEDGER:?CAMPAIGN_LEDGER is required}"
    python scripts/evaluate_dynamic_v2_gate.py \
      --v1-run "$V1_RUN" --v2-run "$V2_RUN" \
      --control-run "$UNIFORM_RUN" --control-run "$FIXED_RUN" \
      --revision-eval "$REVISION_EVAL" --campaign-ledger "$CAMPAIGN_LEDGER" \
      --output "${GATE_OUTPUT:-research_outputs/dynamic_v2_hard_gate.json}"
    ;;
  *)
    echo "usage: $0 {tests|smoke20|development50|matched_allocators50|budget_curve50|revision_development|revision_evaluation|score_revision|hard_gate|heldout200|cross_smoke20|cross_heldout200}" >&2
    exit 2
    ;;
esac
