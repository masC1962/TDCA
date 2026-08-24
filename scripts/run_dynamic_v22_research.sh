#!/usr/bin/env bash
set -euo pipefail

stage="${1:-tests}"
dev_config="configs/dynamic_hypergraph_v22_qwen_development50.yaml"
smoke_config="configs/dynamic_hypergraph_v22_qwen_smoke20.yaml"
heldout_config="configs/dynamic_hypergraph_v22_qwen_heldout200.yaml"
dev_ledger="analysis_outputs/dynamic_v22_campaign/campaign_budget.json"

require_passing_gate() {
  local report="${TDCA_V22_GATE_REPORT:-}"
  if [[ -z "$report" || ! -r "$report" ]]; then
    echo "TDCA_V22_GATE_REPORT must name a readable passing v2.2 gate report" >&2
    exit 2
  fi
  PYTHONPATH="${PYTHONPATH:-}:src:." python - "$report" <<'PY'
import json
import sys
from pathlib import Path

from tdca_research.runtime import _source_tree_hash

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("schema_version") != "dynamic-hypergraph-v2.2-gate-evaluation-v1":
    raise SystemExit("v2.2 heldout gate report has the wrong schema")
if not report.get("passed") or not all(report.get("checks", {}).values()):
    raise SystemExit("v2.2 heldout gate remains closed")
recorded = str(report.get("evidence", {}).get("adaptive_source_tree_sha256", ""))
current = _source_tree_hash()
if not recorded or recorded != current:
    raise SystemExit("source tree changed after the passing development gate")
PY
}

case "$stage" in
  tests)
    PYTHONPATH="${PYTHONPATH:-}:src:." python -m pytest -q
    ;;
  campaign_status)
    if [[ ! -r "$dev_ledger" ]]; then
      echo '{"status":"not_started","provider_calls":0,"provider_reported_tokens":0}'
    else
      python - "$dev_ledger" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "campaign_id": value.get("campaign_id"),
    "status": value.get("status"),
    "limits": value.get("limits"),
    "usage": value.get("usage"),
    "pending_request_count": len(value.get("pending", {})),
    "last_stop_reason": value.get("last_stop_reason"),
}, indent=2, sort_keys=True))
PY
    fi
    ;;
  smoke20)
    bash scripts/run_qwen_experiment.sh --config "$smoke_config"
    ;;
  development50)
    bash scripts/run_qwen_experiment.sh --config "$dev_config" --allocator_mode adaptive_evc
    ;;
  uniform50)
    bash scripts/run_qwen_experiment.sh --config "$dev_config" --allocator_mode uniform
    ;;
  fixed50)
    bash scripts/run_qwen_experiment.sh --config "$dev_config" --allocator_mode fixed_order
    ;;
  budget_curve50)
    for point in "8 8000 4" "12 12000 6" "24 24000 12"; do
      read -r calls tokens retrievals <<<"$point"
      for mode in adaptive_evc uniform fixed_order; do
        bash scripts/run_qwen_experiment.sh \
          --config "$dev_config" --allocator_mode "$mode" \
          --max_llm_calls "$calls" --max_total_tokens "$tokens" \
          --max_retrieval_calls "$retrievals"
      done
    done
    ;;
  build_budget_curve)
    if [[ -z "${TDCA_V22_BUDGET_RUNS:-}" ]]; then
      echo "TDCA_V22_BUDGET_RUNS must contain space-separated run directories" >&2
      exit 2
    fi
    args=()
    for run_dir in $TDCA_V22_BUDGET_RUNS; do
      args+=(--run "$run_dir")
    done
    python scripts/build_dynamic_v22_budget_curve.py "${args[@]}" \
      --output "${TDCA_V22_BUDGET_CURVE_OUTPUT:-analysis_outputs/dynamic_v22_campaign/budget_curve.json}"
    ;;
  hard_gate)
    : "${V1_RUN:?V1_RUN is required}"
    : "${ADAPTIVE_RUN:?ADAPTIVE_RUN is required}"
    : "${UNIFORM_RUN:?UNIFORM_RUN is required}"
    : "${FIXED_RUN:?FIXED_RUN is required}"
    : "${REVISION_EVAL:?REVISION_EVAL is required}"
    : "${BUDGET_CURVE:?BUDGET_CURVE is required}"
    python scripts/evaluate_dynamic_v22_gate.py \
      --v1-run "$V1_RUN" --adaptive-run "$ADAPTIVE_RUN" \
      --control-run "$UNIFORM_RUN" --control-run "$FIXED_RUN" \
      --revision-eval "$REVISION_EVAL" --campaign-ledger "$dev_ledger" \
      --budget-curve "$BUDGET_CURVE" \
      --output "${GATE_OUTPUT:-analysis_outputs/dynamic_v22_campaign/hard_gate.json}"
    ;;
  heldout200)
    require_passing_gate
    heldout_ledger="analysis_outputs/dynamic_v22_campaign/heldout_campaign_budget.json"
    if [[ -e "$heldout_ledger" && -z "${HELDOUT_RESUME_DIR:-}" ]]; then
      echo "heldout campaign already started; refuse a second launch without HELDOUT_RESUME_DIR" >&2
      exit 2
    fi
    resume_args=()
    if [[ -n "${HELDOUT_RESUME_DIR:-}" ]]; then
      resume_args=(--resume_dir "$HELDOUT_RESUME_DIR")
    fi
    bash scripts/run_qwen_experiment.sh --config "$heldout_config" "${resume_args[@]}"
    ;;
  *)
    echo "usage: $0 {tests|campaign_status|smoke20|development50|uniform50|fixed50|budget_curve50|build_budget_curve|hard_gate|heldout200}" >&2
    exit 2
    ;;
esac
