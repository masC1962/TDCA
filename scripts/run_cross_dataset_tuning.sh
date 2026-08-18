#!/usr/bin/env bash
set -euo pipefail

# Run only after both methods have completed smoke-20 without infrastructure
# failures. The pre-registered order matches the smoke launcher and all calls
# remain sequential to avoid API-rate and latency interference.
configs=(
  configs/structured_tdca_qwen_hotpot_tuning50_frozen.yaml
  configs/structured_tdca_qwen_2wiki_tuning50_frozen.yaml
)

gate="research_outputs/cross_dataset_smoke_gate.json"
if [[ ! -r "$gate" ]]; then
  echo "cross-dataset tuning is closed: missing audited smoke gate $gate" >&2
  exit 2
fi
python - "$gate" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("status") != "open" or gate.get("expected_count") != 20:
    raise SystemExit(f"cross-dataset tuning is closed: {gate.get('reason', 'smoke gate is not open')}")
audits = gate.get("artifact_audits", [])
if len(audits) != 4 or any(
    not row.get("verified")
    or row.get("count") != 20
    or row.get("infrastructure_failures") != 0
    for row in audits
):
    raise SystemExit("cross-dataset tuning is closed: four clean smoke artifact audits are required")
PY

bash scripts/test_all.sh
python scripts/verify_hotpot_scorer_parity.py >/dev/null
python scripts/verify_2wiki_scorer_parity.py >/dev/null
python scripts/audit_research_datasets.py --output research_outputs/dataset_audit.json >/dev/null

for config in "${configs[@]}"; do
  bash scripts/run_qwen_experiment.sh --config "$config"
  bash scripts/run_qwen_experiment.sh --config "$config" --method ircot --retriever bm25
done
