#!/usr/bin/env bash
set -euo pipefail

gate="configs/cross_dataset_validation_gate.json"
python - "$gate" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("status") != "open":
    raise SystemExit(
        "cross-dataset validation-200 is closed: "
        + gate.get("reason", "tuning gate has not passed")
    )
PY

# This block is unreachable while the versioned gate is closed. Keeping the
# frozen commands makes a future, explicitly reviewed gate opening reproducible.
bash scripts/test_all.sh
for config in \
  configs/structured_tdca_qwen_hotpot_validation200_frozen.yaml \
  configs/structured_tdca_qwen_2wiki_validation200_frozen.yaml
do
  bash scripts/run_qwen_experiment.sh --config "$config"
  bash scripts/run_qwen_experiment.sh --config "$config" --method ircot --retriever bm25
done
