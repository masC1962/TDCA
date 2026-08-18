#!/usr/bin/env bash
set -euo pipefail

stage="${1:-preflight}"
config="${CONFIG:-configs/structured_tdca_qwen.yaml}"

case "$stage" in
  preflight)
    python scripts/research_preflight.py
    ;;
  stage1)
    bash scripts/test_all.sh
    ;;
  smoke20)
    bash scripts/run_qwen_experiment.sh --config "$config" --split smoke
    ;;
  tuning50)
    echo "Run only after smoke20 review. Use the official full-data disjoint split manifest." >&2
    bash scripts/run_qwen_experiment.sh --config "$config" --split tuning
    ;;
  validation200)
    echo "Run only after freezing the tuning configuration." >&2
    bash scripts/run_qwen_experiment.sh --config "$config" --split validation
    ;;
  final1000)
    python - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("configs/stage5_gate.json").read_text(encoding="utf-8"))
if gate.get("status") != "open":
    raise SystemExit(f"Stage 5 is closed: {gate.get('reason', 'Stage 4 has not passed')}")
PY
    bash scripts/run_qwen_experiment.sh --config "$config" --split final
    ;;
  *)
    echo "usage: $0 {preflight|stage1|smoke20|tuning50|validation200|final1000}" >&2
    exit 2
    ;;
esac
