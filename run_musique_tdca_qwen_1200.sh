#!/usr/bin/env bash
set -euo pipefail

# Run TDCA only on MuSiQue with the same defaults as the all-in-one runner.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_musique_tdca_qwen_1200.sh 50
#
# Tunables:
#   DATASET=...                    default: official MuSiQue-Ans dev split
#   DATASET_NAME=musique
#   TOKEN_BUDGET=1200              TDCA total generated-token cap
#   TDCA_MAX_STEPS=18              TDCA search steps
#   TOP_K=5                        evidence retrieval top-k inside TDCA
#   RUN_TAG=...                    default: musique_tdca_qwen_tok${TOKEN_BUDGET}_n${LIMIT}

: "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${LLM_MODEL:?Please set LLM_MODEL, e.g. qwen-plus}"
: "${DASHSCOPE_API_KEY:?Please set DASHSCOPE_API_KEY}"

LIMIT="${1:-50}"
DEFAULT_MUSIQUE_DATA="musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"
DATASET="${DATASET:-$DEFAULT_MUSIQUE_DATA}"
DATASET_NAME="${DATASET_NAME:-musique}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-18}"
TOP_K="${TOP_K:-5}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
RUN_TAG="${RUN_TAG:-musique_tdca_qwen_tok${TOKEN_BUDGET}_n${LIMIT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-batch_outputs}"

if [[ ! -f "$DATASET" ]]; then
  echo "[musique-tdca] dataset not found: $DATASET" >&2
  echo "[musique-tdca] expected official data at: $DEFAULT_MUSIQUE_DATA" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

latest_dir() {
  local prefix="$1"
  ls -td "${prefix}"* 2>/dev/null | head -n 1 || true
}

print_summary() {
  local summary_path="$1"
  if [[ ! -f "$summary_path" ]]; then
    echo "[musique-tdca] summary missing: ${summary_path}"
    return
  fi
  python - "$summary_path" <<'PY'
import csv
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("[musique-tdca] rows=0")
    raise SystemExit
em = sum(int(float(r.get("exact_match") or 0)) for r in rows)
nonempty = sum(1 for r in rows if (r.get("pred") or "").strip())
metric_cols = ["soft_em", "answer_f1", "rougeL_f", "meteor", "title_hit"]
metrics = []
for col in metric_cols:
    vals = [float(r.get(col) or 0.0) for r in rows if col in r]
    if vals:
        metrics.append(f"{col}={sum(vals)/len(vals):.3f}")
metric_text = " " + " ".join(metrics) if metrics else ""
print(f"[musique-tdca] rows={len(rows)} nonempty={nonempty} EM={em}/{len(rows)} acc={em/len(rows):.3f}{metric_text}")
PY
}

echo "[musique-tdca] dataset=${DATASET}"
echo "[musique-tdca] dataset_name=${DATASET_NAME}"
echo "[musique-tdca] limit=${LIMIT}"
echo "[musique-tdca] model=${LLM_MODEL}"
echo "[musique-tdca] token_budget=${TOKEN_BUDGET}"
echo "[musique-tdca] max_steps=${TDCA_MAX_STEPS}"
echo "[musique-tdca] top_k=${TOP_K}"

echo "[musique-tdca] probing API"
python scripts/probe_llm_chat.py

echo "[musique-tdca] running TDCA"
python tdca_batch_hotpotqa.py \
  --dataset "$DATASET" \
  --dataset_name "$DATASET_NAME" \
  --limit "$LIMIT" \
  --llm_backend openai \
  --openai_base_url "$LLM_BASE_URL" \
  --served_model_name "$LLM_MODEL" \
  --openai_api_key "$DASHSCOPE_API_KEY" \
  --reasoning_effort "$REASONING_EFFORT" \
  --scheduler_mode tdca \
  --scoring_mode hybrid \
  --top_k "$TOP_K" \
  --max_steps "$TDCA_MAX_STEPS" \
  --max_total_generated_tokens "$TOKEN_BUDGET" \
  --max_new_tokens_answer "$TOKEN_BUDGET" \
  --output_root "$OUTPUT_ROOT" \
  --run_tag "$RUN_TAG"

tdca_out="$(latest_dir "${OUTPUT_ROOT}/$(basename "$DATASET" .jsonl)_")"
echo "[musique-tdca] output_dir=${tdca_out}"
print_summary "${tdca_out}/summary.csv"

echo "[musique-tdca] complete"
