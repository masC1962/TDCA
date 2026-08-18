#!/usr/bin/env bash
set -euo pipefail

# Run all MuSiQue baselines plus TDCA with the same dataset/model and token budget.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_musique_all_qwen_1200.sh 15
#
# Tunables:
#   DATASET=...                    default: official MuSiQue-Ans dev split
#   DATASET_NAME=musique
#   TOKEN_BUDGET=1200              shared answer budget; also TDCA total generated-token cap
#   TDCA_MAX_STEPS=18              TDCA search steps
#   IRCOT_STEP_TOKEN_BUDGET=512    IRCoT intermediate step budget
#   TOP_K=5
#   DENSE_INDEX=indexes/musique_ans_dev_dense_index.npz
#   AUTO_BUILD_DENSE_INDEX=0       optionally build a global dense index if you need one

: "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${LLM_MODEL:?Please set LLM_MODEL, e.g. qwen-plus}"
: "${DASHSCOPE_API_KEY:?Please set DASHSCOPE_API_KEY}"

LIMIT="${1:-15}"
DEFAULT_MUSIQUE_DATA="musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"
DATASET="${DATASET:-$DEFAULT_MUSIQUE_DATA}"
DATASET_NAME="${DATASET_NAME:-musique}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-18}"
IRCOT_STEP_TOKEN_BUDGET="${IRCOT_STEP_TOKEN_BUDGET:-512}"
TOP_K="${TOP_K:-5}"
DENSE_INDEX="${DENSE_INDEX:-indexes/musique_ans_dev_dense_index.npz}"
AUTO_BUILD_DENSE_INDEX="${AUTO_BUILD_DENSE_INDEX:-0}"
RUN_TAG="${RUN_TAG:-musique_qwen_tok${TOKEN_BUDGET}_n${LIMIT}}"

if [[ ! -f "$DATASET" ]]; then
  echo "[musique-runner] dataset not found: $DATASET" >&2
  echo "[musique-runner] expected official data at: $DEFAULT_MUSIQUE_DATA" >&2
  exit 1
fi

mkdir -p outputs batch_outputs indexes

latest_dir() {
  local prefix="$1"
  ls -td "${prefix}"* 2>/dev/null | head -n 1 || true
}

print_summary() {
  local label="$1"
  local summary_path="$2"
  if [[ ! -f "$summary_path" ]]; then
    echo "[musique-runner] ${label}: summary missing: ${summary_path}"
    return
  fi
  python - "$label" "$summary_path" <<'PY'
import csv
import sys
label, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"[musique-runner] {label}: rows=0")
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
print(f"[musique-runner] {label}: rows={len(rows)} nonempty={nonempty} EM={em}/{len(rows)} acc={em/len(rows):.3f}{metric_text}")
PY
}

run_baseline() {
  local label="$1"
  local baseline="$2"
  local output_prefix="$3"
  shift 3

  echo "[musique-runner] running baseline=${label}"
  python scripts/run_baseline.py \
    --baseline "$baseline" \
    --dataset_path "$DATASET" \
    --dataset_name "$DATASET_NAME" \
    --limit "$LIMIT" \
    --llm_backend openai \
    --openai_base_url "$LLM_BASE_URL" \
    --served_model_name "$LLM_MODEL" \
    --openai_api_key "$DASHSCOPE_API_KEY" \
    --max_new_tokens_answer "$TOKEN_BUDGET" \
    --reasoning_effort none \
    --run_tag "$RUN_TAG" \
    "$@" \
    --output_dir "$output_prefix"
  local out
  out="$(latest_dir "${output_prefix}_")"
  echo "[musique-runner] ${label} output_dir=${out}"
  print_summary "$label" "${out}/summary.csv"
}

echo "[musique-runner] dataset=${DATASET}"
echo "[musique-runner] dataset_name=${DATASET_NAME}"
echo "[musique-runner] limit=${LIMIT}"
echo "[musique-runner] model=${LLM_MODEL}"
echo "[musique-runner] token_budget=${TOKEN_BUDGET}"

echo "[musique-runner] probing API"
python scripts/probe_llm_chat.py

run_baseline "closed_book" "closed_book" "outputs/musique_closed_book_qwen_api"
run_baseline "sparse_rag" "sparse_rag" "outputs/musique_sparse_rag_qwen_api" --top_k "$TOP_K"

if [[ -f "$DENSE_INDEX" ]]; then
  run_baseline "dense_rag" "dense_rag" "outputs/musique_dense_rag_qwen_api" --top_k "$TOP_K" --index_path "$DENSE_INDEX"
elif [[ "$AUTO_BUILD_DENSE_INDEX" == "1" ]]; then
  echo "[musique-runner] building dense index at ${DENSE_INDEX}"
  python scripts/build_dense_index.py \
    --dataset_path "$DATASET" \
    --dataset_name "$DATASET_NAME" \
    --output "$DENSE_INDEX"
  run_baseline "dense_rag" "dense_rag" "outputs/musique_dense_rag_qwen_api" --top_k "$TOP_K" --index_path "$DENSE_INDEX"
else
  run_baseline "dense_rag" "dense_rag" "outputs/musique_dense_rag_qwen_api" --top_k "$TOP_K"
fi

run_baseline "ircot_sparse" "ircot" "outputs/musique_ircot_sparse_qwen_api" \
  --top_k "$TOP_K" \
  --retriever_type sparse \
  --ircot_max_steps 3 \
  --ircot_step_max_new_tokens "$IRCOT_STEP_TOKEN_BUDGET"

echo "[musique-runner] running TDCA"
python tdca_batch_hotpotqa.py \
  --dataset "$DATASET" \
  --dataset_name "$DATASET_NAME" \
  --limit "$LIMIT" \
  --llm_backend openai \
  --openai_base_url "$LLM_BASE_URL" \
  --served_model_name "$LLM_MODEL" \
  --openai_api_key "$DASHSCOPE_API_KEY" \
  --scheduler_mode tdca \
  --scoring_mode hybrid \
  --max_steps "$TDCA_MAX_STEPS" \
  --max_total_generated_tokens "$TOKEN_BUDGET" \
  --max_new_tokens_answer "$TOKEN_BUDGET" \
  --reasoning_effort none \
  --output_root batch_outputs \
  --run_tag "$RUN_TAG"

tdca_out="$(latest_dir "batch_outputs/$(basename "$DATASET" .jsonl)_")"
echo "[musique-runner] TDCA output_dir=${tdca_out}"
print_summary "tdca" "${tdca_out}/summary.csv"

echo "[musique-runner] complete"
