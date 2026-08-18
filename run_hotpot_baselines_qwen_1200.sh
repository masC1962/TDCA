#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_hotpot_baselines_qwen_1200.sh 15

: "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${LLM_MODEL:?Please set LLM_MODEL, e.g. qwen-plus}"
: "${DASHSCOPE_API_KEY:?Please set DASHSCOPE_API_KEY}"

LIMIT="${1:-15}"
DATASET="${DATASET:-data/hotpot_dev_distractor_v1.jsonl}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TOP_K="${TOP_K:-5}"
DENSE_INDEX="${DENSE_INDEX:-indexes/hotpot_dev_dense_index.npz}"
AUTO_BUILD_DENSE_INDEX="${AUTO_BUILD_DENSE_INDEX:-0}"

mkdir -p outputs indexes

latest_output_dir() {
  local prefix="$1"
  local latest=""
  latest=$(ls -td "${prefix}"_* 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
  else
    printf '%s\n' "${prefix}"
  fi
}

run_baseline_and_print() {
  local label="$1"
  local output_prefix="$2"
  shift 2

  echo "[baseline-runner] running ${label}"
  python scripts/run_baseline.py "$@"
  echo "[baseline-runner] ${label} output_dir=$(latest_output_dir "${output_prefix}")"
}

COMMON_ARGS=(
  --dataset_path "$DATASET"
  --dataset_name "$DATASET_NAME"
  --limit "$LIMIT"
  --llm_backend openai
  --openai_base_url "$LLM_BASE_URL"
  --served_model_name "$LLM_MODEL"
  --openai_api_key "$DASHSCOPE_API_KEY"
  --max_new_tokens_answer "$TOKEN_BUDGET"
  --reasoning_effort none
)

run_baseline_and_print "closed_book" "outputs/hotpot_closed_book_qwen_api" \
  --baseline closed_book \
  "${COMMON_ARGS[@]}" \
  --output_dir outputs/hotpot_closed_book_qwen_api

run_baseline_and_print "sparse_rag" "outputs/hotpot_sparse_rag_qwen_api" \
  --baseline sparse_rag \
  "${COMMON_ARGS[@]}" \
  --top_k "$TOP_K" \
  --output_dir outputs/hotpot_sparse_rag_qwen_api

if [[ -f "$DENSE_INDEX" ]]; then
  run_baseline_and_print "dense_rag" "outputs/hotpot_dense_rag_qwen_api" \
    --baseline dense_rag \
    "${COMMON_ARGS[@]}" \
    --top_k "$TOP_K" \
    --index_path "$DENSE_INDEX" \
    --output_dir outputs/hotpot_dense_rag_qwen_api
elif [[ "$AUTO_BUILD_DENSE_INDEX" == "1" ]]; then
  echo "[baseline-runner] building dense index at $DENSE_INDEX"
  python scripts/build_dense_index.py \
    --dataset_path "$DATASET" \
    --dataset_name "$DATASET_NAME" \
    --output "$DENSE_INDEX"
  run_baseline_and_print "dense_rag" "outputs/hotpot_dense_rag_qwen_api" \
    --baseline dense_rag \
    "${COMMON_ARGS[@]}" \
    --top_k "$TOP_K" \
    --index_path "$DENSE_INDEX" \
    --output_dir outputs/hotpot_dense_rag_qwen_api
else
  echo "[baseline-runner] skipping dense_rag because dense index is missing: $DENSE_INDEX"
  echo "[baseline-runner] build it first with: python scripts/build_dense_index.py --dataset_path \"$DATASET\" --dataset_name \"$DATASET_NAME\" --output \"$DENSE_INDEX\""
fi

run_baseline_and_print "ircot_sparse" "outputs/hotpot_ircot_sparse_qwen_api" \
  --baseline ircot \
  "${COMMON_ARGS[@]}" \
  --top_k "$TOP_K" \
  --retriever_type sparse \
  --ircot_max_steps 3 \
  --ircot_step_max_new_tokens 512 \
  --output_dir outputs/hotpot_ircot_sparse_qwen_api
