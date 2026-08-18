#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_musique_baselines_qwen_1200.sh 15

: "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
: "${LLM_MODEL:?Please set LLM_MODEL, e.g. qwen-plus}"
: "${DASHSCOPE_API_KEY:?Please set DASHSCOPE_API_KEY}"

LIMIT="${1:-15}"
DEFAULT_MUSIQUE_DATA="musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"
DATASET="${DATASET:-$DEFAULT_MUSIQUE_DATA}"
DATASET_NAME="${DATASET_NAME:-musique}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TOP_K="${TOP_K:-5}"
DENSE_INDEX="${DENSE_INDEX:-indexes/musique_ans_dev_dense_index.npz}"
AUTO_BUILD_DENSE_INDEX="${AUTO_BUILD_DENSE_INDEX:-0}"
RUN_TAG="${RUN_TAG:-musique_qwen_tok${TOKEN_BUDGET}_n${LIMIT}}"

if [[ ! -f "$DATASET" ]]; then
  echo "[musique-baseline] dataset not found: $DATASET" >&2
  exit 1
fi

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

  echo "[musique-baseline] running ${label}"
  python scripts/run_baseline.py "$@"
  echo "[musique-baseline] ${label} output_dir=$(latest_output_dir "${output_prefix}")"
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
  --run_tag "$RUN_TAG"
)

run_baseline_and_print "closed_book" "outputs/musique_closed_book_qwen_api" \
  --baseline closed_book \
  "${COMMON_ARGS[@]}" \
  --output_dir outputs/musique_closed_book_qwen_api

run_baseline_and_print "sparse_rag" "outputs/musique_sparse_rag_qwen_api" \
  --baseline sparse_rag \
  "${COMMON_ARGS[@]}" \
  --top_k "$TOP_K" \
  --output_dir outputs/musique_sparse_rag_qwen_api

if [[ -f "$DENSE_INDEX" ]]; then
  run_baseline_and_print "dense_rag" "outputs/musique_dense_rag_qwen_api" \
    --baseline dense_rag \
    "${COMMON_ARGS[@]}" \
    --top_k "$TOP_K" \
    --index_path "$DENSE_INDEX" \
    --output_dir outputs/musique_dense_rag_qwen_api
elif [[ "$AUTO_BUILD_DENSE_INDEX" == "1" ]]; then
  echo "[musique-baseline] building dense index at $DENSE_INDEX"
  python scripts/build_dense_index.py \
    --dataset_path "$DATASET" \
    --dataset_name "$DATASET_NAME" \
    --output "$DENSE_INDEX"
  run_baseline_and_print "dense_rag" "outputs/musique_dense_rag_qwen_api" \
    --baseline dense_rag \
    "${COMMON_ARGS[@]}" \
    --top_k "$TOP_K" \
    --index_path "$DENSE_INDEX" \
    --output_dir outputs/musique_dense_rag_qwen_api
else
  run_baseline_and_print "dense_rag" "outputs/musique_dense_rag_qwen_api" \
    --baseline dense_rag \
    "${COMMON_ARGS[@]}" \
    --top_k "$TOP_K" \
    --output_dir outputs/musique_dense_rag_qwen_api
fi

run_baseline_and_print "ircot_sparse" "outputs/musique_ircot_sparse_qwen_api" \
  --baseline ircot \
  "${COMMON_ARGS[@]}" \
  --top_k "$TOP_K" \
  --retriever_type sparse \
  --ircot_max_steps 3 \
  --ircot_step_max_new_tokens 512 \
  --output_dir outputs/musique_ircot_sparse_qwen_api
