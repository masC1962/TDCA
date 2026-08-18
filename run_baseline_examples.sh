#!/usr/bin/env bash
set -euo pipefail

: "${OPENROUTER_BASE_URL:=https://yh.m7ai.com/v1}"
: "${OPENROUTER_MODEL:=gpt-5.4}"
: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY}"

COMMON_ARGS=(
  --dataset_path data/hotpotqa_subset_50.jsonl
  --dataset_name hotpotqa
  --llm_backend openrouter
  --openai_base_url "$OPENROUTER_BASE_URL"
  --openai_api_key "$OPENROUTER_API_KEY"
  --served_model_name "$OPENROUTER_MODEL"
  --max_new_tokens_answer 1200
  --reasoning_effort none
)

python scripts/run_baseline.py \
  --baseline closed_book \
  "${COMMON_ARGS[@]}" \
  --output_dir outputs/hotpot_closed_book_api

python scripts/run_baseline.py \
  --baseline sparse_rag \
  "${COMMON_ARGS[@]}" \
  --top_k 5 \
  --output_dir outputs/hotpot_sparse_rag_api

python scripts/build_dense_index.py \
  --dataset_path data/hotpotqa_subset_50.jsonl \
  --dataset_name hotpotqa \
  --output indexes/hotpot_dense_index.npz

python scripts/run_baseline.py \
  --baseline dense_rag \
  "${COMMON_ARGS[@]}" \
  --index_path indexes/hotpot_dense_index.npz \
  --top_k 5 \
  --output_dir outputs/hotpot_dense_rag_api

python scripts/run_baseline.py \
  --baseline ircot \
  "${COMMON_ARGS[@]}" \
  --retriever_type sparse \
  --top_k 5 \
  --ircot_max_steps 3 \
  --ircot_step_max_new_tokens 512 \
  --output_dir outputs/hotpot_ircot_sparse_api
