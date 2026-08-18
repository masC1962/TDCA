#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-10}"
DATASET="${2:-data/hotpot_dev_distractor_v1.jsonl}"

export LLM_BACKEND="${LLM_BACKEND:-openrouter}"
export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://yh.m7ai.com/v1}"
export OPENROUTER_MODEL="${OPENROUTER_MODEL:-gpt-5.4}"
export OPENROUTER_APP_NAME="${OPENROUTER_APP_NAME:-TDCA}"
export REASONING_EFFORT="${REASONING_EFFORT:-none}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

cd /workspace/TDCA
python tdca_batch_hotpotqa.py \
  --dataset "$DATASET" \
  --limit "$LIMIT" \
  --llm_backend openrouter \
  --openai_base_url "$OPENROUTER_BASE_URL" \
  --served_model_name "$OPENROUTER_MODEL" \
  --openai_api_key "$OPENROUTER_API_KEY" \
  --openrouter_app_name "$OPENROUTER_APP_NAME" \
  --reasoning_effort "$REASONING_EFFORT" \
  ${OPENROUTER_SITE_URL:+--openrouter_site_url "$OPENROUTER_SITE_URL"}
