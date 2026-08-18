#!/usr/bin/env bash
set -euo pipefail

: "${OPENROUTER_BASE_URL:=https://yh.m7ai.com/v1}"
: "${OPENROUTER_MODEL:=gpt-5.4}"
: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY}"

QUERY="${1:-What is the birth city of the director of the movie Inception?}"

cd /workspace/TDCA
python main.py \
  --llm_backend openrouter \
  --openai_base_url "$OPENROUTER_BASE_URL" \
  --served_model_name "$OPENROUTER_MODEL" \
  --openai_api_key "$OPENROUTER_API_KEY" \
  --reasoning_effort none \
  --query "$QUERY"
