#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-5}"
DATASET="${2:-data/hotpot_dev_distractor_v1.jsonl}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/TDCA}"

: "${OPENROUTER_BASE_URL:=https://yh.m7ai.com/v1}"
: "${OPENROUTER_MODEL:=gpt-5.4}"
: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY}"

cd "$PROJECT_ROOT"

python tdca_batch_hotpotqa.py \
  --dataset "$DATASET" \
  --limit "$LIMIT" \
  --project_root "$PROJECT_ROOT" \
  --llm_backend openrouter \
  --openai_base_url "$OPENROUTER_BASE_URL" \
  --served_model_name "$OPENROUTER_MODEL" \
  --openai_api_key "$OPENROUTER_API_KEY" \
  --reasoning_effort none \
  --scheduler_mode tdca \
  --scoring_mode hybrid \
  --max_total_generated_tokens 1200
