#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${OPENROUTER_BASE_URL:=https://yh.m7ai.com/v1}"
: "${OPENROUTER_MODEL:=gpt-5.4}"
: "${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY}"

cd /workspace/TDCA

echo "=== GPU check ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

echo "=== Running TDCA via OpenRouter API ==="
python main.py \
  --llm_backend openrouter \
  --openai_base_url "$OPENROUTER_BASE_URL" \
  --served_model_name "$OPENROUTER_MODEL" \
  --openai_api_key "$OPENROUTER_API_KEY" \
  --reasoning_effort none
