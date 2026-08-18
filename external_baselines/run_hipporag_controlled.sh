#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/workspace/TDCA}"
dataset="${2:-sample}"
save_dir="${3:-outputs/controlled_qwen_canary}"
force_rebuild="${4:-true}"
env_file="${TDCA_LLM_ENV_FILE:-/root/.config/tdca/qwen.env}"

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

api_key="${LLM_API_KEY:-${DASHSCOPE_API_KEY:-}}"
if [[ -z "$api_key" ]]; then
  echo "LLM_API_KEY or DASHSCOPE_API_KEY is required" >&2
  exit 2
fi
export OPENAI_API_KEY="$api_key"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

cd "$repo_root/external_repos/HippoRAG"
exec .venv_controlled/bin/python main.py \
  --dataset "$dataset" \
  --rag_type hipporag \
  --llm_base_url "${LLM_BASE_URL:?LLM_BASE_URL is required}" \
  --llm_name "${LLM_MODEL:?LLM_MODEL is required}" \
  --embedding_name Transformers/sentence-transformers/all-MiniLM-L6-v2 \
  --embedding_batch_size 8 \
  --force_index_from_scratch "$force_rebuild" \
  --force_openie_from_scratch "$force_rebuild" \
  --save_dir "$save_dir"
