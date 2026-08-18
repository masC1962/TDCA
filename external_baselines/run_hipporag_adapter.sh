#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/workspace/TDCA}"
dataset="${2:?dataset name is required}"
save_dir="${3:?save directory is required}"
output="${4:?output artifact is required}"
force_rebuild="${5:-false}"
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

if [[ "$output" != /* ]]; then
  output="$repo_root/$output"
fi

cd "$repo_root/external_repos/HippoRAG"
args=(
  "$repo_root/external_baselines/hipporag_controlled_adapter.py"
  --dataset "$dataset"
  --save_dir "$save_dir"
  --output "$output"
  --llm_base_url "${LLM_BASE_URL:?LLM_BASE_URL is required}"
  --llm_name "${LLM_MODEL:?LLM_MODEL is required}"
)
if [[ "$force_rebuild" == "true" ]]; then
  args+=(--force_rebuild)
fi
exec .venv_controlled/bin/python "${args[@]}"
