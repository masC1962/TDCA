#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/workspace/TDCA}"
dataset_file="${2:?dataset file is required}"
save_dir="${3:?save directory is required}"
shared_cache_dir="${4:?shared cache directory is required}"
output="${5:?output artifact is required}"
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

for variable in dataset_file save_dir shared_cache_dir output; do
  value="${!variable}"
  if [[ "$value" != /* ]]; then
    printf -v "$variable" '%s/%s' "$repo_root" "$value"
  fi
done

cd "$repo_root/external_repos/HippoRAG"
exec .venv_controlled/bin/python "$repo_root/external_baselines/hipporag_distractor_adapter.py" \
  --dataset_file "$dataset_file" \
  --save_dir "$save_dir" \
  --shared_cache_dir "$shared_cache_dir" \
  --output "$output" \
  --llm_base_url "${LLM_BASE_URL:?LLM_BASE_URL is required}" \
  --llm_name "${LLM_MODEL:?LLM_MODEL is required}"
