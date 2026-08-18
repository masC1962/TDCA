#!/usr/bin/env bash
set -euo pipefail

env_file="${TDCA_ENV_FILE:-/root/.config/tdca/qwen.env}"
if [[ ! -r "$env_file" ]]; then
  echo "environment file is not readable: $env_file" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -z "${DASHSCOPE_API_KEY:-${LLM_API_KEY:-}}" ]]; then
  echo "neither DASHSCOPE_API_KEY nor LLM_API_KEY was loaded" >&2
  exit 2
fi

exec python -m tdca_research.run "$@"
