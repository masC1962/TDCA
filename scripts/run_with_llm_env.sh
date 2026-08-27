#!/usr/bin/env bash
set -euo pipefail

env_file="${HARA_LLM_ENV_FILE:-/root/.config/tdca/qwen.env}"
if [[ ! -f "${env_file}" ]]; then
  echo "LLM environment file not found: ${env_file}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

if [[ -z "${LLM_API_KEY:-${DASHSCOPE_API_KEY:-}}" ]]; then
  echo "LLM_API_KEY or DASHSCOPE_API_KEY is missing from ${env_file}" >&2
  exit 2
fi

exec "$@"
