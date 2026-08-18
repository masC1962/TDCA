#!/usr/bin/env bash
set -euo pipefail

# Overnight 100-example HotpotQA run: closed-book, sparse RAG, optional dense RAG,
# IRCoT-sparse, and TDCA under the same model/token budget.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_hotpot_all_qwen_1200_n100_overnight.sh
#
# Optional overrides:
#   DATASET=data/hotpot_dev_distractor_v1.jsonl
#   TOKEN_BUDGET=1200
#   TDCA_MAX_STEPS=18
#   TOP_K=5
#   AUTO_BUILD_DENSE_INDEX=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
export TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-18}"
export IRCOT_STEP_TOKEN_BUDGET="${IRCOT_STEP_TOKEN_BUDGET:-512}"
export TOP_K="${TOP_K:-5}"
export RUN_TAG="${RUN_TAG:-overnight_qwen_tok${TOKEN_BUDGET}_n100}"

bash "${SCRIPT_DIR}/run_hotpot_all_qwen_1200.sh" 100
