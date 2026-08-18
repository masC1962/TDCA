#!/usr/bin/env bash
set -euo pipefail

# Random HotpotQA comparison runner: Sparse RAG + IRCoT + TDCA.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   SEEDS=20260504,42 RANDOM_COUNT=100 bash run_hotpot_random_compare_tmp.sh
#
# Defaults are intentionally small enough for quick regression. For a stronger
# validation pass use RANDOM_COUNT=300 with one or more seeds.

DATASET="${DATASET:-data/hotpot_dev_distractor_v1.jsonl}"
DATASET_NAME="${DATASET_NAME:-hotpotqa}"
RANDOM_COUNT="${RANDOM_COUNT:-100}"
SEEDS="${SEEDS:-20260504}"
EXCLUDE_FIRST="${EXCLUDE_FIRST:-100}"
ALLOW_RANDOM_OVERLAP="${ALLOW_RANDOM_OVERLAP:-0}"
SAMPLE_ROOT="${SAMPLE_ROOT:-tmp/random_compare_samples}"
OUTPUT_ROOT="${OUTPUT_ROOT:-batch_outputs}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-outputs}"
LOG_ROOT="${LOG_ROOT:-tmp/random_compare_logs}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-18}"
ANSWER_RESERVE="${ANSWER_RESERVE:-240}"
INTERMEDIATE_BUDGET_FRACTION="${INTERMEDIATE_BUDGET_FRACTION:-0.72}"
OPEN_GOAL_INTERMEDIATE_BUDGET_FRACTION="${OPEN_GOAL_INTERMEDIATE_BUDGET_FRACTION:-0.88}"
TOP_K="${TOP_K:-5}"
IRCOT_MAX_STEPS="${IRCOT_MAX_STEPS:-3}"
IRCOT_STEP_TOKEN_BUDGET="${IRCOT_STEP_TOKEN_BUDGET:-512}"
LLM_BACKEND="${LLM_BACKEND:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-${DASHSCOPE_BASE_URL:-}}}"
LLM_MODEL="${LLM_MODEL:-${SERVED_MODEL_NAME:-${DASHSCOPE_MODEL:-qwen-plus}}}"
LLM_API_KEY="${LLM_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}}}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
RUN_SPARSE_RAG="${RUN_SPARSE_RAG:-1}"
RUN_IRCOT="${RUN_IRCOT:-1}"
RUN_TDCA="${RUN_TDCA:-1}"

if [[ "${LLM_BACKEND}" != "mock" ]]; then
  : "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
  : "${LLM_API_KEY:?Please set DASHSCOPE_API_KEY or LLM_API_KEY}"
fi

mkdir -p "${SAMPLE_ROOT}" "${OUTPUT_ROOT}" "${BASELINE_OUTPUT_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_ROOT}/random_compare_${STAMP}.log"

latest_prefix_dir() {
  local prefix="$1"
  ls -td "${prefix}_"* 2>/dev/null | head -n 1 || true
}

latest_batch_dir() {
  local dataset_path="$1"
  local name stem
  name="$(basename "${dataset_path}")"
  stem="${name%.*}"
  ls -td "${OUTPUT_ROOT}/${stem}_"* 2>/dev/null | head -n 1 || true
}

print_summary() {
  local label="$1"
  local summary_path="$2"
  if [[ ! -f "${summary_path}" ]]; then
    echo "[compare] ${label}: summary missing: ${summary_path}"
    return
  fi
  python - "${label}" "${summary_path}" <<'PY'
import csv
import sys

label, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"[compare] {label}: rows=0")
    raise SystemExit(0)

def avg(col):
    vals = [float(r.get(col) or 0.0) for r in rows if col in r]
    return sum(vals) / len(vals) if vals else 0.0

em = sum(int(float(r.get("exact_match") or 0)) for r in rows)
nonempty = sum(1 for r in rows if (r.get("pred") or "").strip())
print(
    f"[compare] {label}: rows={len(rows)} nonempty={nonempty} "
    f"EM={em}/{len(rows)} acc={em/len(rows):.3f} "
    f"soft_em={avg('soft_em'):.3f} f1={avg('answer_f1'):.3f} "
    f"calls={avg('llm_calls'):.2f} gen_tokens={avg('generated_tokens'):.2f}"
)
PY
}

make_random_dataset() {
  local seed="$1"
  local out="${SAMPLE_ROOT}/hotpot_random${RANDOM_COUNT}_seed${seed}.jsonl"
  local meta="${out}.meta.json"
  local exclude_first="${EXCLUDE_FIRST}"
  if [[ "${ALLOW_RANDOM_OVERLAP}" == "1" ]]; then
    exclude_first="0"
  fi
  python - "${DATASET}" "${out}" "${meta}" "${RANDOM_COUNT}" "${seed}" "${exclude_first}" <<'PY'
import json
import random
import sys
from pathlib import Path

dataset, output, meta_path, count, seed, exclude_first = sys.argv[1:7]
count = int(count)
seed = int(seed)
exclude_first = int(exclude_first)
path = Path(dataset)
text = path.read_text(encoding="utf-8")
if path.suffix.lower() == ".json":
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported JSON shape: {path}")
else:
    items = [json.loads(line) for line in text.splitlines() if line.strip()]

pool = list(range(max(0, exclude_first), len(items)))
if count > len(pool):
    raise ValueError(f"Requested {count} samples, but pool has only {len(pool)} items")

rng = random.Random(seed)
selected = sorted(rng.sample(pool, count))
out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for idx in selected:
        row = dict(items[idx])
        row["_tdca_source_index"] = idx
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

meta = {
    "source_dataset": str(path),
    "output_dataset": str(out),
    "count": count,
    "seed": seed,
    "exclude_first": exclude_first,
    "selected_indices": selected,
}
Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
}

run_sparse_rag() {
  local dataset="$1"
  local seed="$2"
  local prefix="${BASELINE_OUTPUT_ROOT}/hotpot_sparse_rag_random_seed${seed}"
  echo "[compare] running sparse_rag seed=${seed}"
  python scripts/run_baseline.py \
    --baseline sparse_rag \
    --dataset_path "${dataset}" \
    --dataset_name "${DATASET_NAME}" \
    --limit "${RANDOM_COUNT}" \
    --top_k "${TOP_K}" \
    --llm_backend "${LLM_BACKEND}" \
    --openai_base_url "${LLM_BASE_URL}" \
    --served_model_name "${LLM_MODEL}" \
    --openai_api_key "${LLM_API_KEY}" \
    --max_new_tokens_answer "${TOKEN_BUDGET}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    --run_tag "random${RANDOM_COUNT}_seed${seed}_tok${TOKEN_BUDGET}" \
    --output_dir "${prefix}"
  local out
  out="$(latest_prefix_dir "${prefix}")"
  echo "[compare] sparse_rag output_dir=${out}"
  print_summary "sparse_rag_seed_${seed}" "${out}/summary.csv"
}

run_ircot() {
  local dataset="$1"
  local seed="$2"
  local prefix="${BASELINE_OUTPUT_ROOT}/hotpot_ircot_sparse_random_seed${seed}"
  echo "[compare] running ircot seed=${seed}"
  python scripts/run_baseline.py \
    --baseline ircot \
    --dataset_path "${dataset}" \
    --dataset_name "${DATASET_NAME}" \
    --limit "${RANDOM_COUNT}" \
    --top_k "${TOP_K}" \
    --retriever_type sparse \
    --ircot_max_steps "${IRCOT_MAX_STEPS}" \
    --ircot_step_max_new_tokens "${IRCOT_STEP_TOKEN_BUDGET}" \
    --llm_backend "${LLM_BACKEND}" \
    --openai_base_url "${LLM_BASE_URL}" \
    --served_model_name "${LLM_MODEL}" \
    --openai_api_key "${LLM_API_KEY}" \
    --max_new_tokens_answer "${TOKEN_BUDGET}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    --run_tag "random${RANDOM_COUNT}_seed${seed}_tok${TOKEN_BUDGET}" \
    --output_dir "${prefix}"
  local out
  out="$(latest_prefix_dir "${prefix}")"
  echo "[compare] ircot output_dir=${out}"
  print_summary "ircot_seed_${seed}" "${out}/summary.csv"
}

run_tdca() {
  local dataset="$1"
  local seed="$2"
  echo "[compare] running tdca seed=${seed}"
  python tdca_batch_hotpotqa.py \
    --dataset "${dataset}" \
    --dataset_name "${DATASET_NAME}" \
    --limit "${RANDOM_COUNT}" \
    --algorithm tdca \
    --baseline tdca \
    --llm_backend "${LLM_BACKEND}" \
    --openai_base_url "${LLM_BASE_URL}" \
    --served_model_name "${LLM_MODEL}" \
    --openai_api_key "${LLM_API_KEY}" \
    --scheduler_mode tdca \
    --scoring_mode hybrid \
    --max_steps "${TDCA_MAX_STEPS}" \
    --max_total_generated_tokens "${TOKEN_BUDGET}" \
    --max_new_tokens_answer "${TOKEN_BUDGET}" \
    --answer_synthesis_reserve_tokens "${ANSWER_RESERVE}" \
    --intermediate_generation_budget_fraction "${INTERMEDIATE_BUDGET_FRACTION}" \
    --open_goal_intermediate_budget_fraction "${OPEN_GOAL_INTERMEDIATE_BUDGET_FRACTION}" \
    --top_k "${TOP_K}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    --output_root "${OUTPUT_ROOT}" \
    --run_tag "random${RANDOM_COUNT}_seed${seed}_tok${TOKEN_BUDGET}"
  local out
  out="$(latest_batch_dir "${dataset}")"
  echo "[compare] tdca output_dir=${out}"
  print_summary "tdca_seed_${seed}" "${out}/summary.csv"
}

{
  echo "[compare] started_at=${STAMP}"
  echo "[compare] dataset=${DATASET}"
  echo "[compare] random_count=${RANDOM_COUNT} seeds=${SEEDS} exclude_first=${EXCLUDE_FIRST}"
  echo "[compare] backend=${LLM_BACKEND} model=${LLM_MODEL} token_budget=${TOKEN_BUDGET}"
  echo "[compare] log=${LOG_PATH}"

  if [[ "${LLM_BACKEND}" != "mock" ]]; then
    echo "[compare] probing API"
    python scripts/probe_llm_chat.py
  fi

  IFS=',' read -ra seed_list <<< "${SEEDS}"
  for raw_seed in "${seed_list[@]}"; do
    seed="$(echo "${raw_seed}" | xargs)"
    if [[ -z "${seed}" ]]; then
      continue
    fi
    random_dataset="$(make_random_dataset "${seed}")"
    echo "[compare] random_dataset=${random_dataset}"
    if [[ "${RUN_SPARSE_RAG}" == "1" ]]; then
      run_sparse_rag "${random_dataset}" "${seed}"
    fi
    if [[ "${RUN_IRCOT}" == "1" ]]; then
      run_ircot "${random_dataset}" "${seed}"
    fi
    if [[ "${RUN_TDCA}" == "1" ]]; then
      run_tdca "${random_dataset}" "${seed}"
    fi
  done

  echo "[compare] complete"
} 2>&1 | tee "${LOG_PATH}"
