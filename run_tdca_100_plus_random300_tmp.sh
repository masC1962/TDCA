#!/usr/bin/env bash
set -euo pipefail

# TDCA overnight validation: first 100 HotpotQA examples + random 300 examples.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export LLM_MODEL="qwen-plus"
#   export DASHSCOPE_API_KEY="..."
#   bash run_tdca_100_plus_random300_tmp.sh
#
# Optional overrides:
#   DATASET=data/hotpot_dev_distractor_v1.jsonl
#   FIRST_COUNT=100
#   RANDOM_COUNT=300
#   RANDOM_SEED=20260504
#   ALLOW_RANDOM_OVERLAP=1
#   TOKEN_BUDGET=1200
#   TDCA_MAX_STEPS=18
#   OUTPUT_ROOT=batch_outputs

DATASET="${DATASET:-data/hotpot_dev_distractor_v1.jsonl}"
FIRST_COUNT="${FIRST_COUNT:-100}"
RANDOM_COUNT="${RANDOM_COUNT:-300}"
RANDOM_SEED="${RANDOM_SEED:-20260504}"
ALLOW_RANDOM_OVERLAP="${ALLOW_RANDOM_OVERLAP:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-batch_outputs}"
SAMPLE_ROOT="${SAMPLE_ROOT:-tmp/nightly_samples}"
LOG_ROOT="${LOG_ROOT:-tmp/nightly_logs}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-18}"
ANSWER_RESERVE="${ANSWER_RESERVE:-240}"
INTERMEDIATE_BUDGET_FRACTION="${INTERMEDIATE_BUDGET_FRACTION:-0.72}"
OPEN_GOAL_INTERMEDIATE_BUDGET_FRACTION="${OPEN_GOAL_INTERMEDIATE_BUDGET_FRACTION:-0.88}"
TOP_K="${TOP_K:-5}"
LLM_BACKEND="${LLM_BACKEND:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-${DASHSCOPE_BASE_URL:-}}}"
LLM_MODEL="${LLM_MODEL:-${SERVED_MODEL_NAME:-${DASHSCOPE_MODEL:-qwen-plus}}}"
LLM_API_KEY="${LLM_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}}}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"

if [[ "${LLM_BACKEND}" != "mock" ]]; then
  : "${LLM_BASE_URL:?Please set LLM_BASE_URL, e.g. https://dashscope.aliyuncs.com/compatible-mode/v1}"
  : "${LLM_API_KEY:?Please set DASHSCOPE_API_KEY or LLM_API_KEY}"
fi

mkdir -p "${OUTPUT_ROOT}" "${SAMPLE_ROOT}" "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_ROOT}/tdca_100_plus_random300_${STAMP}.log"

latest_batch_dir() {
  local dataset_path="$1"
  local name
  name="$(basename "${dataset_path}")"
  local stem="${name%.*}"
  ls -td "${OUTPUT_ROOT}/${stem}_"* 2>/dev/null | head -n 1 || true
}

print_summary() {
  local label="$1"
  local summary_path="$2"
  if [[ ! -f "${summary_path}" ]]; then
    echo "[nightly] ${label}: summary missing: ${summary_path}"
    return
  fi
  python - "${label}" "${summary_path}" <<'PY'
import csv
import sys

label, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"[nightly] {label}: rows=0")
    raise SystemExit(0)

def avg(col):
    vals = [float(r.get(col) or 0.0) for r in rows if col in r]
    return sum(vals) / len(vals) if vals else 0.0

em = sum(int(float(r.get("exact_match") or 0)) for r in rows)
nonempty = sum(1 for r in rows if (r.get("pred") or "").strip())
print(
    f"[nightly] {label}: rows={len(rows)} nonempty={nonempty} "
    f"EM={em}/{len(rows)} acc={em/len(rows):.3f} "
    f"soft_em={avg('soft_em'):.3f} f1={avg('answer_f1'):.3f} "
    f"steps={avg('steps'):.2f} calls={avg('llm_calls'):.2f} "
    f"gen_tokens={avg('generated_tokens'):.2f}"
)
PY
}

run_tdca() {
  local label="$1"
  local dataset="$2"
  local limit="$3"
  local run_tag="$4"

  echo "[nightly] running ${label}"
  python tdca_batch_hotpotqa.py \
    --dataset "${dataset}" \
    --dataset_name hotpotqa \
    --limit "${limit}" \
    --start_index 0 \
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
    --run_tag "${run_tag}"

  local out
  out="$(latest_batch_dir "${dataset}")"
  echo "[nightly] ${label} output_dir=${out}"
  print_summary "${label}" "${out}/summary.csv"
}

{
  echo "[nightly] started_at=${STAMP}"
  echo "[nightly] dataset=${DATASET}"
  echo "[nightly] backend=${LLM_BACKEND} model=${LLM_MODEL} token_budget=${TOKEN_BUDGET} max_steps=${TDCA_MAX_STEPS}"
  echo "[nightly] log=${LOG_PATH}"

  if [[ "${LLM_BACKEND}" != "mock" ]]; then
    echo "[nightly] probing API"
    python scripts/probe_llm_chat.py
  fi

  RANDOM_DATASET="${SAMPLE_ROOT}/hotpot_random${RANDOM_COUNT}_seed${RANDOM_SEED}.jsonl"
  SAMPLE_META="${RANDOM_DATASET}.meta.json"
  EXCLUDE_FIRST_100="true"
  if [[ "${ALLOW_RANDOM_OVERLAP}" == "1" ]]; then
    EXCLUDE_FIRST_100="false"
  fi

  python - "${DATASET}" "${RANDOM_DATASET}" "${SAMPLE_META}" "${RANDOM_COUNT}" "${RANDOM_SEED}" "${EXCLUDE_FIRST_100}" <<'PY'
import json
import random
import sys
from pathlib import Path

dataset, output, meta_path, count, seed, exclude_first = sys.argv[1:7]
count = int(count)
seed = int(seed)
exclude_first = exclude_first.lower() == "true"
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

start = 100 if exclude_first else 0
pool = list(range(start, len(items)))
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
    "exclude_first_100": exclude_first,
    "selected_indices": selected,
}
Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[nightly] random_sample={out} count={count} seed={seed} exclude_first_100={exclude_first}")
print(f"[nightly] random_meta={meta_path}")
PY

  run_tdca "tdca_first_${FIRST_COUNT}" "${DATASET}" "${FIRST_COUNT}" "tdca_patch_first${FIRST_COUNT}_tok${TOKEN_BUDGET}"
  run_tdca "tdca_random_${RANDOM_COUNT}" "${RANDOM_DATASET}" "${RANDOM_COUNT}" "tdca_patch_random${RANDOM_COUNT}_seed${RANDOM_SEED}_tok${TOKEN_BUDGET}"

  echo "[nightly] complete"
  echo "[nightly] random_dataset=${RANDOM_DATASET}"
  echo "[nightly] log=${LOG_PATH}"
} 2>&1 | tee "${LOG_PATH}"
