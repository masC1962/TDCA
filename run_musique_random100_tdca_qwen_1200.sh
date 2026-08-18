#!/usr/bin/env bash
set -euo pipefail

# TDCA-only random MuSiQue runner.
#
# This uses the same deterministic sampling scheme and SAMPLE_ROOT as
# run_musique_random100_all_qwen_1200.sh. With the same DATASET, RANDOM_COUNT,
# and SEEDS, both scripts run on the exact same jsonl sample file.
#
# Usage:
#   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   export DASHSCOPE_API_KEY="..."
#   bash run_musique_random100_tdca_qwen_1200.sh
#
# To match a previous all-baseline run, set the same overrides, especially:
#   SEEDS=520 RANDOM_COUNT=100 TOKEN_BUDGET=1200 TOP_K=5

DEFAULT_MUSIQUE_DATA="musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl"
DATASET="${DATASET:-$DEFAULT_MUSIQUE_DATA}"
DATASET_NAME="${DATASET_NAME:-musique}"
RANDOM_COUNT="${RANDOM_COUNT:-100}"
SEEDS="${SEEDS:-520}"
SAMPLE_ROOT="${SAMPLE_ROOT:-tmp/musique_random_samples}"
OUTPUT_ROOT="${OUTPUT_ROOT:-batch_outputs}"
LOG_ROOT="${LOG_ROOT:-tmp/musique_random_logs}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1200}"
TDCA_MAX_STEPS="${TDCA_MAX_STEPS:-22}"
TDCA_RESERVE_TOKENS="${TDCA_RESERVE_TOKENS:-96}"
TDCA_INTERMEDIATE_BUDGET_FRACTION="${TDCA_INTERMEDIATE_BUDGET_FRACTION:-0.86}"
TDCA_OPEN_GOAL_BUDGET_FRACTION="${TDCA_OPEN_GOAL_BUDGET_FRACTION:-0.94}"
ENABLE_FINAL_CHAIN_BUFFER="${ENABLE_FINAL_CHAIN_BUFFER:-0}"
ENABLE_SCORE_BASED_FINAL_ADMISSION="${ENABLE_SCORE_BASED_FINAL_ADMISSION:-0}"
FINAL_CHAIN_SCORE_THRESHOLD="${FINAL_CHAIN_SCORE_THRESHOLD:-0.72}"
ENABLE_TERMINAL_CHAIN_CLOSURE="${ENABLE_TERMINAL_CHAIN_CLOSURE:-0}"
ENABLE_TCC_FINAL_AUDIT="${ENABLE_TCC_FINAL_AUDIT:-0}"
TCC_FINAL_AUDIT_MODE="${TCC_FINAL_AUDIT_MODE:-audit_only}"
TCC_RERANK_POLICY="${TCC_RERANK_POLICY:-longhop_or_weak}"
TCC_SCORE_THRESHOLD="${TCC_SCORE_THRESHOLD:-0.70}"
TCC_MIN_PATH_COMPLETENESS="${TCC_MIN_PATH_COMPLETENESS:-0.45}"
TCC_MIN_DEPENDENCY_CLOSURE="${TCC_MIN_DEPENDENCY_CLOSURE:-0.45}"
TCC_MIN_LAST_HOP_ENTAILMENT="${TCC_MIN_LAST_HOP_ENTAILMENT:-0.50}"
TCC_MIN_TERMINALITY="${TCC_MIN_TERMINALITY:-0.60}"
TCC_MIN_ROOT_CONSISTENCY="${TCC_MIN_ROOT_CONSISTENCY:-0.55}"
TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP="${TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP:-0.30}"
TCC_MIN_ROOT_CONSISTENCY_SHORTHOP="${TCC_MIN_ROOT_CONSISTENCY_SHORTHOP:-0.45}"
TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP="${TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP:-0.45}"
TCC_MIN_TERMINALITY_SHORTHOP="${TCC_MIN_TERMINALITY_SHORTHOP:-0.55}"
TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP="${TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP:-0.45}"
TCC_MIN_ROOT_CONSISTENCY_LONGHOP="${TCC_MIN_ROOT_CONSISTENCY_LONGHOP:-0.55}"
TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP="${TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP:-0.50}"
TCC_MIN_TERMINALITY_LONGHOP="${TCC_MIN_TERMINALITY_LONGHOP:-0.60}"
ENABLE_TCC_VERIFIED_PROMOTION="${ENABLE_TCC_VERIFIED_PROMOTION:-0}"
TCC_PROMOTION_POLICY="${TCC_PROMOTION_POLICY:-empty_only_strict}"
TCC_PROMOTION_MIN_HOP="${TCC_PROMOTION_MIN_HOP:-3}"
ALLOW_STRICT_2HOP_PROMOTION="${ALLOW_STRICT_2HOP_PROMOTION:-0}"
TCC_PROMOTION_SCORE_THRESHOLD="${TCC_PROMOTION_SCORE_THRESHOLD:-0.70}"
TCC_PROMOTION_MIN_TERMINALITY="${TCC_PROMOTION_MIN_TERMINALITY:-0.60}"
TCC_PROMOTION_MIN_ROOT_CONSISTENCY="${TCC_PROMOTION_MIN_ROOT_CONSISTENCY:-0.55}"
TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE="${TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE:-0.40}"
TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT="${TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT:-0.45}"
ENABLE_TERMINAL_MEMORY_CONSOLIDATION="${ENABLE_TERMINAL_MEMORY_CONSOLIDATION:-0}"
ENABLE_ITERATIVE_MEMORY_CONSTRUCTION="${ENABLE_ITERATIVE_MEMORY_CONSTRUCTION:-0}"
IMC_MAX_ROUNDS="${IMC_MAX_ROUNDS:-2}"
IMC_MAX_REPAIR_GOALS="${IMC_MAX_REPAIR_GOALS:-2}"
TMC_CANDIDATE_LIMIT="${TMC_CANDIDATE_LIMIT:-5}"
ENABLE_ANYTIME_FALLBACK="${ENABLE_ANYTIME_FALLBACK:-0}"
ANYTIME_FALLBACK_THRESHOLD="${ANYTIME_FALLBACK_THRESHOLD:-0.82}"
FINAL_MIN_ROOT_ALIGNMENT="${FINAL_MIN_ROOT_ALIGNMENT:-0.55}"
FINAL_MIN_DEPENDENCY_SATISFACTION="${FINAL_MIN_DEPENDENCY_SATISFACTION:-0.40}"
FINAL_MIN_LAST_HOP_SUPPORT="${FINAL_MIN_LAST_HOP_SUPPORT:-0.50}"
FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP="${FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP:-0.55}"
FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP="${FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP:-0.60}"
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

if [[ ! -f "${DATASET}" ]]; then
  echo "[musique-tdca-random] dataset not found: ${DATASET}" >&2
  echo "[musique-tdca-random] expected official data at: ${DEFAULT_MUSIQUE_DATA}" >&2
  exit 1
fi

mkdir -p "${SAMPLE_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_ROOT}/musique_tdca_random${RANDOM_COUNT}_${STAMP}.log"

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
    echo "[musique-tdca-random] ${label}: summary missing: ${summary_path}"
    return
  fi
  python - "${label}" "${summary_path}" <<'PY'
import csv
import sys

label, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"[musique-tdca-random] {label}: rows=0")
    raise SystemExit(0)

def avg(col):
    vals = [float(r.get(col) or 0.0) for r in rows if col in r]
    return sum(vals) / len(vals) if vals else 0.0

em = sum(int(float(r.get("exact_match") or 0)) for r in rows)
nonempty = sum(1 for r in rows if (r.get("pred") or "").strip())
print(
    f"[musique-tdca-random] {label}: rows={len(rows)} nonempty={nonempty} "
    f"EM={em}/{len(rows)} acc={em/len(rows):.3f} "
    f"soft_em={avg('soft_em'):.3f} f1={avg('answer_f1'):.3f} "
    f"title_hit={avg('title_hit'):.3f} steps={avg('steps'):.2f} "
    f"calls={avg('llm_calls'):.2f} gen_tokens={avg('generated_tokens'):.2f}"
)
PY
}

make_random_dataset() {
  local seed="$1"
  local out="${SAMPLE_ROOT}/musique_random${RANDOM_COUNT}_seed${seed}.jsonl"
  local meta="${out}.meta.json"
  if [[ -f "${out}" && -f "${meta}" ]]; then
    echo "${out}"
    return
  fi
  python - "${DATASET}" "${out}" "${meta}" "${RANDOM_COUNT}" "${seed}" <<'PY'
import json
import random
import sys
from pathlib import Path

dataset, output, meta_path, count, seed = sys.argv[1:6]
count = int(count)
seed = int(seed)
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

if count > len(items):
    raise ValueError(f"Requested {count} samples, but dataset has only {len(items)} items")

rng = random.Random(seed)
selected = sorted(rng.sample(range(len(items)), count))
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
    "selected_indices": selected,
}
Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
}

run_tdca() {
  local dataset="$1"
  local seed="$2"
  local tdca_extra_args=()
  if [[ "${ENABLE_FINAL_CHAIN_BUFFER}" == "1" ]]; then
    tdca_extra_args+=(--enable_final_chain_buffer)
  fi
  if [[ "${ENABLE_SCORE_BASED_FINAL_ADMISSION}" == "1" ]]; then
    tdca_extra_args+=(--enable_score_based_final_admission)
  fi
  tdca_extra_args+=(--final_chain_score_threshold "${FINAL_CHAIN_SCORE_THRESHOLD}")
  if [[ "${ENABLE_TERMINAL_CHAIN_CLOSURE}" == "1" ]]; then
    tdca_extra_args+=(--enable_terminal_chain_closure)
  fi
  if [[ "${ENABLE_TCC_FINAL_AUDIT}" == "1" ]]; then
    tdca_extra_args+=(--enable_tcc_final_audit)
  fi
  tdca_extra_args+=(--tcc_final_audit_mode "${TCC_FINAL_AUDIT_MODE}")
  tdca_extra_args+=(--tcc_rerank_policy "${TCC_RERANK_POLICY}")
  tdca_extra_args+=(--tcc_score_threshold "${TCC_SCORE_THRESHOLD}")
  tdca_extra_args+=(--tcc_min_path_completeness "${TCC_MIN_PATH_COMPLETENESS}")
  tdca_extra_args+=(--tcc_min_dependency_closure "${TCC_MIN_DEPENDENCY_CLOSURE}")
  tdca_extra_args+=(--tcc_min_last_hop_entailment "${TCC_MIN_LAST_HOP_ENTAILMENT}")
  tdca_extra_args+=(--tcc_min_terminality "${TCC_MIN_TERMINALITY}")
  tdca_extra_args+=(--tcc_min_root_consistency "${TCC_MIN_ROOT_CONSISTENCY}")
  tdca_extra_args+=(--tcc_min_dependency_closure_shorthop "${TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP}")
  tdca_extra_args+=(--tcc_min_root_consistency_shorthop "${TCC_MIN_ROOT_CONSISTENCY_SHORTHOP}")
  tdca_extra_args+=(--tcc_min_last_hop_entailment_shorthop "${TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP}")
  tdca_extra_args+=(--tcc_min_terminality_shorthop "${TCC_MIN_TERMINALITY_SHORTHOP}")
  tdca_extra_args+=(--tcc_min_dependency_closure_longhop "${TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP}")
  tdca_extra_args+=(--tcc_min_root_consistency_longhop "${TCC_MIN_ROOT_CONSISTENCY_LONGHOP}")
  tdca_extra_args+=(--tcc_min_last_hop_entailment_longhop "${TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP}")
  tdca_extra_args+=(--tcc_min_terminality_longhop "${TCC_MIN_TERMINALITY_LONGHOP}")
  if [[ "${ENABLE_TCC_VERIFIED_PROMOTION}" == "1" ]]; then
    tdca_extra_args+=(--enable_tcc_verified_promotion)
  fi
  tdca_extra_args+=(--tcc_promotion_policy "${TCC_PROMOTION_POLICY}")
  tdca_extra_args+=(--tcc_promotion_min_hop "${TCC_PROMOTION_MIN_HOP}")
  if [[ "${ALLOW_STRICT_2HOP_PROMOTION}" == "1" ]]; then
    tdca_extra_args+=(--allow_strict_2hop_promotion)
  fi
  tdca_extra_args+=(--tcc_promotion_score_threshold "${TCC_PROMOTION_SCORE_THRESHOLD}")
  tdca_extra_args+=(--tcc_promotion_min_terminality "${TCC_PROMOTION_MIN_TERMINALITY}")
  tdca_extra_args+=(--tcc_promotion_min_root_consistency "${TCC_PROMOTION_MIN_ROOT_CONSISTENCY}")
  tdca_extra_args+=(--tcc_promotion_min_dependency_closure "${TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE}")
  tdca_extra_args+=(--tcc_promotion_min_last_hop_entailment "${TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT}")
  if [[ "${ENABLE_TERMINAL_MEMORY_CONSOLIDATION}" == "1" ]]; then
    tdca_extra_args+=(--enable_terminal_memory_consolidation)
  fi
  if [[ "${ENABLE_ITERATIVE_MEMORY_CONSTRUCTION}" == "1" ]]; then
    tdca_extra_args+=(--enable_iterative_memory_construction)
  fi
  tdca_extra_args+=(--imc_max_rounds "${IMC_MAX_ROUNDS}")
  tdca_extra_args+=(--imc_max_repair_goals "${IMC_MAX_REPAIR_GOALS}")
  tdca_extra_args+=(--tmc_candidate_limit "${TMC_CANDIDATE_LIMIT}")
  if [[ "${ENABLE_ANYTIME_FALLBACK}" == "1" ]]; then
    tdca_extra_args+=(--enable_anytime_fallback)
  fi
  tdca_extra_args+=(--anytime_fallback_threshold "${ANYTIME_FALLBACK_THRESHOLD}")
  tdca_extra_args+=(--final_min_root_alignment "${FINAL_MIN_ROOT_ALIGNMENT}")
  tdca_extra_args+=(--final_min_dependency_satisfaction "${FINAL_MIN_DEPENDENCY_SATISFACTION}")
  tdca_extra_args+=(--final_min_last_hop_support "${FINAL_MIN_LAST_HOP_SUPPORT}")
  tdca_extra_args+=(--final_min_dependency_satisfaction_longhop "${FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP}")
  tdca_extra_args+=(--final_min_last_hop_support_longhop "${FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP}")

  echo "[musique-tdca-random] running tdca seed=${seed}"
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
    --answer_synthesis_reserve_tokens "${TDCA_RESERVE_TOKENS}" \
    --intermediate_generation_budget_fraction "${TDCA_INTERMEDIATE_BUDGET_FRACTION}" \
    --open_goal_intermediate_budget_fraction "${TDCA_OPEN_GOAL_BUDGET_FRACTION}" \
    --max_new_tokens_answer "${TOKEN_BUDGET}" \
    --top_k "${TOP_K}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    "${tdca_extra_args[@]}" \
    --output_root "${OUTPUT_ROOT}" \
    --run_tag "random${RANDOM_COUNT}_seed${seed}_tok${TOKEN_BUDGET}"

  local out
  out="$(latest_batch_dir "${dataset}")"
  echo "[musique-tdca-random] tdca output_dir=${out}"
  print_summary "tdca_seed_${seed}" "${out}/summary.csv"
}

{
  echo "[musique-tdca-random] started_at=${STAMP}"
  echo "[musique-tdca-random] dataset=${DATASET}"
  echo "[musique-tdca-random] random_count=${RANDOM_COUNT} seeds=${SEEDS}"
  echo "[musique-tdca-random] backend=${LLM_BACKEND} model=${LLM_MODEL} token_budget=${TOKEN_BUDGET}"
  echo "[musique-tdca-random] top_k=${TOP_K} tdca_max_steps=${TDCA_MAX_STEPS}"
  echo "[musique-tdca-random] tdca_reserve=${TDCA_RESERVE_TOKENS} intermediate_fraction=${TDCA_INTERMEDIATE_BUDGET_FRACTION} open_goal_fraction=${TDCA_OPEN_GOAL_BUDGET_FRACTION}"
  echo "[musique-tdca-random] final_chain_buffer=${ENABLE_FINAL_CHAIN_BUFFER} score_admission=${ENABLE_SCORE_BASED_FINAL_ADMISSION} final_chain_threshold=${FINAL_CHAIN_SCORE_THRESHOLD} anytime_fallback=${ENABLE_ANYTIME_FALLBACK} anytime_threshold=${ANYTIME_FALLBACK_THRESHOLD}"
  echo "[musique-tdca-random] tcc=${ENABLE_TERMINAL_CHAIN_CLOSURE} tcc_audit=${ENABLE_TCC_FINAL_AUDIT} tcc_audit_mode=${TCC_FINAL_AUDIT_MODE} tcc_rerank_policy=${TCC_RERANK_POLICY} tcc_threshold=${TCC_SCORE_THRESHOLD} tcc_floors path=${TCC_MIN_PATH_COMPLETENESS} dep=${TCC_MIN_DEPENDENCY_CLOSURE} last=${TCC_MIN_LAST_HOP_ENTAILMENT} term=${TCC_MIN_TERMINALITY} root=${TCC_MIN_ROOT_CONSISTENCY}"
  echo "[musique-tdca-random] tcc adaptive floors short dep=${TCC_MIN_DEPENDENCY_CLOSURE_SHORTHOP} root=${TCC_MIN_ROOT_CONSISTENCY_SHORTHOP} last=${TCC_MIN_LAST_HOP_ENTAILMENT_SHORTHOP} term=${TCC_MIN_TERMINALITY_SHORTHOP} long dep=${TCC_MIN_DEPENDENCY_CLOSURE_LONGHOP} root=${TCC_MIN_ROOT_CONSISTENCY_LONGHOP} last=${TCC_MIN_LAST_HOP_ENTAILMENT_LONGHOP} term=${TCC_MIN_TERMINALITY_LONGHOP}"
  echo "[musique-tdca-random] tcc_promotion=${ENABLE_TCC_VERIFIED_PROMOTION} promotion_policy=${TCC_PROMOTION_POLICY} promotion_min_hop=${TCC_PROMOTION_MIN_HOP} allow_strict_2hop=${ALLOW_STRICT_2HOP_PROMOTION} promotion_threshold=${TCC_PROMOTION_SCORE_THRESHOLD} promotion_floors dep=${TCC_PROMOTION_MIN_DEPENDENCY_CLOSURE} last=${TCC_PROMOTION_MIN_LAST_HOP_ENTAILMENT} term=${TCC_PROMOTION_MIN_TERMINALITY} root=${TCC_PROMOTION_MIN_ROOT_CONSISTENCY}"
  echo "[musique-tdca-random] terminal_memory_consolidation=${ENABLE_TERMINAL_MEMORY_CONSOLIDATION} iterative_memory_construction=${ENABLE_ITERATIVE_MEMORY_CONSTRUCTION} imc_max_rounds=${IMC_MAX_ROUNDS} imc_max_repair_goals=${IMC_MAX_REPAIR_GOALS} tmc_candidate_limit=${TMC_CANDIDATE_LIMIT}"
  echo "[musique-tdca-random] final floors root=${FINAL_MIN_ROOT_ALIGNMENT} dep=${FINAL_MIN_DEPENDENCY_SATISFACTION} last_hop=${FINAL_MIN_LAST_HOP_SUPPORT} dep_long=${FINAL_MIN_DEPENDENCY_SATISFACTION_LONGHOP} last_hop_long=${FINAL_MIN_LAST_HOP_SUPPORT_LONGHOP}"
  echo "[musique-tdca-random] log=${LOG_PATH}"

  if [[ "${LLM_BACKEND}" != "mock" ]]; then
    echo "[musique-tdca-random] probing API"
    python scripts/probe_llm_chat.py
  fi

  IFS=',' read -ra seed_list <<< "${SEEDS}"
  for raw_seed in "${seed_list[@]}"; do
    seed="$(echo "${raw_seed}" | xargs)"
    if [[ -z "${seed}" ]]; then
      continue
    fi

    random_dataset="$(make_random_dataset "${seed}")"
    echo "[musique-tdca-random] random_dataset=${random_dataset}"
    echo "[musique-tdca-random] random_meta=${random_dataset}.meta.json"
    run_tdca "${random_dataset}" "${seed}"
  done

  echo "[musique-tdca-random] complete"
} 2>&1 | tee "${LOG_PATH}"
