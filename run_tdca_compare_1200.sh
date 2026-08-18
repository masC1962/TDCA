#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-4}"
LIMIT="${2:-30}"
DATASET="${3:-data/hotpot_dev_distractor_v1.jsonl}"

PROJECT_ROOT="/workspace/TDCA"
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://yh.m7ai.com/v1}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-gpt-5.4}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
RUNNER_SRC="/mnt/data/tdca_batch_hotpotqa.py"
RUNNER_DST="$PROJECT_ROOT/tdca_batch_hotpotqa.py"

MODES=("tdca" "greedy" "no_diffusion")
TOKEN_BUDGET="1200"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/exp_outputs"

if [[ -f "$RUNNER_SRC" ]]; then
  cp "$RUNNER_SRC" "$RUNNER_DST"
elif [[ ! -f "$RUNNER_DST" ]]; then
  echo "Batch runner not found. Expected one of:" >&2
  echo "  $RUNNER_SRC" >&2
  echo "  $RUNNER_DST" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$PROJECT_ROOT/exp_outputs/tdca_grid_${STAMP}_algo1200"
mkdir -p "$OUT_ROOT/logs"

LEADERBOARD_CSV="$OUT_ROOT/leaderboard.csv"
LEADERBOARD_JSON="$OUT_ROOT/leaderboard.json"

echo "mode,token_budget,count,exact_match,avg_steps,avg_llm_calls,avg_generated_tokens,aggregate_json" > "$LEADERBOARD_CSV"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

cd "$PROJECT_ROOT"

for MODE in "${MODES[@]}"; do
  EXP_NAME="${MODE}_tok${TOKEN_BUDGET}"
  LOG_PATH="$OUT_ROOT/logs/${EXP_NAME}.log"

  echo "=== Running ${MODE} @ ${TOKEN_BUDGET} tokens on ${LIMIT} samples ==="
  python tdca_batch_hotpotqa.py \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --project_root "$PROJECT_ROOT" \
    --llm_backend "${LLM_BACKEND:-openrouter}" \
    --openai_base_url "$OPENROUTER_BASE_URL" \
    --served_model_name "$OPENROUTER_MODEL" \
    ${OPENROUTER_API_KEY:+--openai_api_key "$OPENROUTER_API_KEY"} \
    --reasoning_effort "$REASONING_EFFORT" \
    --scheduler_mode "$MODE" \
    --scoring_mode hybrid \
    --max_total_generated_tokens "$TOKEN_BUDGET" \
    > "$LOG_PATH" 2>&1

  LATEST_BATCH="$(ls -td "$PROJECT_ROOT"/batch_outputs/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$LATEST_BATCH" ]]; then
    echo "No batch output found for ${MODE}" >&2
    continue
  fi

  DEST="$OUT_ROOT/$EXP_NAME"
  mkdir -p "$DEST"
  cp -r "$LATEST_BATCH"/. "$DEST"/

  AGG="$DEST/aggregate.json"
  if [[ -f "$AGG" ]]; then
    python - "$MODE" "$TOKEN_BUDGET" "$AGG" "$LEADERBOARD_CSV" <<'PY'
import json, sys, csv
mode, token_budget, agg_path, csv_path = sys.argv[1:5]
with open(agg_path, "r", encoding="utf-8") as f:
    data = json.load(f)
row = {
    "mode": mode,
    "token_budget": token_budget,
    "count": data.get("count"),
    "exact_match": data.get("exact_match"),
    "avg_steps": data.get("avg_steps"),
    "avg_llm_calls": data.get("avg_llm_calls"),
    "avg_generated_tokens": data.get("avg_generated_tokens"),
    "aggregate_json": agg_path,
}
with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        row["mode"], row["token_budget"], row["count"], row["exact_match"],
        row["avg_steps"], row["avg_llm_calls"], row["avg_generated_tokens"],
        row["aggregate_json"]
    ])
PY
  fi
done

python - "$LEADERBOARD_CSV" "$LEADERBOARD_JSON" <<'PY'
import csv, json, sys
csv_path, json_path = sys.argv[1:3]
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"wrote: {json_path}")
PY

echo "Done."
echo "Results root: $OUT_ROOT"
echo "Leaderboard:   $LEADERBOARD_CSV"
