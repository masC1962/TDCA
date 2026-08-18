#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-4}"
LIMIT="${2:-10}"
DATASET="${3:-data/hotpot_dev_distractor_v1.jsonl}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/TDCA}"
RUNNER_SRC="${RUNNER_SRC:-/mnt/data/tdca_batch_hotpotqa.py}"
RUNNER_DST="${RUNNER_DST:-$PROJECT_ROOT/tdca_batch_hotpotqa.py}"

# Defaults can be overridden via environment variables.
MODES_STR="${MODES:-tdca greedy uniform no_diffusion}"
BUDGETS_STR="${BUDGETS:-300 600 1200 2400}"
LLM_BACKEND="${LLM_BACKEND:-openrouter}"
SCORING_MODE="${SCORING_MODE:-hybrid}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENROUTER_BASE_URL:-https://yh.m7ai.com/v1}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${OPENROUTER_MODEL:-gpt-5.4}}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
MAX_STEPS="${MAX_STEPS:--1}"
EXTRA_TAG="${EXTRA_TAG:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG_PART=""
if [[ -n "$EXTRA_TAG" ]]; then
  TAG_PART="_${EXTRA_TAG}"
fi
EXP_ROOT="$PROJECT_ROOT/exp_outputs/tdca_grid_${STAMP}${TAG_PART}"
LOG_ROOT="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

if [[ ! -f "$RUNNER_DST" ]]; then
  if [[ -f "$RUNNER_SRC" ]]; then
    cp "$RUNNER_SRC" "$RUNNER_DST"
  else
    echo "[ERROR] Batch runner not found: $RUNNER_DST or $RUNNER_SRC" >&2
    exit 1
  fi
fi

if [[ ! -f "$PROJECT_ROOT/$DATASET" && ! -f "$DATASET" ]]; then
  echo "[ERROR] dataset not found: $DATASET" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRANSFORMERS_VERBOSITY="error"
export TOKENIZERS_PARALLELISM="false"

printf '\n[GRID] project_root = %s\n' "$PROJECT_ROOT"
printf '[GRID] dataset      = %s\n' "$DATASET"
printf '[GRID] limit        = %s\n' "$LIMIT"
printf '[GRID] modes        = %s\n' "$MODES_STR"
printf '[GRID] budgets      = %s\n' "$BUDGETS_STR"
printf '[GRID] backend      = %s\n' "$LLM_BACKEND"
printf '[GRID] gpu          = %s\n' "$GPU_ID"
printf '[GRID] outputs      = %s\n\n' "$EXP_ROOT"

for mode in $MODES_STR; do
  for budget in $BUDGETS_STR; do
    RUN_NAME="${mode}_tok${budget}"
    OUT_DIR="$EXP_ROOT/$RUN_NAME"
    mkdir -p "$OUT_DIR"
    LOG_FILE="$LOG_ROOT/$RUN_NAME.log"

    echo "[GRID] >>> running mode=$mode budget=$budget"
    python "$RUNNER_DST" \
      --dataset "$DATASET" \
      --limit "$LIMIT" \
      --project_root "$PROJECT_ROOT" \
      --output_root "$OUT_DIR" \
      --llm_backend "$LLM_BACKEND" \
      --scheduler_mode "$mode" \
      --scoring_mode "$SCORING_MODE" \
      ${OPENAI_BASE_URL:+--openai_base_url "$OPENAI_BASE_URL"} \
      ${SERVED_MODEL_NAME:+--served_model_name "$SERVED_MODEL_NAME"} \
      ${OPENROUTER_API_KEY:+--openai_api_key "$OPENROUTER_API_KEY"} \
      --reasoning_effort "$REASONING_EFFORT" \
      --max_steps "$MAX_STEPS" \
      --max_total_generated_tokens "$budget" \
      > "$LOG_FILE" 2>&1

    echo "[GRID] <<< done mode=$mode budget=$budget"
  done
  echo
 done

python - <<'PY' "$EXP_ROOT"
import csv, json, sys
from pathlib import Path

exp_root = Path(sys.argv[1])
rows = []
for run_dir in sorted([p for p in exp_root.iterdir() if p.is_dir() and p.name != 'logs']):
    # each run dir contains exactly one timestamped subdir created by tdca_batch_hotpotqa.py
    subdirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    if not subdirs:
        continue
    batch_dir = subdirs[-1]
    agg_path = batch_dir / 'aggregate.json'
    if not agg_path.exists():
        continue
    agg = json.loads(agg_path.read_text(encoding='utf-8'))
    name = run_dir.name
    mode = name.split('_tok')[0] if '_tok' in name else name
    budget = int(name.split('_tok', 1)[1]) if '_tok' in name else None
    rows.append({
        'run_name': name,
        'mode': mode,
        'token_budget': budget,
        'count': agg.get('count'),
        'exact_match': agg.get('exact_match'),
        'avg_steps': agg.get('avg_steps'),
        'avg_llm_calls': agg.get('avg_llm_calls'),
        'avg_generated_tokens': agg.get('avg_generated_tokens'),
        'batch_dir': str(batch_dir),
    })

rows.sort(key=lambda r: (r['token_budget'] if r['token_budget'] is not None else 10**9, r['mode']))
leaderboard_json = exp_root / 'leaderboard.json'
leaderboard_csv = exp_root / 'leaderboard.csv'
leaderboard_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
with leaderboard_csv.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'run_name', 'mode', 'token_budget', 'count', 'exact_match',
        'avg_steps', 'avg_llm_calls', 'avg_generated_tokens', 'batch_dir'
    ])
    writer.writeheader()
    writer.writerows(rows)
print(f'[GRID] leaderboard.csv  = {leaderboard_csv}')
print(f'[GRID] leaderboard.json = {leaderboard_json}')
PY

echo
printf '[GRID] all runs finished.\n'
printf '[GRID] leaderboard: %s\n' "$EXP_ROOT/leaderboard.csv"
printf '[GRID] logs:        %s\n' "$LOG_ROOT"
