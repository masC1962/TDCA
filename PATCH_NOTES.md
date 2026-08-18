# TDCA baseline patch

This patch adds a minimal but runnable unified baseline layer on top of the existing TDCA repo.

## Added
- `dataset_adapters/`
  - `base.py`
  - `loader.py`
- `baselines/`
  - `common.py`
  - `closed_book.py`
  - `sparse_rag.py`
  - `dense_rag.py`
  - `ircot.py`
- `scripts/`
  - `build_dense_index.py`
  - `run_baseline.py`
- `example_baseline_config.json`
- `run_baseline_examples.sh`

## Updated
- `config.py`
  - add baseline / dataset / retriever fields
- `retriever.py`
  - keep `SparseTextRetriever`
  - add `BaseRetriever`, `DenseTextRetriever`, `HybridRetriever`, `build_retriever`
- `knowledge_memory.py`
  - allow injected retriever backends
- `tdca_batch_hotpotqa.py`
  - support `answer_aliases` and `paragraph_text`
- `evaluate_batch_metrics_qwen.py`
  - list-gold aware EM/F1 and simple title-hit metric

## What is runnable now
- `closed_book`
- `sparse_rag`
- `dense_rag`
- `ircot`

## Dense retrieval behavior
- If `sentence-transformers` is installed and `--encoder_path` is provided, the dense retriever uses it.
- Otherwise it falls back to a TF-IDF vector backend, so the script still runs.

## Example commands
```bash
python scripts/build_dense_index.py \
  --dataset_path data/hotpotqa_subset_50.jsonl \
  --dataset_name hotpotqa \
  --output indexes/hotpot_dense_index.npz

python scripts/run_baseline.py \
  --baseline sparse_rag \
  --dataset_path data/hotpotqa_subset_50.jsonl \
  --dataset_name hotpotqa \
  --llm_backend local \
  --model_path /workspace/models/Qwen3-4B \
  --output_dir outputs/hotpot_sparse_rag

python scripts/run_baseline.py \
  --baseline dense_rag \
  --dataset_path data/hotpotqa_subset_50.jsonl \
  --dataset_name hotpotqa \
  --llm_backend local \
  --model_path /workspace/models/Qwen3-4B \
  --index_path indexes/hotpot_dense_index.npz \
  --output_dir outputs/hotpot_dense_rag

python scripts/run_baseline.py \
  --baseline ircot \
  --dataset_path data/hotpotqa_subset_50.jsonl \
  --dataset_name hotpotqa \
  --llm_backend local \
  --model_path /workspace/models/Qwen3-4B \
  --retriever_type dense \
  --index_path indexes/hotpot_dense_index.npz \
  --ircot_max_steps 3 \
  --output_dir outputs/hotpot_ircot
```


## 2026-04-24 hotfix
- Fixed empty predictions when using reasoning-style API models such as `gpt-5.x` in baseline runner.
- Cause: `run_baseline.py` used a very small default `--max_new_tokens_answer=64`, while reasoning models consume hidden reasoning budget from `max_completion_tokens`, often leaving no visible answer.
- Fix: raised default to 512, auto-bump for reasoning models, and increased IRCoT final-answer budget.
