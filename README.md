# TDCA Research: Structured Working-Memory Graphs for Multi-Hop QA

This repository now has two deliberately separated paths:

- `legacy/tdca_v0/` preserves the original TDCA entrypoint and audited source list.
- `src/tdca_research/` is a training-free research framework for dependency-aware
  multi-hop question answering. It is not presented as a general long-term-memory agent.

The new method represents a question as a dependency DAG, stores intermediate
conclusions as typed and evidence-provenanced claims, retrieves for the current
unresolved slot, and schedules work by expected information utility per cost.
Generation, verification and final synthesis are separate calls.

The research tree also includes an isolated next-generation method,
`dynamic_hypergraph_tdca`: a transactional Dynamic Reasoning Hypergraph with
CandidateSets, independent raw scoring, lazy branching, bounded revision, an
event-triggered graph editor, normalized operation scheduling, and graph-grounded
answers. Its design and sealed development protocol are documented in
[`docs/dynamic_hypergraph_tdca.md`](docs/dynamic_hypergraph_tdca.md). The original
`structured_tdca` code path and historical artifacts remain frozen for comparison.
The corresponding development results, paired intervals, budget curve, and known
negative results are in
[`docs/dynamic_hypergraph_results_20260820.md`](docs/dynamic_hypergraph_results_20260820.md).

`dynamic_hypergraph_tdca_v2` is an independent, training-free successor in
`src/tdca_research/dynamic_v2/`. It adds typed directional belief diffusion,
explicit relational JOIN materialization, event-triggered structural editing,
versioned revision cascades, graph-state-driven EVC allocation, measured allocation
cost ledgers, and three-way ANSWER/ABSTAIN/BUDGET_EXHAUSTED termination. The method,
invariants, equations, and fail-closed evaluation gate are documented in
[`docs/dynamic_hypergraph_tdca_v2.md`](docs/dynamic_hypergraph_tdca_v2.md).
For a self-contained, paper-level Chinese description of the complete research idea,
algorithm, implementation, current v2.2 evidence, limitations, and roadmap, see
[`docs/tdca.md`](docs/tdca.md).
The historical first mechanism-complete smoke was a negative result; the newer v2.2
development campaign and its still-closed gate are recorded in
[`docs/dynamic_v22_safe_stop_20260824.json`](docs/dynamic_v22_safe_stop_20260824.json).

## Install

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
# Dense retrieval (in the Linux GPU environment):
python -m pip install -e ".[dense]"
# Only when converting the pinned 2Wiki Parquet source:
python -m pip install -e ".[data]"
```

Run the preflight before downloading data or building an index:

```bash
python scripts/research_preflight.py
```

The report is written to `research_outputs/preflight.json`. The currently mounted
Windows view is not the GPU container, so CUDA/Docker checks must be repeated inside
the real Linux container.

## API

```bash
# In the Docker environment, create this once with chmod 600:
# /root/.config/tdca/qwen.env
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL=qwen-plus
# DASHSCOPE_API_KEY=...

bash scripts/run_qwen_experiment.sh \
  --config configs/structured_tdca_qwen.yaml
```

The key is read only from `LLM_API_KEY` or `DASHSCOPE_API_KEY` and is never placed in
manifests. API responses are content-address cached by provider, endpoint, model,
messages, generation settings, schema and prompt version. Failed responses are not
cached as successes. Timeout and maximum provider attempts are explicit config fields,
included in the cache key, and the SDK's hidden retry layer is disabled. The launcher reads the root-owned file inside `mc_env`; do not
place the key in repository files, shell history or Docker image layers.

## Run

```bash
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml --method ircot
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml --method bm25_rag
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml --method dense_rag
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml --retriever dense
python -m tdca_research.run --config configs/structured_tdca_qwen.yaml --oracle_evidence --oracle_decomposition
```

The default config uses the complete bundled MuSiQue dev file (2,417 rows, 20
paragraphs per question). The compact `musique_subset_50.jsonl` contains only gold
support passages, so the runtime integrity gate intentionally rejects it when it is
labeled `setting: distractor`. It remains useful for direct component tests. Create an
explicit diagnostic manifest with:

```bash
python -m tdca_research.split \
  --dataset_path data/musique_subset_50.jsonl --dataset musique \
  --nested_diagnostic --output configs/splits/musique_subset50_diagnostic.json
```

Nested manifests are labeled non-disjoint and must not be used for held-out claims.
The checked-in `configs/splits/musique_dev_seed520.json` is the versioned, pairwise
disjoint 20/50/200/1000 manifest used by the default config.

Full HotpotQA and 2Wiki cross-dataset manifests are generated deterministically with:

```bash
python scripts/build_split_manifest.py --dataset hotpotqa \
  --dataset-path data/hotpot_dev_distractor_v1.jsonl --seed 520 \
  --output configs/splits/hotpot_dev_seed520.json

# After downloading the pinned 30,056,098-byte dev.parquet described in
# docs/cross_dataset_protocol.md:
python scripts/prepare_2wiki_dev.py \
  --input data/external/2wikimultihopqa_612bc503/dev.parquet \
  --output data/external/2wikimultihopqa_612bc503/dev.jsonl
python scripts/build_split_manifest.py --dataset 2wikimultihopqa \
  --dataset-path data/external/2wikimultihopqa_612bc503/dev.jsonl --seed 520 \
  --output configs/splits/2wiki_dev_612bc503_seed520.json
```

The converter refuses an unexpected source SHA-256. Run
`python scripts/audit_research_datasets.py` before spending API calls.

Global-corpus runs require `global_corpus_path` in the config. HippoRAG 2 data is not
silently downloaded; follow `external_baselines/hipporag2_adapter.yaml`, verify sizes,
then build a corpus manifest:

```bash
python -m tdca_research.build_index --corpus external_data/hipporag2/corpus.jsonl \
  --output indexes/hipporag2 --encoder sentence-transformers/all-MiniLM-L6-v2
```

This command materializes embeddings (or an explicitly labeled TF-IDF diagnostic
index), records encoder/corpus fingerprints, dimensions, code version, size and build
time. Set `dense_index_path` to reuse it. Official external adapters refuse to execute
unless the repository exists at the pinned commit:

```bash
python -m tdca_research.run_external --name hipporag2 \
  --adapter external_baselines/hipporag2_adapter.yaml \
  --input data/questions.jsonl --output research_outputs/hipporag2.jsonl --verify_only
```

## Tests

```bash
bash scripts/test_all.sh
```

This runs the research suite, the frozen legacy regression suite, and a source
compilation check from the repository root. The legacy mock entrypoint remains
available at `python legacy/tdca_v0/run.py --mock_llm`.

The research tests include a structural offline smoke over 20 real MuSiQue rows. That
mock run tests execution and provenance only; it is never reported as accuracy.

## Experiment gates

For Dynamic Hypergraph v2, use the separate runner:

```bash
bash scripts/run_dynamic_v2_research.sh tests
bash scripts/run_dynamic_v2_research.sh smoke20
bash scripts/run_dynamic_v2_research.sh development50
bash scripts/run_dynamic_v2_research.sh matched_allocators50
bash scripts/run_dynamic_v2_research.sh budget_curve50
bash scripts/run_dynamic_v2_research.sh revision_development
```

`heldout200` and `cross_heldout200` fail closed unless `TDCA_V2_GATE_REPORT` points to
a passing `dynamic-hypergraph-v2-gate-evaluation-v2` report. Editing a status bit is
not sufficient. The seed is frozen at `20260820`; HotpotQA and 2Wiki use the same
frozen post-MuSiQue parameters and their own disjoint manifests. The later v2.2
development-50 adaptive and uniform runs completed, but the fixed-order control and
matched budget curve did not. The campaign then stopped at its provider-token cap.
The hard gate therefore remains closed and sealed heldout stages must not be run;
see `docs/dynamic_v22_safe_stop_20260824.json`. The subsequent v2.3 smoke
mechanism study also stopped before matched controls; its exact positive and
negative results are recorded in
[`docs/dynamic_v23_safe_stop_20260825.md`](docs/dynamic_v23_safe_stop_20260825.md).

1. Stage 1: adapters, leakage, metrics, budgets and integration tests.
2. Stage 2: Qwen-plus 20-row smoke; no crashes or ambiguous empty answers.
3. Stage 3: fixed 50-row tuning against RAG, IRCoT, HippoRAG 2 and legacy TDCA.
4. Stage 4: frozen 200-row held-out validation with paired bootstrap.
5. Stage 5: official 1,000/full protocol only after Stage 4.

The current frozen Stage-4 result did not pass. `configs/stage5_gate.json` records the
decision, and `scripts/run_research_gates.sh final1000` fails closed while its status is
`closed`; this prevents an accidental look at final labels.

The frozen Stage-4 MuSiQue run is:

```bash
bash scripts/run_validation200.sh
```

Cross-dataset smoke configs are
`configs/structured_tdca_qwen_hotpot_smoke20_frozen.yaml` and
`configs/structured_tdca_qwen_2wiki_smoke20_frozen.yaml`. Follow the pre-registered
sequence in `docs/cross_dataset_protocol.md`; do not choose Dense/Hybrid after viewing
answer labels. After all four smoke runs are independently audited with zero
infrastructure failures, use `scripts/build_cross_dataset_smoke_gate.py` on those four
run directories; only then does `bash scripts/run_cross_dataset_tuning.sh` run the
frozen, disjoint tuning-50 sequence.

The frozen cross-dataset tuning did not pass its expansion criterion.
`configs/cross_dataset_validation_gate.json` records that decision, and
`bash scripts/run_cross_dataset_validation.sh` fails closed before any API call. The
validation configs are versioned solely to preserve the preregistered protocol; do not
run them by bypassing the gate.

Every run writes manifests, resolved config, environment, predictions, retrieval and
reasoning traces, metrics, cost statistics, failures and checksums under
`research_outputs/`. New long runs append a `partial_progress.json` checkpoint and
per-question predictions/traces. A clean completion rewrites canonical JSONL files and
marks the checkpoint complete. `code_version` always records a deterministic SHA-256
over the active research source, scripts and external adapters, plus the Git commit and
dirty state when available. This also covers untracked research files.

Audit any completed native artifact without rerunning the model:

```bash
python scripts/verify_artifact.py research_outputs/<experiment_id> --expected-count 200
```

This rejects incomplete runs, checksum mutations, duplicate IDs, or a mismatch between
the ordered manifest, predictions, per-example metrics and aggregate count.
Artifacts created before checkpoint/type files were introduced can be audited without
mutation using `--allow-legacy-schema`; the report labels them `legacy_pre_checkpoint`.

Resume an interrupted new-CLI run with the identical configuration:

```bash
bash scripts/run_qwen_experiment.sh --config configs/structured_tdca_qwen.yaml \
  --resume_dir research_outputs/<interrupted_experiment_id>
```

Resume fails closed if the resolved configuration, dataset hash, ordered sample IDs or
durable checkpoint prefix differ. A torn append after the last progress checkpoint is
discarded before execution continues.

See [architecture](docs/architecture.md), [algorithm](docs/algorithm.md),
[experimental protocol](docs/experimental_protocol.md), and
[cross-dataset protocol](docs/cross_dataset_protocol.md), and
[reproduction status](docs/reproduction_status.md). Frozen validation-200 results and
the failed Stage-5 gate are recorded in
[validation results](docs/validation200_results.md). The corrected real Qwen-plus
smoke results and their setting boundaries are in
[smoke-20 results](docs/smoke20_results.md).
The current frozen 50-example checkpoint is in
[tuning-50 results](docs/tuning50_results.md).
Frozen HotpotQA/2Wiki transfer results are in
[cross-dataset smoke-20](docs/cross_dataset_smoke_results.md) and
[cross-dataset tuning-50](docs/cross_dataset_tuning50_results.md).
The consolidated empirical verdict and next research direction are in
[the final research report](docs/final_research_report.md).
