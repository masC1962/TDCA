# Cross-dataset validation protocol

This protocol was frozen after the MuSiQue tuning-50 configuration was frozen. Cross-dataset
labels may be used for reporting and error analysis, but they must not change the MuSiQue
validation-200 algorithm or thresholds.

## Data

- HotpotQA: full local distractor dev, 7,405 examples.
- 2WikiMultiHopQA: Apache-2.0 official mirror `xanhho/2WikiMultihopQA`, commit
  `612bc5039a457880d9e7d84c3b0a4cf154b70e4f`. Only `dev.parquet` was downloaded
  (30,056,098 bytes; SHA-256
  `c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d`).
  The converted JSONL hash and source columns are recorded in its provenance sidecar.
- Each full dataset uses seed 520 and pairwise-disjoint 20/50/200/1,000 IDs.

## Stages

1. Run retrieval-only probes at k=2 and k=5. These are diagnostics and do not mutate
   frozen inference configs.
2. Run Qwen-plus Structured-TDCA smoke-20 with the unchanged MuSiQue thresholds/budgets.
3. Run controlled IRCoT on the identical IDs and budgets.
4. If both have zero infrastructure failures, expand to the disjoint tuning-50 split.
5. Report answer EM/F1, selectivity, evidence recall, calls/tokens/latency, by-hop, and
   by-type. Do not select a favorable dataset seed.

The main transfer track retains Dense retrieval because it was frozen before cross-dataset
gold inspection. Hybrid retrieval is a separately labeled ablation: k=2 question-retrieval
probes gave higher support-recall point estimates on both HotpotQA and 2Wiki, so silently
switching the main track would be post-selection.

## Current status

Data integrity, provenance, fixed splits, scorer semantics, retrieval probes and all four
Qwen-plus smoke-20 runs are complete. The strict smoke gate opened with zero
infrastructure failures; see `docs/cross_dataset_smoke_results.md`. Disjoint tuning-50
runs are also complete with the unchanged frozen configs; neither dataset passes the
quality expansion criterion, so validation-200 is not run. See
`docs/cross_dataset_tuning50_results.md`.

To reproduce the frozen smoke sequence:

```bash
bash scripts/run_cross_dataset_smoke.sh
```

The launcher first reruns tests, official scorer parity and dataset integrity, then
executes Structured-Dense and IRCoT sequentially for HotpotQA followed by 2Wiki. It
does not run the Hybrid diagnostic or expand to tuning-50 automatically.

Only after all four smoke artifacts pass `scripts/verify_artifact.py` with zero
infrastructure failures, create the fail-closed gate and run the frozen disjoint
tuning sequence with:

```bash
python scripts/build_cross_dataset_smoke_gate.py \
  research_outputs/<hotpot-structured-smoke> \
  research_outputs/<hotpot-ircot-smoke> \
  research_outputs/<2wiki-structured-smoke> \
  research_outputs/<2wiki-ircot-smoke>
bash scripts/run_cross_dataset_tuning.sh
```

This separate gate prevents a partial or failed smoke from silently authorizing more
API expenditure.
