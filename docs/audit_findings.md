# Legacy Audit Findings

The initial audit reproduced the supplied diagnostics:

- `tdca_scheduler.py`: 12,280 lines and 314 functions.
- `TDCAConfig`: 151 fields.
- The scheduler contains final-chain, chain-closure, memory-consolidation, promotion,
  reranking and answer-admission mechanisms in one class.
- Existing MuSiQue 100-row artifacts record TDCA EM/F1 0.08/0.0933 with 80 unanswered,
  versus sparse RAG 0.19/0.2823 and IRCoT 0.29/0.3799.
- Per-sample runtime memory is rebuilt from template memories, so it does not evaluate
  cross-task long-term memory.
- Compact `data/*_subset_50.jsonl` evidence is nested in decomposition metadata and was
  absent from the former top-level-only adapter path.
- The compact MuSiQue file is support-only (two gold passages per row), so it cannot
  be reported as a distractor benchmark. The complete bundled MuSiQue dev has 2,417
  rows and 20 paragraphs per row and is now the default.
- HotpotQA and 2Wiki compact contexts use parallel title/sentences-or-content arrays;
  the initial new adapter also missed that schema. This is fixed and regression-tested.
- Eight of the 50 compact Hotpot rows name supporting titles that are absent from the
  supplied candidate context, so that derivative is not suitable for a formal table.
- Legacy `--top_k` and TDCA retrieval fields are distinct.
- Legacy title hit is any-set intersection and does not measure complete evidence.
- Legacy soft EM accepts substring containment.
- Dense retrieval can use a TF-IDF fallback.
- The original tests cover five final-chain/closure/consolidation cases.

These facts motivate the new package but do not themselves demonstrate that the new
algorithm is better. That requires gated controlled experiments.

The frozen-ID tuning-50 rerun now supplies that first controlled checkpoint. Legacy
TDCA reaches independently rescored EM/F1 0.16/0.2067 and answers 48% of questions,
despite title-hit 1.00. Structured-Dense reaches 0.48/0.5186 and answers 60%, while
the strongest completed official-code controlled baseline, HippoRAG+MiniLM, reaches
0.44/0.4534. These are tuning results, not held-out significance or SOTA claims.

The frozen Structured-Dense validation-200 run completed with 200 predictions and
independently verified EM/F1 0.40/0.4530. A launcher file was edited while its shell
was still active; the already completed main artifact is intact, but the chained IRCoT
command exited 127 before starting. IRCoT was subsequently launched as a separate
immutable command. This orchestration error is not hidden and does not alter samples,
prompts, predictions or main-run metrics.
