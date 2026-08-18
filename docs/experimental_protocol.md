# Experimental Protocol

## Settings

Core datasets are MuSiQue, 2WikiMultiHopQA and HotpotQA. Each is evaluated in:

1. Controlled distractor mode using only per-question official candidate paragraphs.
2. Global-corpus mode using the HippoRAG 2 official data/corpus where compatible.

Bamboogle is the preferred small generalization set because IRCoT and HippoRAG-family
work report it; MultiHop-RAG may replace it if official corpus/scorer compatibility is
materially better. The choice must be frozen before final evaluation.

## Tracks

Paper reproduction uses upstream commits, dependencies, data, models and scorers.
Controlled comparison uses Qwen-plus, the same corpus, retrieval/evidence budget,
total LLM token budget, maximum calls, retry policy and official answer scorer.
Reported numbers are labeled paper-reported, locally reproduced, reimplementation or
failed to reproduce.

For native iterative methods, `top_k` and the per-call evidence-character cap are
matched, as are the global LLM-call/token ceilings. Total retrieval calls are an
outcome of each algorithm and are reported rather than falsely described as matched.
Official HippoRAG has internal retrieval/linking/QA cutoffs that cannot be made
identical without changing upstream semantics; its result is therefore a controlled
configuration comparison, accompanied by calls/tokens/latency, not a strict
matched-retrieval-budget claim.

Required controlled methods are closed-book, BM25, dense, hybrid, IRCoT, HippoRAG 2,
legacy TDCA, the new method, oracle evidence, oracle decomposition, and their combined
oracle. Training-dependent KiRAG is reproduction-track only.

## Splits

Official full data generates disjoint smoke/tuning/validation/final manifests of
20/50/200/1000 examples with seed 520. Compact 50-row files may use explicitly marked
nested diagnostic subsets but cannot support held-out validation claims.

The bundled MuSiQue full dev is the default controlled corpus. Its compact 50-row
derivative is support-only and is rejected by the distractor integrity gate. The
bundled Hotpot compact derivative is also not a formal benchmark artifact: 8/50 rows
name supporting titles absent from their supplied context. These files may test parser
behavior, but cannot back paper comparisons.

Cross-dataset validation uses the full local HotpotQA dev and the 2WikiMultiHopQA dev
file from the `xanhho/2WikiMultihopQA` official mirror pinned at commit
`612bc5039a457880d9e7d84c3b0a4cf154b70e4f`. Only `dev.parquet` (30,056,098 bytes,
SHA-256 prefix `c0d8b60b`) is downloaded and converted; the converted JSONL,
provenance sidecar and frozen sample IDs are retained. The mirror's nested values are
JSON strings, so preparation decodes those fields generically before adapter parsing.

## Metrics

Primary answer metrics use official normalized EM and token F1. Retrieval reports
support precision/recall/F1, all-gold-document recall and ordered path recall;
`title_hit` is auxiliary. Reasoning diagnostics report decomposition nodes/edges,
binding, verified/grounded claim precision and full-chain correctness when oracle
annotations exist. Per-example rows make paired bootstrap possible.
Efficiency reports calls, prompt/completion tokens, retrievals, wall time, index time
and size. Calibration reports ECE and bins; selectivity reports answered/abstention
rates and selective accuracy. Final comparisons use aligned paired bootstrap with a
fixed seed and 10,000 resamples.

Official answer scorer parity is checked programmatically against the pinned HotpotQA
and 2Wiki source. Supporting-fact joint metrics are not claimed by that answer-only
parity check. Provider-policy and other infrastructure failures remain in the primary
denominator and are reported separately from algorithmic abstentions.

## Gates and tuning

No final test labels may drive changes. After the 50-row tuning gate, configuration is
frozen for 200-row validation. Failure analysis uses retrieval miss, decomposition,
binding, hallucination, verifier false accept/reject, budget and final synthesis
categories. Entity-specific production fixes are forbidden.

The frozen MuSiQue validation-200 comparison failed the Stage-5 gate: Structured-Dense
did not significantly improve EM/F1 over controlled IRCoT and used more tokens. The
1,000-row final split is therefore not run. Cross-dataset smoke remains diagnostic of
transfer under the already frozen configuration and cannot retroactively unlock the
MuSiQue gate.
