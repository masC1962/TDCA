# MuSiQue Smoke-20 Results

Run date: 2026-08-13. Model: `qwen-plus`. Split: the frozen `smoke` IDs in
`configs/splits/musique_dev_seed520.json`. These 20 examples are diagnostic and
are not sufficient for a statistical or SOTA claim.

## Distractor setting

| Method | EM | F1 | Answered | Selective accuracy | Support recall | All gold recalled | Logical LLM calls | Provider calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 0.30 | 0.321 | 0.65 | 0.462 | 0.567 | 0.25 | 20 | 20 cold / 0 replay |
| Dense RAG | 0.25 | 0.282 | 0.85 | 0.294 | 0.783 | 0.50 | 20 | 20 |
| Controlled IRCoT + BM25 | 0.35 | 0.443 | 0.75 | 0.467 | 0.871 | 0.70 | 66 | 66 cold / 0 replay |
| HippoRAG 2 controlled + MiniLM | 0.25 | 0.378 | 1.00 | 0.250 | 0.833 @10 | — | see note | see note |
| Structured-TDCA + BM25 | 0.45 | 0.549 | 0.60 | 0.750 | 0.854 | 0.75 | 110 | 110 cold / 0 replay |
| Structured-TDCA + Dense | **0.50** | **0.592** | 0.75 | 0.667 | **1.000** | **1.00** | 120 | 100 (20 cache hits) |

Dense question-level retrieval had higher evidence recall than BM25, but direct
Dense RAG was worse. Dense became useful when paired with dependency-conditioned
multi-step retrieval and typed working memory. Structured-Dense is therefore the
quality candidate for the non-overlapping 50-example tuning split; Structured-BM25
remains the cheaper and better-selective candidate.

## HippoRAG 2 controlled official-code track

The official repository was checked out at
`c617143f01477243992a63b2e2151cc003dd3b21`. The run used unmodified upstream
HippoRAG execution code with Qwen-plus and the upstream-supported
`Transformers/sentence-transformers/all-MiniLM-L6-v2` embedding backend.

The fair distractor track built one independent graph over each question's original
20 passages. It obtained EM 0.25, F1 0.3784, Recall@5 0.6792 and Recall@10 0.8333.
Independent scoring exactly reproduced the upstream metrics and found no final parser
failure. This is the HippoRAG row in the distractor table. Upstream HippoRAG reports
mean support recall at each cutoff; the table therefore leaves the stricter
all-gold-per-question column blank instead of conflating the two metrics.

A second diagnostic used a 395-passage `mini_global_union` corpus made from the same
20 questions. It is not distractor-equivalent and is reported separately:

- QA EM: 0.30; F1: 0.4019.
- Retrieval recall: @5 0.6292, @10 0.7958, @20 0.9375.
- Graph: 4,669 nodes, 16,959 edges.
- Cold graph build + retrieval + QA: about 389 seconds.
- Cached provider workload: 830 calls, 456,198 prompt tokens and 170,321 completion tokens.
- Independent answer scoring reproduced upstream EM/F1 exactly; no final empty answer remained.

The shared cold OpenIE cache contains 845 distinct calls and 658,145 tokens after both
tracks. The distractor run reused the 395 documents' content-addressed OpenIE cache,
then built independent graphs and ran per-question retrieval/QA in about 201 seconds.
This avoids paying for the same document extraction twice while preserving independent
per-question graphs. Neither controlled track is the paper-default NV-Embed-v2 setup.

## Metric correction

An audit found that the original loader expanded a supporting MuSiQue title to every
same-title paragraph. For example, a 2-hop item could be scored as having 14 gold
paragraphs. The loader now preserves exact paragraph IDs whenever `is_supporting`
labels exist. Answer EM/F1 never used this mapping and are unchanged; all evidence
numbers in this document were recomputed after the fix. Older evidence summaries
created before this correction are superseded.

## Gate decision

The engineering smoke gate passes: no infrastructure failures, no budget violations,
fixed sample IDs, auditable artifacts, and a quality improvement over controlled
IRCoT. The 20-example paired difference is not a statistical claim. Proceed to the
disjoint 50-example tuning split; do not proceed to 200/1,000 until the 50-example
gate is reviewed and the configuration is frozen.
