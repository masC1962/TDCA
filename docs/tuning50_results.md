# MuSiQue Frozen Tuning-50 Checkpoint

Run date: 2026-08-13. All native methods use `qwen-plus`, the exact frozen
`tuning` IDs from `configs/splits/musique_dev_seed520.json`, top-k 10, and the
same English answer normalizer. The main Structured-Dense configuration was
frozen before these labels were inspected.

| Method | EM | F1 | Answered | Selective accuracy | Support recall | All gold recalled | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 0.20 | 0.265 | 0.60 | 0.333 | 0.733 | 0.44 | 68,608 |
| Frozen legacy TDCA | 0.16 | 0.207 | 0.48 | 0.333 | title-hit 1.00 | — | output only: 67,770 |
| Dense RAG | 0.28 | 0.336 | 0.58 | 0.483 | 0.788 | 0.50 | 67,923 |
| Hybrid RAG | 0.30 | 0.340 | 0.60 | 0.500 | 0.800 | 0.58 | 67,197 |
| Controlled IRCoT + BM25 | 0.34 | 0.415 | 0.68 | 0.500 | 0.915 | 0.76 | 297,748 |
| HippoRAG 2 controlled + MiniLM | 0.44 | 0.453 | 1.00 | 0.440 | 0.863 @10 | 0.64 @10 | see note |
| Structured-TDCA + Dense | **0.48** | **0.519** | 0.60 | **0.800** | **0.935** | **0.86** | 439,513 |

The first IRCoT attempt had six truncated JSON responses. Those provider
responses are now counted against the budget and receive at most one compact
repair call. The repaired full run has zero infrastructure failures; its 0.34
EM replaces the invalid 0.32 checkpoint.

The paired exact-match difference between Structured-Dense and IRCoT is +0.14.
A 10,000-sample paired bootstrap with seed 520 gives 95% CI [-0.02, 0.30]. This
is a positive point estimate but not a statistically significant result on 50
questions. It does not support a SOTA claim.

The point improvement is not free: Structured-Dense averages 8,790 total tokens and
6.28 logical LLM calls per question, versus 5,955 and 3.38 for controlled IRCoT. Thus
the method uses about 1.48x the tokens at this checkpoint. IRCoT's low observed provider
call count is a shared-cache artifact and is not an algorithmic efficiency advantage;
logical calls and tokens are the fair comparison. Against one-call RAG, Structured uses
roughly 6.5x the tokens. Stage 4 must therefore establish a quality/cost Pareto case,
not merely a point EM lead.

Against HippoRAG, the paired EM difference is only +0.04 with 95% bootstrap CI
[-0.12, 0.20]. The methods are statistically indistinguishable on this 50-question
checkpoint; Structured-Dense's current advantage is higher F1 and selective accuracy,
not a significant EM lead.

Post-hoc error analysis (never exposed to inference) assigns the 26 non-EM rows
to: seven claim extraction/acceptance errors, six incomplete evidence retrievals,
six decomposition errors, six budget-exhaustion cases, and one verifier rejection
or missing-terminal case. Relative to IRCoT, 13 questions move from wrong to right,
while six move from right to wrong.

The HippoRAG row uses official code at pinned commit
`c617143f01477243992a63b2e2151cc003dd3b21`, Qwen-plus and MiniLM, with one
independent graph per question's 20 distractor passages. Upstream and independent
answer scoring agree exactly (EM 0.44, F1 0.4534; zero unrecovered parser failures).
Recall@5/@10 is 0.703/0.863 and strict all-gold recall is 0.40/0.64. Wall time is
1,707 seconds. Because its OpenIE cache is shared with smoke runs and this adapter
version did not snapshot pre-run cache counters, the final cumulative cache is
reported transparently (2,685 unique calls, 2,028,044 tokens), but no unsupported
tuning-only token delta is invented. This is not the paper-default NV-Embed-v2 track.

The frozen legacy batch is now complete and independently rescored against the source
aliases: EM 0.16, F1 0.2067, answered rate 0.48, average 14.1 calls and 1,355 output
tokens per question. Its reported title-hit is 1.00, yet only eight answers are exact;
this quantitatively confirms that the old bottleneck is evidence composition/final
admission rather than initial title retrieval. Legacy instrumentation does not expose
prompt tokens, so its output-only count is not compared as total cost. The old
`soft_em=0.22` remains auxiliary and is excluded from the table.

## Completed key ablations

Removing structured working memory collapses the plan to direct reasoning and lowers
EM/F1 from 0.48/0.5186 to 0.40/0.41, with answered rate 0.50. This supports a memory
contribution at the tuning point estimate, though significance is not claimed. The
paired EM difference is +0.08 with 95% bootstrap CI [-0.08, 0.24]. Mean total tokens
increase from 4,055 to 8,790 per question, so the observed point improvement carries
a substantial inference-cost tradeoff and remains to be confirmed on validation200.

Greedy, diffusion and expected-utility scheduling all produce exactly 0.48/0.5186 on
this split. Structural inspection finds 41/50 plans (82%) have maximum ready width one;
only nine have any scheduling choice. Aligned reasoning traces show that the three
schedulers actually execute a different slot order on only 3/50 questions (6%), with
no resulting answer-metric difference. This is a non-informative scheduler ablation,
not evidence that the schedulers are equivalent or that scheduling contributes to the
observed gain. Scheduler contribution must be tested on branch/comparison or
global-corpus workloads before it can be claimed.

Oracle evidence with the learned plan raises EM/F1 to 0.62/0.6533 and answered rate
to 0.74. The +0.14 EM gap over the normal method quantifies remaining retrieval and
evidence-selection headroom; the oracle is not treated as a deployable baseline.

Oracle decomposition with normal retrieval reaches EM/F1 0.58/0.59, answered rate
0.64 and selective accuracy 0.906. Its +0.10 EM gap shows a separate planning/binding
ceiling. Oracle evidence's slightly larger +0.14 gap suggests retrieval/evidence
selection is the larger of the two at this checkpoint, but both remain material.

The joint oracle reaches EM/F1 0.62/0.6433, answered rate 0.66 and selective accuracy
0.939. It does not exceed evidence-only EM: per-hop oracle evidence narrows context
under the gold plan, improving selectivity while occasionally losing useful adjacent
support. Oracle gaps are therefore interactions, not additive component effects.
