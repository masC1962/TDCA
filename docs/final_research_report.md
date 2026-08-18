# Final research report for the current checkpoint

## Bottom line

The repository has been converted from a monolithic, patch-heavy TDCA prototype into
a training-free multi-hop QA research framework with a frozen legacy baseline,
dependency DAGs, typed and evidence-provenanced working memory, explicit variable
binding, modular retrieval/scheduling/verification/finalization, official-compatible
scoring, strict experiment artifacts and fail-closed staged evaluation.

The engineering hypothesis is implemented and testable, but the empirical hypothesis
is not supported at this checkpoint. Structured-TDCA does not beat the strongest
reproducible baselines on held-out MuSiQue and transfers poorly to HotpotQA and
2WikiMultiHopQA. It must not be described as SOTA.

## What is implemented and verified

- The original implementation is frozen under `legacy/tdca_v0/` with a runnable mock
  entrypoint and regression tests. The new package does not import the old scheduler.
- MuSiQue, HotpotQA and 2Wiki adapters, disjoint seed-520 split manifests, leakage
  tests and official-scorer parity are in place.
- Sparse, dense, hybrid and dependency-aware retrieval are explicit. Dense dependency
  failures are errors rather than silent TF-IDF substitutions.
- Structured-TDCA, closed-book, BM25/Dense/Hybrid RAG, controlled IRCoT, legacy TDCA
  and oracle modes share one evaluation and artifact layer.
- Official HippoRAG 2 is pinned and isolated in its own environment. Controlled
  canary, smoke-20, tuning-50 and MuSiQue validation-200 runs are complete.
- Every current long-run artifact has ordered IDs, hashes, traces, failures, cost
  summaries and resume checkpoints. Postprocessors independently rescore predictions
  and audit artifact integrity before comparison.

## Empirical verdict

On frozen MuSiQue validation-200, Structured-TDCA obtains EM/F1 0.400/0.453 versus
controlled IRCoT 0.380/0.473. Paired differences are not significant, while Structured
uses 1.55 times as many tokens. Official-code controlled HippoRAG 2 reaches
0.480/0.552; its advantage over Structured is significant for both EM and F1.

On disjoint cross-dataset tuning-50, Structured/IRCoT reach 0.480/0.630 versus
0.520/0.722 on HotpotQA and 0.520/0.650 versus 0.620/0.774 on 2Wiki. Structured uses
3.59 and 2.42 times as many logical tokens, respectively. Therefore neither the
MuSiQue final-1,000 gate nor the cross-dataset validation-200 gate is open.

## Main defects identified by evidence

1. **Evidence-to-answer synthesis is the dominant transfer failure.** Both
   cross-dataset tuning runs have support and all-gold recall of 1.0, yet final
   synthesis accounts for 14/26 Hotpot errors and 17/24 2Wiki errors.
2. **Verification loses valid terminal answers.** False rejection or missing terminal
   coverage accounts for eight Hotpot and five 2Wiki errors. Strict grounding is useful,
   but current verifier recall is too low.
3. **The dependency scheduler is often observationally inactive.** Only 28% of Hotpot
   and 18% of 2Wiki plans expose multiple ready slots, so the proposed scheduling rule
   rarely faces a real choice. A scheduler contribution cannot be inferred from these
   runs alone.
4. **Long-chain generalization is weak.** MuSiQue four-hop validation accuracy drops
   materially, and the six-example 2Wiki four-hop slice is too small to rescue or
   refute that result.
5. **The method is inefficient.** Extra planning, claim and verification calls do not
   yield a quality-cost Pareto advantage against IRCoT; HippoRAG is more expensive but
   significantly more accurate on MuSiQue.
6. **Global-corpus evidence is incomplete.** A frozen-ID mini-global HippoRAG run and
   reusable-index machinery exist, but a paper-default, full global-corpus reproduction
   has not been completed.

## Next research direction

The next version should be a new algorithm line, not a patch to frozen validation
outputs. The most defensible sequence is:

1. Build a new disjoint development split and keep all existing validation/final IDs
   sealed.
2. Replace free-form final synthesis with a generic typed terminal-candidate lattice:
   retain all evidence-supported spans, normalize only by answer type, and separately
   score entailment and minimality. Do not add entity, dataset or question templates.
3. Optimize verifier recall using calibrated candidate ranking plus explicit
   contradiction checks; measure candidate-oracle recall before changing thresholds.
4. Make planning adaptive: bypass the DAG/scheduler when only one executable path
   exists, and reserve graph search for genuine branches. This should cut calls without
   pretending scheduling helped on linear questions.
5. Evaluate each change first on mechanism-level counterfactual tests and the new
   development split, then preregister a fresh smoke/tuning sequence. Reopen a frozen
   validation gate only after quality and cost criteria pass.
6. For the global track, reproduce one official paper-default HippoRAG protocol before
   adding further GraphRAG systems. KiRAG/CIRAG require training and belong outside the
   training-free main comparison; SAG, HopRAG and Youtu-GraphRAG currently carry
   infrastructure or license/protocol constraints recorded in the baseline manifest.

## Status labels

- **Locally verified:** refactored framework, tests, controlled distractor experiments,
  controlled official-code HippoRAG runs and saved comparisons.
- **Controlled reimplementation:** IRCoT and native RAG baselines.
- **Official code, controlled configuration:** HippoRAG 2 with Qwen-plus and MiniLM;
  this is not a paper-default reproduction.
- **Unverified/pending:** paper-default HippoRAG global protocol and additional official
  global GraphRAG baselines.
- **Intentionally not run:** MuSiQue final-1,000 and cross-dataset validation-200,
  because their preregistered quality gates failed.
