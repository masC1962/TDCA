# Known Limitations

- The current reference scheduler uses hand-defined but general signal proxies. Its
  expected-information interpretation must be validated experimentally. On tuning-50,
  only 18% of plans expose more than one ready slot and scheduler choice changes the
  executed order on only 6% of questions; all tested schedulers have identical EM/F1.
  The current performance claim therefore belongs to structured memory/dependency
  gating, not to a demonstrated scheduler advantage.
- Qwen-plus planning, extraction, verification and finalization have been measured on
  frozen MuSiQue smoke-20, tuning-50 and validation-200 splits in the Linux container.
  Validation-200 does not show a significant advantage over controlled IRCoT and is
  not cost-Pareto-superior, so Stage 5 remains closed. Frozen HotpotQA and 2Wiki
  smoke-20/tuning-50 runs are complete; both favor IRCoT on F1 point estimates and
  Structured is substantially more expensive, so cross-dataset validation-200 is not
  expanded under the staged accuracy gate.
- Dense retrieval and official HippoRAG 2 code now run in the Linux GPU container;
  the completed controlled validation-200 significantly outperforms Structured-TDCA,
  but at substantially higher call/token cost.
  The controlled HippoRAG track uses MiniLM rather than paper-default NV-Embed-v2;
  full official-corpus indexing and the paper reproduction track remain pending.
- The completed HippoRAG tuning artifact records cumulative shared-cache counters but
  the original shared-cache path is no longer present. A validation run must therefore
  use and record a fresh explicit cache directory; tuning-only cache reuse or token
  deltas must not be inferred retrospectively.
- The in-memory BM25 implementation prioritizes transparency over corpus-scale speed.
- Oracle decomposition conversion currently targets chain-style numbered references;
  dataset-specific official DAG conversion needs fixtures for all branch formats.
- Answer normalization has regression parity with the bundled official MuSiQue scorer
  and the pinned official HotpotQA and 2Wiki scorers on nine normalization/type edge
  cases each. This verifies answer EM/F1 semantics, not supporting-fact joint metrics.
- The checked-in compact MuSiQue file is support-only; 8/50 compact Hotpot rows lack
  their annotated supporting titles. Runtime validation prevents either from being
  mislabeled as valid distractor evaluation. Full HotpotQA is local. Full 2Wiki dev is pinned to the Apache-2.0
  Hugging Face official mirror commit `612bc503...`; the original Dropbox endpoint
  times out from the server, so mirror provenance and the source hash are retained.
- No SOTA claim is made.
- A 2026 literature refresh identified SAG and Youtu-GraphRAG plus the official HopRAG
  repository. They are not silently mixed into the distractor table: SAG needs separate
  embedding/reranker endpoints and database services, while HopRAG/Youtu require a
  frozen global-corpus graph-build protocol. CIRAG is excluded from the training-free
  controlled track because its core integration policy is trajectory-distilled with LoRA.
- Held-out confidence is overconfident (validation-200 ECE 0.187); the field named
  `calibrated_confidence` is currently a verifier-derived selection score, not a reliable
  probability. Future calibration must be fit/frozen on tuning data only.
- Controlled IRCoT has an explicitly budgeted one-shot JSON repair. The structured
  planner/extractor/verifier/finalizer still fail closed on malformed JSON rather than
  sharing the same repair helper. Unifying repair semantics is future work and would
  require a new frozen experiment because it changes calls/tokens.
- Historical artifacts define `provider_calls` as uncached logical calls because the
  original client did not expose underlying HTTP retry counts. New runs record actual
  `provider_attempts` separately, with explicit timeout/attempt policy and SDK retries
  disabled. Do not compare the historical and new provider-call columns as identical
  semantics; prompt/completion tokens remain provider-reported in both.
- New artifacts separate logical/model-equivalent prompt/completion tokens (including
  cache hits, used for algorithmic budget comparisons) from uncached successful
  provider tokens (used for incremental cost accounting). Failed HTTP attempts may not
  expose token usage; they are still counted in `provider_attempts`.
