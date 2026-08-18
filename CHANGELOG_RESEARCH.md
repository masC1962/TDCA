# Research Changelog

## 0.1.0

- Preserved a runnable legacy TDCA entrypoint and source manifest.
- Added a standalone `tdca_research` package with strict configuration and budgets.
- Added multi-format dataset adapters and gold-isolated inference views.
- Added validated dependency DAGs and explicit variable binding.
- Added typed working-memory claims with verification, contradiction and supersession.
- Added BM25, explicit dense dependency handling and hybrid retrieval.
- Added greedy, best-first, beam, stable diffusion and expected-utility schedulers.
- Separated extraction, verification and finalization.
- Added answer/retrieval/calibration metrics and versioned run artifacts.
- Added controlled baseline adapters and external baseline provenance manifests.
- Added unit, property-oriented, integration, regression and 20-row offline smoke tests.
- Added reusable fingerprinted dense indices and a generic entity-aware retriever.
- Added real ablation controls for memory/DAG/binding/verifier/scheduler/finalization.
- Added dataset-integrity gates and fixed HotpotQA/2Wiki parallel-context parsing.
- Added per-example reasoning/retrieval/selectivity metrics and paired-bootstrap inputs.
- Added a fail-closed pinned-commit launcher for official external baselines.
- Fixed MuSiQue evidence evaluation so repeated titles do not expand one supporting
  paragraph into same-title distractors; answer EM/F1 were unaffected.
- Added a pre-LLM retrieval probe and process-scoped SentenceTransformer model reuse.
- Completed real Qwen-plus smoke-20 runs for BM25/Dense RAG, controlled IRCoT and
  Structured-TDCA with BM25 and Dense retrieval.
- Pinned and verified HippoRAG 2, isolated its dependencies, ran official-code
  end-to-end canaries and saved an aligned mini-global smoke-20 artifact.
- Added independent HippoRAG answer-parser auditing and token/call cache accounting.
- Added an explicitly budgeted one-shot IRCoT structured-output repair path for
  provider responses truncated at the JSON token boundary.
- Added secure container launchers that load Qwen credentials only from the root-owned
  environment file and pin the working Hugging Face mirror endpoint.
- Added a frozen legacy batch wrapper and independent legacy EM/F1 rescoring adapter;
  the legacy algorithm implementation remains unchanged.
- Completed frozen tuning-50 Structured-Dense, BM25/Dense/Hybrid RAG and repaired
  controlled-IRCoT runs; official HippoRAG and legacy tuning runs are tracked separately.
- Completed official-code controlled HippoRAG 2 and frozen legacy TDCA tuning-50 runs,
  independent rescoring, paired bootstrap analysis, memory/scheduler ablations and
  evidence/decomposition oracle ceilings.
- Added pinned full 2Wiki dev preparation with source-hash verification, deterministic
  full HotpotQA/2Wiki staged manifests, strict dataset audits and cross-dataset configs.
- Aligned yes/no/noanswer F1 behavior with the official HotpotQA/2Wiki scorer, and added
  evaluation-only by-hop/by-type reports without exposing labels to inference.
- Added per-question long-run checkpoints, a shared grouped-metrics implementation and
  k-selectable retrieval diagnostics with by-hop reporting.
- Completed frozen MuSiQue validation-200 Structured-Dense and controlled IRCoT runs,
  independent rescoring, paired EM/F1 bootstrap and depth/error/cost diagnostics; the
  Stage-5 gate remains closed because there is no significant or Pareto advantage.
- Pinned the official HotpotQA scorer and verified unified answer EM/F1 parity with
  official HotpotQA and 2Wiki implementations on normalization/type edge cases.
- Added fail-closed exact-prefix resume for interrupted native runs and deterministic
  source-tree version hashes for mounted workspaces without Git metadata.
- Added optional generic verbatim query-sentence evidence compaction, default off and
  explicitly excluded from post-hoc validation-200 claims.
- Added one-command research/legacy/compile verification through `scripts/test_all.sh`.
- Tightened dataset integrity to reject duplicate IDs, empty evaluation answers and
  missing/out-of-context gold evidence; full MuSiQue, HotpotQA and 2Wiki pass.
- Made HTTP timeout/retry policy explicit and separated logical LLM calls, cache hits
  and actual provider attempts for all newly launched native experiments.
- Added completed-artifact checksum/count/order verification with an explicit,
  non-mutating compatibility mode for pre-checkpoint historical artifacts.
- Made native JSON checkpoints atomic and fsynced per-question JSONL checkpoint appends
  before advancing the durable progress marker.
- Added fail-closed HippoRAG validation postprocessing with exact qid-set checks,
  independent scoring, by-hop metrics, paired bootstrap, cache deltas and input hashes.
- Completed official-code controlled HippoRAG validation-200; it significantly
  outperforms frozen Structured-TDCA on both EM and F1 while using substantially more
  Qwen calls/tokens, so the Stage-5 gate remains closed.
- Completed frozen HotpotQA and 2Wiki smoke-20/tuning-50 transfer runs, strict artifact
  audits, independent scoring, paired bootstrap, by-hop and failure/cost diagnostics.
- Added fail-closed cross-dataset smoke-to-tuning authorization and monitored sequential
  launchers; tuning-50 did not meet the quality condition for expansion to 200.
- Added compact postprocessing summaries and fixed Unicode scorer fixtures against
  transport/editor encoding corruption.
