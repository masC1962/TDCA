# Dynamic Hypergraph TDCA v2 closure result (2026-08-22)

## Decision

The implementation now contains the requested training-free n-ary hypergraph JOIN,
operation-outcome feedback, adaptive/uniform/fixed allocators, public natural
revision suite, matched-compute audit, API ledger, and fail-closed heldout gate.
Offline causal and invariant tests pass. The MuSiQue smoke quality gate does not
pass, so no development-50, matched allocator, frozen revision evaluation, or
heldout experiment was opened.

The final permitted smoke artifact is:

`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787405220660799660`

## Result

| Metric | frozen v1 | prior v2 | closure v2 |
|---|---:|---:|---:|
| Exact match | 0.350 | 0.250 | 0.150 |
| F1 | 0.410 | 0.295 | 0.150 |
| Candidate presence | 0.550 | 0.450 | 0.350 |
| Full-chain completion | 0.500 | 0.250 | 0.050 |
| Infrastructure failure rate | 0.000 | 0.000 | 0.000 |
| Unsupported accepted answers | 0 | 0 | 0 |
| Complete outcome-feedback trace | n/a | n/a | 1.000 |
| Feedback influenced a later allocation | n/a | n/a | 0.450 |
| Natural accepted n-ary JOINs | n/a | n/a | 0 |

The closure campaign used 234 uncached provider calls and 271,227
provider-reported tokens across five bounded smoke iterations, below the registered
caps of 1,500 calls and 1,500,000 tokens. The machine ledger is
`analysis_outputs/dynamic_v2_closure_campaign_usage.json`.

## What was fixed

- Every selected action now has an allocation-namespaced operation ID, including
  failed actions that do not advance graph step. This removed a real duplicate JOIN
  audit-ID crash.
- N-ary discovery enumerates only connected typed constraint frontiers, requires
  independent premise support, records accepted and rejected attempts, prevents
  lineage cycles, and supports explicit set intersection.
- Redundant sibling candidates are no longer accepted as necessary projection
  premises. N-ary work enters model validation only when it has structural evidence
  of additional conjunctive value.
- Every selected allocation records predicted EVC, actual cost, pre/post graph
  summaries, component-level actual utility, and posterior updates.
- Feedback is keyed by exact operation context. New evidence batches never inherit
  penalties from earlier evidence; identical failed actions do.
- The public VitaminC suite is case-disjoint and label-separated, with 30 positive
  and 30 negative frozen examples.
- The heldout runner requires a passing machine report; editing the preregistration
  JSON cannot open it.

## Why the gate remains closed

The final deterministic failure taxonomy attributes the 17 non-perfect smoke cases
primarily to six claim-extraction misses, six recoverable provider/structured-output
events, two JOIN verification failures, two retrieval misses, and one type/binding
mismatch. These are trace-derived symptoms, not post-hoc per-question repair
targets. The report is in
`analysis_outputs/dynamic_v2_closure_smoke_final/failure_taxonomy.json`.

The mechanism tests prove n-ary execution, set intersection, causal feedback,
revision, and allocator separation. They do not justify a quality or Pareto claim.
On natural MuSiQue smoke data, no accepted n-ary JOIN reached downstream use and the
full-chain rate regressed. Running larger samples would therefore spend compute
without satisfying the preregistered expansion condition.

## Reproduction

```bash
bash scripts/run_dynamic_v2_research.sh tests
bash scripts/run_dynamic_v2_research.sh smoke20
python scripts/analyze_dynamic_v2_failures.py \
  --run research_outputs/<smoke-run> \
  --baseline-commit e4dac8c7621b0d0e28534954260eec23feea1012 \
  --output-dir analysis_outputs/<name>
```

Do not run `development50`, `matched_allocators50`, `revision_evaluation`, or any
heldout stage from this result. The appropriate next research step is to improve
training-free claim extraction and reduce structured-output fragility on a new,
declared development revision—not to add question-specific patches or relax the
hard gate.
