# Dynamic Hypergraph TDCA v2.4 Adaptive Smoke-20 Safe-Stop Report

Date: 2026-08-25

Decision: **SAFE_STOP**

Authorized next API stage: **none**

## 1. Frozen protocol

The v2.4 campaign was preregistered before any v2.4 provider call in
`configs/dynamic_v24_preregistration.json`. Its global safety limits are 2,000
provider attempts and 2,000,000 provider-reported tokens. The first authorized
stage was one adaptive Qwen-plus smoke-20 run. Any failed hard gate blocks all
matched controls and development-50 runs.

The frozen adaptive run is:

```text
research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787644117936463856
```

The run completed all 20 frozen IDs, passed artifact checksums, and used the
source checkpoint formed by commits `aae08236` and `705a0c23`.

## 2. Implemented v2.4 mechanisms

1. **JOIN pre-allocation feasibility** factors the existing independent premise
   support gate into a pure, zero-call predicate. Deterministically infeasible
   JOINs are removed before allocation and receive explicit reason codes.
2. **Bounded extraction recovery** permits at most one coverage extraction and
   one direct-answer extraction for an unchanged evidence/dependency/revision
   fingerprint. New evidence or a relevant revision opens a new fingerprint.
3. **Region-level retrieval stopping** requires a materially novel query and an
   explicit recovery reason before a second retrieval in the same graph region.
4. **Graph-proof audit** separately reports dependency coverage, evidence-leaf
   coverage, evidence independence, connectivity, depth, and graph-proof
   completion. The old `full_chain_complete` field is retained only as a
   backward-compatible alias of execution-plan completion.

All features are opt-in. Frozen v2.2 and v2.3 configs retain their old behavior.
The complete zero-API test suite passed: 232/232.

## 3. Frozen gate result

| Metric | v2.4 | Frozen gate | Result |
|---|---:|---:|---|
| Infrastructure failures | 0 | 0 | pass |
| Graph invariant violations | 0 | 0 | pass |
| Controller-only mutation violations | 0 | 0 | pass |
| Unsupported answers | 0 | 0 | pass |
| Selected infeasible JOINs | 0 | 0 | pass |
| Repeated extraction fingerprints | 0 | 0 | pass |
| Candidate presence | 0.50 | >= 0.60 | **fail** |
| Execution-plan completion | 0.60 | >= 0.65 | **fail** |
| Graph-proof completion | 0.75 | >= 0.75 | pass |
| F1 | 0.4593 | >= 0.3550 | pass |
| Logical LLM calls | 140 | <= 174 | pass |
| Logical tokens | 150,832 | <= 196,939 | pass |
| Budget exhaustion | 0.00 | <= 0.10 | pass |
| Spearman(EVC, actual utility) | 0.1344 | >= 0.15 | **fail** |
| Complete EVC trace | 1.00 | 1.00 | pass |
| Non-uniform allocation | 1.00 | > 0 | pass |

Additional outcome metrics are EM 0.40, selective accuracy 0.6667, zero budget
exhaustion, 12 ANSWER outcomes, and 8 ABSTAIN outcomes. Campaign usage after the
safe stop is 138 provider attempts and 148,554 provider-reported tokens.

## 4. What improved

Against frozen v2.2 smoke-20, v2.4 reduces logical calls from 159 to 140,
logical tokens from 179,035 to 150,832, and retrieval calls from 63 to 50 while
raising F1 from 0.3550 to 0.4593 and EM from 0.30 to 0.40. Graph-proof completion
matches the preregistered v2.2 recalculation at 0.75.

The region gate eliminated repeated within-subgoal retrieval entirely in this
run. All 50 retrievals were first attempts in their respective subgoal/branch
regions; their mean measured utility was positive. The former v2.3.3 pattern of
27 low-value second retrievals is therefore removed rather than merely assigned
a smaller soft penalty.

JOIN execution also became efficient: 18 of 18 selected JOINs were accepted,
17 reached accepted-answer support, and all were deterministic zero-provider-call
compositions. The pre-allocation filter removed an average of 7.2 unsupported
JOIN candidates per question, while no deterministically infeasible JOIN was
selected. No extraction fingerprint was repeated.

Paired against v2.2, v2.4 gained candidate, chain, and exact match together on
three questions. This confirms that the new policy can convert saved compute
into deeper successful proofs, including 3-hop and 4-hop cases.

## 5. Why the hard gate still failed

The paired result is non-monotonic: three candidate/chain gains are offset by
three candidate losses and four execution-chain losses. The lost chains are:

```text
2hop__424908_500483
2hop__89764_827343
3hop1__145924_131905_41948
3hop2__90327_87184_76291
```

The dominant terminal pattern is now `no_executable_computation`, not budget
exhaustion. In three lost cases no JOIN was selected because the graph never
formed a feasible verified premise frontier. In the fourth, one JOIN was
accepted but did not close the answer proof. Twelve graph-editor EXPAND
operations were selected across the run and all twelve were no-ops with negative
mean utility. Thus the current fallback after retrieval/extraction saturation is
not a useful proof repair operation.

Extraction remains the largest non-answer bottleneck: 5 of 8 non-answers are
classified as context-to-candidate failures, with 9 of 57 extraction attempts
empty. The current `answers_subgoal` decision can also suppress focused recovery
when a superficially direct but structurally unusable claim exists. This is a
generic state-definition problem, not a single-question exception.

EVC calibration is positive overall but misses the frozen threshold by 0.0156.
Within-family rank correlations remain negative for retrieval, extraction,
verification, and JOIN. The aggregate positive value is therefore partly a
between-family effect; it is not yet strong evidence that EVC correctly orders
similar competing operations.

## 6. Recommended v2.4.1 direction

No v2.4 controls or larger runs should be launched from this checkpoint. The
next iteration should remain training-free and preserve all safety fixes.

1. Replace no-op EXPAND fallback with a deterministic **proof-gap recovery
   planner**. It should consume the graph-proof audit's missing dependency,
   missing evidence leaf, or missing hyperedge reason and emit only operations
   that can close that structural gap.
2. Split `answers_subgoal` into a weak semantic projection signal and a strong
   **proof-usable target claim** predicate. Only the latter should suppress the
   bounded direct-answer recovery.
3. Feed JOIN feasibility reasons back into scheduling. A nearly feasible premise
   should create a targeted evidence/verification repair candidate; an invalid
   or archived premise should not.
4. Recalibrate EVC within operation family using normalized graph-state deltas,
   especially proof-gap reduction and feasibility unlock. Keep predicted EVC,
   actual utility, and cost as separate auditable channels.
5. Add offline fixtures for the four lost structural patterns and require the
   proof-gap recovery to resolve them without gold fields or question-specific
   rules before authorizing a new smoke campaign.

The next smoke must use a new version/config/campaign ID and a newly frozen gate;
the failed v2.4 result and thresholds must remain immutable.
