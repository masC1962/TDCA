# Dynamic Hypergraph TDCA v2.4.1 Adaptive Smoke-A Safe-Stop Report

Date: 2026-08-25

Decision: **SAFE_STOP**

Authorized next API stage: **none**

## 1. Frozen protocol and run identity

The v2.4.1 campaign was preregistered before its first provider request in
`configs/dynamic_v241_preregistration.json`. The global campaign limits were
2,000 provider attempts and 2,000,000 provider-reported tokens. The stage order
was adaptive smoke-A, paired frozen-v2.4/v2.4.1 shadow-B, smoke controls, and
development-50. A failure at smoke-A cancels every later stage.

The completed adaptive smoke-A run is:

```text
research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787648508610197923
```

Its manifest binds the result to commit
`f073e03174310f294aed263a188393d6a59acd59`, source-tree SHA-256
`7399595ca5abe87ccd5981fb72fb1ed131096c4e34c91ae28a47bc451c62abd3`,
dataset SHA-256 `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`,
Qwen-plus, seed 20260820, and the same frozen smoke-A IDs used by v2.4. The
server-container test suite passed 244/244 before inference.

The disjoint shadow-B set was frozen before inference with 20 unused IDs,
zero overlap with all frozen smoke/development/heldout IDs, and a 6/9/5 mix of
2/3/4-hop examples. It was not opened because smoke-A failed.

## 2. Implemented v2.4.1 mechanisms

1. **Proof-usable target gate.** Semantic target projection no longer closes a
   region by itself. A claim must also pass independent support, grounding,
   type, evidence-gap, contradiction, evidence-lineage, dependency-closure, and
   joined-hyperedge connectivity checks.
2. **Proof-gap recovery planner.** Missing target projection, weak support,
   missing evidence lineage, dependency closure, and JOIN precondition failures
   emit existing executable `BRANCH(extract_typed)`, `RETRIEVE`, or `MERGE`
   operations with auditable `proof_gap_reason` metadata. No generic `REPAIR`
   operation or question-specific rule was added.
3. **Feasibility feedback.** Deterministically filtered JOIN premises update the
   recovery target and expose feasibility-unlock value before allocation.
4. **No-diff editor gate.** Model-dependent graph edits that cannot guarantee a
   non-empty executable diff are rejected before allocation. The run selected
   zero no-op editor actions.
5. **Choice-conditioned EVC.** The allocator adds normalized proof-gap
   reducibility and feasibility-unlock components, blends global and
   within-family normalization, and the offline audit separately measures EVC
   calibration only at decisions with at least two distinct executable actions.

All mechanisms are default-off. Frozen v2.2, v2.3, and v2.4 configurations keep
their historical semantics.

## 3. Frozen hard-gate result

| Metric | v2.4.1 | Frozen gate | Result |
|---|---:|---:|---|
| Artifact complete | yes | yes | pass |
| Gold/oracle inference | 0 | 0 | pass |
| Infrastructure failures | 0 | 0 | pass |
| Graph invariant violations | 0 | 0 | pass |
| Controller-only mutation violations | 0 | 0 | pass |
| Unsupported answers | 0 | 0 | pass |
| Selected infeasible JOINs | 0 | 0 | pass |
| Repeated extraction fingerprints | 0 | 0 | pass |
| No-diff editor allocations | 0 | 0 | pass |
| Candidate presence | 0.75 | >= 0.60 | pass |
| Execution-plan completion | 0.75 | >= 0.65 | pass |
| Graph-proof completion | 0.80 | >= 0.75 | pass |
| F1 | 0.5843 | >= 0.4593 | pass |
| Logical LLM calls | 148 | <= 159 | pass |
| Logical tokens | 167,857 | <= 179,035 | pass |
| Budget exhaustion | 0.05 | <= 0.10 | pass |
| Spearman(EVC, immediate actual utility) | 0.12295 | >= 0.15 | **fail** |
| Choice-conditioned Spearman | 0.13852 | > 0 | pass |
| Complete EVC trace | 1.00 | 1.00 | pass |
| Non-uniform allocation | 1.00 | > 0 | pass |
| Typed terminal outcomes | yes | yes | pass |

The run therefore fails exactly one preregistered gate. Thresholds are not
relaxed after observing the result. Paired shadow-B, controls, and development-50
were not run.

Final task metrics are EM 0.55, F1 0.5843, selective accuracy 0.7333, candidate
presence 0.75, execution-plan completion 0.75, graph-proof completion 0.80, and
full-chain correctness 0.55. Outcomes are 15 ANSWER, 4 ABSTAIN, and 1
BUDGET_EXHAUSTED. Campaign usage is 142 provider attempts and 161,189
provider-reported tokens.

## 4. Comparison with frozen v2.4

| Metric | v2.4 | v2.4.1 | Delta |
|---|---:|---:|---:|
| EM | 0.40 | 0.55 | +0.15 |
| F1 | 0.4593 | 0.5843 | +0.1250 |
| Candidate presence | 0.50 | 0.75 | +0.25 |
| Execution-plan completion | 0.60 | 0.75 | +0.15 |
| Graph-proof completion | 0.75 | 0.80 | +0.05 |
| Logical calls | 140 | 148 | +8 |
| Logical tokens | 150,832 | 167,857 | +17,025 |
| EVC correlation | 0.1344 | 0.12295 | -0.01145 |

The new recovery policy repaired four exact-answer failures and five execution
plans relative to v2.4, including the previously lost
`2hop__89764_827343`, `2hop__424908_500483`, and
`3hop1__145924_131905_41948` regions. It lost two execution plans and two exact
or partial answers, so the improvement is substantial but not monotonic.

The graph mechanism itself is efficient and auditable. All 21 selected JOINs
were accepted, all 21 reached accepted-answer support, and every JOIN used zero
provider calls. The feasibility filter removed 7.75 infeasible candidates per
question on average while selecting none. Of 61 retrievals, 63.9% contributed
evidence used by an accepted answer; first retrievals within a subgoal had mean
utility +0.1821, while second retrievals had mean utility -0.0443. Extraction
accepted 69.0% of attempts, but 23.9% returned no raw claim rows. The five
non-answer bottlenecks are two extraction failures, one retrieval failure, one
proof/JOIN closure failure, and one budget exhaustion.

## 5. Why EVC failed despite better task performance

The failure is not missing trace coverage or absence of real choices. There are
267 selected allocations, 56 decisions with at least two distinct executable
actions, a 20.8% real-operation-choice rate, and a positive choice-conditioned
correlation of 0.1385.

The main problem is a horizon mismatch between the prediction and its target:

- `proof_gap_reducibility` predicts downstream option value: a retrieval may be
  useful because a later extraction, verification, and JOIN closes the proof.
- `actual_utility` is reconciled immediately after one operation. It charges the
  retrieval cost before downstream proof closure can receive credit.
- All 13 selected proof-gap-tagged recovery actions therefore have negative
  immediate utility, even though recovery occurs in successful cases such as
  `2hop__89764_827343`, `3hop1__801799_547811_41132`, and
  `3hop2__326964_7845_7713`.

This mismatch also appears within families. Spearman correlation is -0.8925 for
retrieval and -0.3325 for JOIN, although retrieval has positive mean utility and
every selected JOIN is accepted and answer-used. Conversely, low-fidelity
actions correlate positively (+0.4592), while medium-fidelity correlation is
-0.6915. The allocator currently boosts proof-gap opportunity signals by
marginal-efficiency scaling at low fidelity, which can make the prediction
larger than its original [0,1] opportunity value and further confound fidelity
with long-horizon value.

The scientific conclusion is therefore narrow: v2.4.1 materially improves
proof recovery and answer quality under the frozen budget, but the claimed EVC
quantity is not yet calibrated to the realized utility quantity used by the
gate. The failed correlation prevents a computation-allocation claim.

## 6. Recommended v2.4.2 direction

No additional API run should start from v2.4.1. The next iteration should remain
training-free, preserve all safety and proof-recovery behavior, and fix credit
assignment before changing policy weights.

1. **Separate horizons.** Record `immediate_utility` and a distinct delayed
   `realized_proof_return`; do not relabel one as the other. Predicted immediate
   gain and predicted option value must be audited against their matching
   targets.
2. **Controller-owned delayed credit.** Add an immutable credit-assignment
   ledger that attributes later claim creation, feasible JOIN unlock, proof-gap
   closure, and accepted-answer use back to causally linked retrieval/extraction
   operations through graph provenance. Use a preregistered finite horizon or
   terminal event, without learned weights or gold fields.
3. **Fix fidelity semantics.** Proof-gap reducibility and feasibility unlock are
   opportunity channels and should scale with expected gain, not inverse-cost
   efficiency. Clamp all normalized opportunity channels to [0,1] before the
   additive EVC readout.
4. **Family-local calibration.** Require non-negative or explicitly explained
   calibration for substantive operation families, especially retrieval and
   JOIN, rather than relying on a pooled between-family correlation.
5. **Offline counterfactual fixtures.** Recompute delayed returns on frozen v2.4
   and v2.4.1 traces, verify that successful recovery receives positive delayed
   credit and useless second retrievals remain negative, then preregister a new
   v2.4.2 smoke gate. Do not reuse or relax the failed v2.4.1 threshold.

The shadow-B manifest remains unopened and can be retained for a future paired
v2.4.2 evaluation, provided its IDs and labels remain uninspected.
