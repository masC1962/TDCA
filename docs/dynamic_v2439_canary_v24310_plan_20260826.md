# TDCA v2.4.3.9 canary diagnosis and v2.4.3.10 source freeze

## v2.4.3.9 independent canary

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787753836879718249`.
- Complete: 3/3 with zero infrastructure, invariant, and unsupported-answer
  failures.
- Provider usage: 29 attempts and 43,444 provider-reported tokens; cache hits: 0.
- The quantitative alias correction fired and allowed a `fraction` claim to
  participate in an auditable JOIN for a `numerical` slot.
- The two JOIN-dead-end cases did not reproduce their parent preallocation
  rejection trajectories, so the independent canary did not causally test the
  semantic retry key. All three examples stopped at the per-question safety
  budget, and Smoke-A remained closed.

## Frozen counterfactual replay

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787754575216942291`.
- Exact parent responses were copied into a fresh cache namespace; 27/32 logical
  calls were cache hits. Five post-divergence requests consumed 6,593 provider
  tokens and were charged in a separate ledger.
- The replay preserved the parent dead ends: four feasible, goal-relevant JOINs
  remained in the final graphs. All four can be materialized by an existing
  provider-free rule from sealed independent scores.
- Nevertheless the allocator represented every `MERGE` packet as one provider
  call with a positive token budget. Meta-stop therefore treated these four
  zero-cost transitions as unaffordable or below net value and terminated before
  materialization.

The replay did not satisfy the v2.4.3.9 opening gate and Shadow-B remains closed.
It isolates resource typing, rather than JOIN retry identity, as the active
control-flow defect.

## v2.4.3.10 semantic boundary

v2.4.3.10 refactors the existing deterministic JOIN paths into one pure
derivation proof shared by execution, resource accounting, and transition
certification. For a concrete candidate, the controller now records whether the
JOIN requires zero or one provider call. A proved provider-free JOIN receives a
zero-call, zero-token packet and a sealed transition certificate whose realized
claim and accepted JOIN signature are audited after controller application.

No new JOIN rule is introduced: numeric comparison, independently verified
projection, and goal-aligned symbolic path composition are the same accepted
derivations already present in v2.4.3.9. Thresholds, terminal gates, JOIN
feasibility, JOIN attempt caps, and per-question budgets are unchanged.

The offline preflight over the counterfactual final graphs finds four viable
JOINs and certifies all four as provider-free. A zero-API unit test also proves
that the allocator and meta-stop execute such a JOIN when both LLM-call and token
budgets are exhausted, and verifies the realized transition record.

## Budget

Cumulative use before v2.4.3.10 is 1,286 provider attempts and 1,490,369
provider-reported tokens. Remaining global allowance is 714 attempts and 509,631
tokens under the unchanged 2,000 / 2,000,000 hard cap.
