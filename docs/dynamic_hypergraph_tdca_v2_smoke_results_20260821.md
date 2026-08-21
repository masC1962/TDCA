# Dynamic Hypergraph TDCA v2: smoke-20 development result

## Decision

The v2 implementation exercises the proposed mechanisms, but the frozen hard gate
remains **closed**. No development-50, heldout-200, budget-curve, or cross-dataset
heldout result is claimed from this revision. This is an explicit negative result.

The final smoke artifact is
`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787319567455752323`.
It used Qwen-plus, seed `20260820`, no oracle inference fields, and the 16-call /
16,000-logical-token / 8-retrieval development safety caps.

## Result

| Metric | frozen v1 smoke-20 | v2 smoke-20 | v2 - v1 |
|---|---:|---:|---:|
| Exact match | 0.350 | 0.250 | -0.100 |
| F1 | 0.410 | 0.295 | -0.115 |
| Answered rate | 0.600 | 0.350 | -0.250 |
| Selective accuracy | 0.583 | 0.714 | +0.131 |
| Candidate presence | 0.550 | 0.450 | -0.100 |
| Full-chain completion | 0.500 | 0.250 | -0.250 |
| All-gold-document recall | 1.000 | 0.750 | -0.250 |
| Logical LLM calls | 140 | 184 | +44 |
| Retrieval calls | 74 | 72 | -2 |

V2 produced 10 auditable three/four-hop examples containing explicit multi-premise
JOIN hyperedges, versus the required minimum of three. Its mean JOIN count was 1.55.
Every example had non-uniform resource packets and a complete EVC ledger. Across 314
selected allocations, the exported trace count exactly matched the graph ledgers;
predicted EVC and measured cost were present for all of them. Infrastructure failure
rate and unsupported accepted-answer count were both zero.

The quality loss is therefore not an infrastructure artifact. The dominant failure
is coverage: strict explicit-chain execution and exploratory relational joins spend
more computation before terminal commitment, producing 45% budget exhaustion and
20% abstention. Stronger selective accuracy shows that three-way termination is
conservative, but it does not compensate for lost answer coverage.

## Hard-gate audit

Passed:

- zero leakage, zero invariant violation, controller-only mutation;
- at least three auditable 3/4-hop JOIN cases (observed: 10);
- adversarial non-destructive revision test;
- non-uniform allocation;
- complete EVC trace, predicted EVC, and actual-cost recording;
- no unsupported ANSWER and exact ANSWER/ABSTAIN/BUDGET_EXHAUSTED separation.

Failed:

- candidate presence gain of at least v1 +0.10;
- full-chain completion gain of at least v1 +0.10;
- natural revision precision, because zero natural revisions occurred;
- Pareto improvement against the aligned v1 control.

The machine report is `hard_gate_smoke.json` inside the final smoke artifact. The
gate remains `status: closed`; heldout launchers continue to fail closed.

## Mechanism findings

Four-way endpoint unification substantially improves visible multi-hop structure,
but unbounded shared-endpoint validation creates a combinatorial search tax. The
final implementation caps JOIN attempts per question, prioritizes dependency-lineage
coverage, commits the shortest sufficient chain, and uses a zero-generation
deterministic projection hyperedge when Qwen's independent raw scoring has already
validated both premises and their exact dependency binding. Open-endpoint joins that
create a new relation still require separate Qwen validation over grounded spans.

These controls improved auditability and reduced wasted generation, but did not
recover enough terminal coverage under the safety cap. No relation-name, entity,
question-ID, or answer-specific rule was introduced during development.

## Next research step

1. Add a typed n-ary/conjunctive JOIN for set intersection and shared-role
   constraints; pairwise path composition is inadequate for several failures.
2. Feed observed JOIN acceptance and downstream uncertainty reduction back into EVC,
   so repeatedly rejected join families lose heat immediately.
3. Freeze a natural-contradiction evaluation set separately from ordinary QA;
   MuSiQue distractors produced no meaningful natural revision sample.
4. Add a uniform-budget/fixed-order v2 ablation before attributing a Pareto gain to
   adaptive allocation rather than the graph representation.
5. Re-run smoke-20 and only launch development-50 after candidate presence and
   full-chain completion exceed their frozen v1 thresholds.
