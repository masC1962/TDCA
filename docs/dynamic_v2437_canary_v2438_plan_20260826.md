# TDCA v2.4.3.7 canary diagnosis and v2.4.3.8 source freeze

## v2.4.3.7 recovery canary

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787746255871740616`
- Complete: 8/8 with zero infrastructure, invariant, and unsupported-answer failures.
- Provider usage: 69 attempts and 84,118 provider-reported tokens.
- Candidate / legacy chain / graph proof: 0.50 / 0.25 / 0.625.
- Immediate / delayed / choice-conditioned delayed Spearman: 0.700 / 0.245 / -0.161.
- Four retrieval allocations targeted proof-quality obligations, but all persisted
  retrieval-attempt rows had an empty `recovery_policy`; consequently the new
  `proof_recovery_extraction_priority` event fired zero times.

The canary therefore did not test the intended v2.4.3.7 behavior. Its metric
movements cannot be attributed to recovery evidence freshness.

## Root cause

The scheduler allocates a placeholder RETRIEVE operation containing the recovery
label. After retrieval, the executor constructs a concrete operation containing
query and evidence but not placeholder-only metadata. The attached allocation
record correctly retains its nonempty target-obligation IDs, so controller-owned
state is sufficient to reconstruct provenance without trusting model output.

## v2.4.3.8 semantic boundary

v2.4.3.8 adds one opt-in deterministic rule. When a concrete RETRIEVE operation
is committed, the controller labels it as proof-gap recovery if and only if its
sealed allocation targets at least one controller-owned
`insufficient_target_proof` obligation. Any provider-supplied recovery label is
ignored under this mode. No score, threshold, terminal gate, JOIN gate, query,
or per-question behavior changes.

The next stage is a three-example diagnostic canary selected solely because the
v2.4.3.7 traces contained such targeted allocations. It must show a real
`proof_recovery_extraction_priority` event and at least one positive causal
proof return before Smoke-A20 is opened.

## Budget

Usage before v2.4.3.8 is 1,069 provider attempts and 1,230,836 provider-reported
tokens. Remaining global budget is 931 attempts and 769,164 tokens under the
unchanged 2,000 / 2,000,000 hard cap.
