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

## v2.4.3.8 targeted canary result

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787747346189219194`.
- Complete: 3/3; 31 provider attempts and 40,030 provider-reported tokens.
- All three targeted retrievals had controller-derived provenance and all three
  fired the matching freshness-priority extraction event.
- One retrieval received positive structural delayed credit (`0.17`) through a
  newly accepted evidence descendant, giving one auditable end-to-end recovery.
- One previously absent three-hop chain was gained through fresh extraction and
  an accepted n-ary JOIN; the emitted answer was correct. There were no
  unsupported answers or infrastructure failures.

The preregistered canary opening condition is satisfied. These three examples
remain diagnostic-only; the unchanged policy must next pass the full frozen
Smoke-A20 hard gate before Shadow-B can be authorized.

## Budget

Usage after the v2.4.3.8 canary is 1,100 provider attempts and 1,270,866
provider-reported tokens. Remaining global budget is 900 attempts and 729,134
tokens under the unchanged 2,000 / 2,000,000 hard cap.
