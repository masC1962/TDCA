# TDCA v2.4.3.5 result and v2.4.3.6 source freeze

## v2.4.3.5 Smoke-A result

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787736117754451385`
- Complete: 20/20, zero infrastructure failures.
- Provider usage: 123 attempts and 151,417 provider-reported tokens.
- Candidate presence / execution-plan completion / graph-proof completion: 0.55 / 0.55 / 0.85.
- EM / F1: 0.35 / 0.395.
- Immediate / delayed / choice-conditioned delayed Spearman: 0.599 / 0.168 / -0.079.
- Certified transition realization: 50/50; no invalid bypass and no blocked accepted terminal readout.
- Proof-gap recovery: 0/8 successful.

The transition-certificate binding and sealed ancestor-lineage fixes are correct,
but the feedback-multiplicative delayed value and compact query trial are not
quality-safe. They lost two chains relative to v2.4.3.4 and are not carried into
the next policy.

## Root cause

All eight allocations labelled as proof-gap recovery had an empty
`target_obligation_ids` list and predicted delayed proof return equal to zero.
The semantic proof-usability diagnostic and the controller-owned proof-obligation
ledger represented different state spaces. Consequently, recovery received an
immediate heuristic score but could neither make an auditable closure promise nor
receive causal delayed credit. Several retrievals produced new evidence, but the
subsequent policy either verified stale claims or stopped before extracting the
new evidence.

## v2.4.3.6 semantic boundary

v2.4.3.6 is training-free and makes no threshold, terminal-gate, JOIN-gate, or
per-question change.

1. Keep concrete transition-certificate binding and sealed ancestor claim lineage.
2. Revert the failed v2.4.3.5 delayed-value and compact-query trials.
3. Project an unusable target proof into a controller-owned
   `insufficient_target_proof` obligation.
4. Permit RETRIEVE, EXTRACT, VERIFY, and JOIN to claim delayed value only when the
   concrete operation can target that obligation.
5. Build recovery queries from dependency values and query-graph known entities;
   if no auditable anchor exists, do not issue a generic meta-instruction query.
6. Count a proof-gap recovery in the gate only when it has an explicit target
   obligation.

The first provider-backed stage is an eight-example, non-heldout diagnostic
canary selected only by the gold-free condition “v2.4.3.5 executed a proof-gap
recovery.” A full frozen Smoke-A20 may run only if the canary preserves safety,
creates nonempty recovery targets, and does not repeat the v2.4.3.5 chain loss.

## Budget

Usage before v2.4.3.6 is 934 provider attempts and 1,065,568 provider-reported
tokens. Remaining global budget is 1,066 attempts and 934,432 tokens under the
unchanged 2,000 / 2,000,000 hard cap.
