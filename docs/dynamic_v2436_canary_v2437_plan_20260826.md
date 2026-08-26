# TDCA v2.4.3.6 canary and v2.4.3.7 source freeze

## v2.4.3.6 recovery canary

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787745200792538494`
- Diagnostic selection: the eight v2.4.3.5 Smoke-A examples that executed a
  gold-free proof-gap recovery; this is not a held-out result.
- Complete: 8/8 with zero infrastructure and safety failures.
- Provider usage: 66 attempts and 81,150 provider-reported tokens.
- Relative to v2.4.3.5 on the same examples: six chains remained absent and two
  complete chains were gained; no chain was lost and both emitted answers were
  correct.
- Immediate / delayed / choice-conditioned delayed Spearman: 0.682 / 0.213 /
  0.104.
- Three recovery allocations had explicit target obligations, compared with zero
  auditable targets in v2.4.3.5.
- None of those three allocations received a positive causal proof return.

## Remaining causal break

The recovered passages were attached to the correct graph region, but proposed
claims already present in that region were verified before the new evidence was
extracted. Retrieval therefore improved local evidence counts without reliably
creating a claim, JOIN, or supported answer in the retrieval allocation's causal
descendant graph.

## v2.4.3.7 semantic boundary

v2.4.3.7 adds one opt-in, deterministic rule:

1. Persist the recovery policy and nonempty target-obligation IDs in the
   controller-owned retrieval-attempt ledger.
2. When a recovery attempt yields new evidence not represented by the latest
   extraction fingerprint, schedule typed extraction before re-verifying stale
   proposed claims in that same region.
3. Apply the rule only to recovery attempts with explicit obligation targets and
   positive new-evidence count.
4. Mark the extraction fingerprint as attempted exactly as in the existing
   bounded extraction policy, so the rule cannot loop.

No score, threshold, terminal gate, JOIN gate, answer type, or question-specific
logic changes. The next stage repeats the same eight-example diagnostic canary and
requires at least one positive causal proof-gap return before Smoke-A20 can open.

## Budget

Usage before v2.4.3.7 is 1,000 provider attempts and 1,146,718 provider-reported
tokens. Remaining global budget is 1,000 attempts and 853,282 tokens under the
unchanged 2,000 / 2,000,000 hard cap.
