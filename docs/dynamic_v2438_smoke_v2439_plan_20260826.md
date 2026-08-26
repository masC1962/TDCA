# TDCA v2.4.3.8 Smoke-A diagnosis and v2.4.3.9 source freeze

## v2.4.3.8 Smoke-A

- Run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787748087015307095`.
- Complete: 20/20 with zero infrastructure, graph-invariant, controller-mutation,
  unsupported-answer, repeated-extraction, or selected-infeasible-JOIN failures.
- Provider usage: 152 attempts and 169,466 provider-reported tokens. Logical
  accounting is 183 calls and 209,496 tokens because 31 requests reused the
  isolated v2.4.3.8 canary cache; this arm is therefore not an independent API
  replicate.
- Exact match / F1: 0.40 / 0.445.
- Candidate / legacy execution chain / graph proof: 0.75 / 0.60 / 0.90.
- Immediate / delayed / choice-conditioned delayed Spearman: 0.662 / 0.128 /
  0.033.
- The hard gate fails legacy chain completion, F1, logical-call and token bounds,
  and delayed calibration. Shadow-B remains closed.

The controller-derived recovery mechanism remained causally active: five
targeted cases produced six freshness events, two positive causal returns, and
two end-to-end recoveries. Relative to v2.4.3.5, the arm gained four candidate
cases and four execution chains and lost none. The remaining error is therefore
downstream state use, not retrieval provenance.

## Gold-free structural diagnosis

Two general implementation gaps account for recoverable failures:

1. Quantitative answer schemas used mutually compatible labels such as
   `numerical`, `fraction`, and `percentage`, while structural projection treated
   them as unrelated. The offline audit found two independently supported
   projections in one question blocked only by this schema mismatch.
2. Failed JOINs were keyed by diffusion belief versions. A premise could later
   become verified and feasible without receiving the precise version transition
   needed to reopen the JOIN. Final dead-end graphs contained four feasible,
   goal-relevant JOIN candidates across two questions.

The diagnosis did not inspect gold answers and does not alter any score or gate.
A proposed feasibility-ranking change recovered zero additional JOIN regions in
offline replay and was rejected before provider use.

## v2.4.3.9 semantic boundary

v2.4.3.9 adds two opt-in deterministic corrections:

- normalize only quantitative output-type aliases to the canonical `number`
  family; nonnumeric types remain incompatible;
- bind a JOIN attempt to the premise status, independent score channels,
  evidence gap, contradiction risk, and evidence references that determine
  feasibility, rather than to diffusion bookkeeping versions.

Thresholds, terminal rules, JOIN feasibility rules, JOIN budget, prompts beyond
the architecture cache namespace, and all per-question inference behavior remain
unchanged. The policy is training-free and contains no answer-specific patch.

The first provider stage is a frozen three-example diagnostic canary selected
only from the structural audit. Smoke-A20 opens only if the intended state
transitions occur without candidate/chain regression. Shadow-B remains gated by
the unchanged Smoke-A hard gate.

## Budget

Usage before v2.4.3.9 is 1,252 provider attempts and 1,440,332
provider-reported tokens. The remaining global allowance is 748 attempts and
559,668 tokens under the unchanged 2,000 / 2,000,000 hard cap.
