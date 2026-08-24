# Dynamic Hypergraph TDCA v2.3 offline diagnostic

- Source: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787593956846465968`
- Samples: 20
- Provider/LLM calls made by this audit: 0
- Gold boundary: post-hoc terminal attribution only

## Allocation calibration

| Slice | Count | Spearman(EVC, utility) | Mean utility | Progress | No-op |
|---|---:|---:|---:|---:|---:|
| overall | 324 | 0.2845 | 0.0009 | 0.8457 | 0.1543 |
| branch:assignments | 2 | n/a | 0.0000 | 1.0000 | 0.0000 |
| branch:extract_typed | 81 | 0.4387 | -0.0713 | 0.6667 | 0.3333 |
| commit:answer | 11 | -0.1284 | 0.0105 | 1.0000 | 0.0000 |
| commit:default | 36 | -0.2049 | 0.0889 | 1.0000 | 0.0000 |
| expand:default | 9 | 0.0693 | -0.0998 | 0.3333 | 0.6667 |
| merge:validate_join | 32 | 0.0257 | 0.0300 | 0.6250 | 0.3750 |
| retrieve:default | 77 | 0.3215 | 0.0963 | 0.9740 | 0.0260 |
| verify:default | 76 | 0.4765 | -0.0621 | 0.9605 | 0.0395 |

## Ready-set choice audit

- Decisions with candidates: 327
- Real operation-choice rate: 0.2997
- Cross-family choice rate: 0.2630
- Cross-region choice rate: 0.2997
- Fidelity-only choice rate: 0.3639
- Selected families: `{"branch:assignments": 2, "branch:extract_typed": 81, "commit:answer": 11, "commit:default": 36, "expand:default": 9, "merge:validate_join": 32, "retrieve:default": 77, "verify:default": 76}`
- Selected fidelities: `{"high": 142, "low": 132, "medium": 50}`

## Retrieval marginal utility

- Retrievals: 77
- Mean new unique passages: 3.3377
- Zero-unique-yield rate: 0.1169
- No final-claim-yield rate: 0.3117
- Accepted-answer evidence-use rate: 0.3506

| Retrieval round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 7.0000 | 3.0000 | 0.5500 | 0.0500 | 0.1674 |
| 2 | 20 | 3.5000 | 2.8000 | 0.4500 | 0.1500 | 0.1683 |
| 3 | 19 | 1.3158 | 1.4737 | 0.1579 | 0.6316 | 0.0037 |
| 4 | 10 | 1.6000 | 3.4000 | 0.4000 | 0.2000 | 0.1040 |
| 5 | 8 | 0.7500 | 0.3750 | 0.0000 | 0.7500 | -0.0513 |

### Within-subgoal retrieval rounds

| Subgoal round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 4.5600 | 3.3800 | 0.5400 | 0.0800 | 0.1754 |
| 2 | 27 | 1.0741 | 0.4444 | 0.0000 | 0.7407 | -0.0502 |

## Non-answer bottlenecks

| Bottleneck | Count | Rate |
|---|---:|---:|
| budget_exhaustion | 4 | 0.4444 |
| retrieval_access | 4 | 0.4444 |
| context_to_candidate_extraction | 1 | 0.1111 |

## Extraction diagnostics

- Trace coverage: all_attempts
- Recorded attempts: 81
- Accepted-attempt rate: 0.6667
- Empty model-output rate: 0.2099
- Budget-compacted context rate: 0.0494
- Rejection reasons: `{"duplicate_triple": 14, "ungrounded": 9}`

## JOIN frontier audit

- Attempts / accepted / charged: 32 / 20 / 21
- Acceptance rate: 0.6250
- Accepted-answer use rate: 0.4688
- JOIN model calls / tokens: 1.0000 / 899.0000
- Rejection reasons: `{"operation_produced_no_commit": 12}`

## Interpretation guardrails

- Correlation is observational over selected actions, not a counterfactual policy-value estimate.
- Retrieval-to-final-claim attribution uses final graph provenance and does not imply sole causality.
- Gold-aware bottlenecks are evaluation-only and must never become inference features.
- A real allocation claim requires distinct executable operations/regions, not only token-fidelity variation.
