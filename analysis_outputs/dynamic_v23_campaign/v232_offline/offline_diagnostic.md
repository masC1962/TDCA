# Dynamic Hypergraph TDCA v2.3 offline diagnostic

- Source: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787591050299957601`
- Samples: 20
- Provider/LLM calls made by this audit: 0
- Gold boundary: post-hoc terminal attribution only

## Allocation calibration

| Slice | Count | Spearman(EVC, utility) | Mean utility | Progress | No-op |
|---|---:|---:|---:|---:|---:|
| overall | 278 | 0.2109 | 0.0144 | 0.8237 | 0.1763 |
| branch:assignments | 2 | n/a | 0.0000 | 1.0000 | 0.0000 |
| branch:extract_typed | 59 | 0.1493 | -0.0597 | 0.8136 | 0.1864 |
| commit:answer | 13 | -0.0717 | 0.0128 | 1.0000 | 0.0000 |
| commit:default | 36 | -0.1320 | 0.0988 | 1.0000 | 0.0000 |
| expand:default | 3 | n/a | -0.1143 | 0.0000 | 1.0000 |
| merge:validate_join | 51 | 0.2756 | -0.0301 | 0.3922 | 0.6078 |
| retrieve:default | 60 | -0.1142 | 0.1332 | 0.9833 | 0.0167 |
| verify:default | 54 | 0.1578 | -0.0430 | 0.9444 | 0.0556 |

## Ready-set choice audit

- Decisions with candidates: 280
- Real operation-choice rate: 0.1536
- Cross-family choice rate: 0.1036
- Cross-region choice rate: 0.1536
- Fidelity-only choice rate: 0.3464
- Selected families: `{"branch:assignments": 2, "branch:extract_typed": 59, "commit:answer": 13, "commit:default": 36, "expand:default": 3, "merge:validate_join": 51, "retrieve:default": 60, "verify:default": 54}`
- Selected fidelities: `{"high": 122, "low": 105, "medium": 51}`

## Retrieval marginal utility

- Retrievals: 60
- Mean new unique passages: 4.0167
- Zero-unique-yield rate: 0.1167
- No final-claim-yield rate: 0.2000
- Accepted-answer evidence-use rate: 0.5000

| Retrieval round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 7.1500 | 2.9500 | 0.6500 | 0.0500 | 0.1694 |
| 2 | 20 | 3.4000 | 2.5500 | 0.4500 | 0.2000 | 0.1463 |
| 3 | 15 | 1.5333 | 3.1333 | 0.4000 | 0.4000 | 0.0768 |
| 4 | 4 | 1.5000 | 4.2500 | 0.5000 | 0.0000 | 0.1444 |
| 5 | 1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0497 |

### Within-subgoal retrieval rounds

| Subgoal round | Count | New passages | Final supported claims | Answer use | No claim yield | Utility |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 4.7500 | 3.5208 | 0.6250 | 0.0625 | 0.1801 |
| 2 | 12 | 1.0833 | 0.4167 | 0.0000 | 0.7500 | -0.0543 |

## Non-answer bottlenecks

| Bottleneck | Count | Rate |
|---|---:|---:|
| budget_exhaustion | 4 | 0.5714 |
| retrieval_access | 2 | 0.2857 |
| context_to_candidate_extraction | 1 | 0.1429 |

## Extraction diagnostics

- Trace coverage: all_attempts
- Recorded attempts: 58
- Accepted-attempt rate: 0.8276
- Empty model-output rate: 0.1207
- Budget-compacted context rate: 0.0000
- Rejection reasons: `{"duplicate_triple": 3, "ungrounded": 8}`

## JOIN frontier audit

- Attempts / accepted / charged: 51 / 20 / 43
- Acceptance rate: 0.3922
- Accepted-answer use rate: 0.3137
- JOIN model calls / tokens: 23.0000 / 25714.0000
- Rejection reasons: `{"StructuredOutputError": 1, "operation_produced_no_commit": 30}`

## Interpretation guardrails

- Correlation is observational over selected actions, not a counterfactual policy-value estimate.
- Retrieval-to-final-claim attribution uses final graph provenance and does not imply sole causality.
- Gold-aware bottlenecks are evaluation-only and must never become inference features.
- A real allocation claim requires distinct executable operations/regions, not only token-fidelity variation.
