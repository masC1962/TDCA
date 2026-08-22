# Dynamic Hypergraph TDCA v2 pre-change failure taxonomy

- Source run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787421433668233978`
- Frozen baseline commit: `97198bef951f88726b213bf199ca53ce8226d13c`
- Cases: 20
- Method: deterministic rules over frozen official metrics and reasoning/retrieval traces.
- Caution: query/EVC labels are trace-grounded hypotheses, not claims of unique causality.

## Counts

| Label | Main cause | Any label |
|---|---:|---:|
| infrastructure_failure | 0 | 0 |
| provider_refusal | 0 | 0 |
| structured_output_failure | 0 | 0 |
| retrieval_miss | 4 | 4 |
| query_formulation_or_missing_binding_miss | 0 | 0 |
| claim_extraction_miss | 6 | 7 |
| type_or_binding_mismatch | 1 | 8 |
| join_verification_rejection | 4 | 6 |
| join_expressivity_failure | 0 | 5 |
| candidate_commit_or_survival_failure | 0 | 0 |
| final_synthesis_failure | 0 | 4 |
| premature_stop | 0 | 3 |
| evc_misallocation | 0 | 5 |
| budget_exhaustion | 0 | 5 |
| correct_or_no_observed_failure | 5 | 5 |

## Per-example attribution

| QID | Hop | Status | F1 | Main cause | Secondary causes |
|---|---:|---|---:|---|---|
| `2hop__16844_20510` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__424908_500483` | 2 | budget_exhausted | 0.000 | join_verification_rejection | final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `2hop__511296_2684` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__62951_64006` | 2 | budget_exhausted | 0.000 | join_verification_rejection | final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `2hop__84937_21969` | 2 | budget_exhausted | 0.000 | join_verification_rejection | final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `2hop__89764_827343` | 2 | answer | 1.000 | join_verification_rejection | - |
| `3hop1__105767_443779_52195` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__132795_40769_64047` | 3 | abstain | 0.000 | claim_extraction_miss | join_expressivity_failure, premature_stop |
| `3hop1__140786_2053_5289` | 3 | budget_exhausted | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_expressivity_failure, evc_misallocation, budget_exhaustion |
| `3hop1__145924_131905_41948` | 3 | answer | 0.000 | claim_extraction_miss | - |
| `3hop1__337705_132457_51423` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__498954_160713_77246` | 3 | abstain | 0.000 | type_or_binding_mismatch | final_synthesis_failure, premature_stop |
| `3hop1__801799_547811_41132` | 3 | answer | 1.000 | retrieval_miss | type_or_binding_mismatch, join_expressivity_failure |
| `3hop2__326964_7845_7713` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop2__90327_87184_76291` | 3 | budget_exhausted | 0.000 | claim_extraction_miss | join_verification_rejection, evc_misallocation, budget_exhaustion |
| `4hop1__107309_457883_650651_7262` | 4 | answer | 1.000 | retrieval_miss | type_or_binding_mismatch |
| `4hop1__152146_5274_458768_33637` | 4 | answer | 0.000 | claim_extraction_miss | type_or_binding_mismatch |
| `4hop1__51465_53706_795904_580996` | 4 | abstain | 0.000 | retrieval_miss | type_or_binding_mismatch, join_expressivity_failure |
| `4hop2__103790_39078_8987_8529` | 4 | abstain | 0.000 | retrieval_miss | claim_extraction_miss, type_or_binding_mismatch, join_verification_rejection, join_expressivity_failure, premature_stop |
| `4hop2__161602_474028_88460_21057` | 4 | answer | 0.400 | claim_extraction_miss | type_or_binding_mismatch |

The companion JSON contains every inference rule and its supporting trace fields.
