# Dynamic Hypergraph TDCA v2 pre-change failure taxonomy

- Source run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787402511722832607`
- Frozen baseline commit: `e4dac8c7621b0d0e28534954260eec23feea1012`
- Cases: 20
- Method: deterministic rules over frozen official metrics and reasoning/retrieval traces.
- Caution: query/EVC labels are trace-grounded hypotheses, not claims of unique causality.

## Counts

| Label | Main cause | Any label |
|---|---:|---:|
| provider_or_infrastructure_failure | 9 | 9 |
| retrieval_miss | 1 | 3 |
| query_formulation_or_missing_binding_miss | 0 | 0 |
| claim_extraction_miss | 7 | 13 |
| type_or_binding_mismatch | 0 | 7 |
| join_verification_rejection | 0 | 7 |
| join_expressivity_failure | 0 | 11 |
| candidate_commit_or_survival_failure | 0 | 0 |
| final_synthesis_failure | 2 | 4 |
| premature_stop | 0 | 4 |
| evc_misallocation | 0 | 6 |
| budget_exhaustion | 0 | 6 |
| correct_or_no_observed_failure | 1 | 1 |

## Per-example attribution

| QID | Hop | Status | F1 | Main cause | Secondary causes |
|---|---:|---|---:|---|---|
| `2hop__16844_20510` | 2 | answer | 0.000 | final_synthesis_failure | - |
| `2hop__424908_500483` | 2 | budget_exhausted | 0.000 | final_synthesis_failure | evc_misallocation, budget_exhaustion |
| `2hop__511296_2684` | 2 | abstain | 0.000 | claim_extraction_miss | premature_stop |
| `2hop__62951_64006` | 2 | abstain | 0.000 | provider_or_infrastructure_failure | final_synthesis_failure |
| `2hop__84937_21969` | 2 | budget_exhausted | 0.000 | provider_or_infrastructure_failure | claim_extraction_miss, join_verification_rejection, evc_misallocation, budget_exhaustion |
| `2hop__89764_827343` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__105767_443779_52195` | 3 | infrastructure_failure | 0.000 | provider_or_infrastructure_failure | claim_extraction_miss, join_expressivity_failure |
| `3hop1__132795_40769_64047` | 3 | answer | 0.500 | claim_extraction_miss | - |
| `3hop1__140786_2053_5289` | 3 | abstain | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_expressivity_failure, premature_stop |
| `3hop1__145924_131905_41948` | 3 | budget_exhausted | 0.000 | provider_or_infrastructure_failure | claim_extraction_miss, join_verification_rejection, join_expressivity_failure, evc_misallocation, budget_exhaustion |
| `3hop1__337705_132457_51423` | 3 | abstain | 0.000 | claim_extraction_miss | join_verification_rejection, join_expressivity_failure, premature_stop |
| `3hop1__498954_160713_77246` | 3 | abstain | 0.000 | provider_or_infrastructure_failure | claim_extraction_miss, type_or_binding_mismatch, join_verification_rejection, join_expressivity_failure |
| `3hop1__801799_547811_41132` | 3 | abstain | 0.000 | retrieval_miss | join_verification_rejection, join_expressivity_failure |
| `3hop2__326964_7845_7713` | 3 | abstain | 0.000 | claim_extraction_miss | join_expressivity_failure, premature_stop |
| `3hop2__90327_87184_76291` | 3 | abstain | 0.000 | provider_or_infrastructure_failure | claim_extraction_miss, join_expressivity_failure |
| `4hop1__107309_457883_650651_7262` | 4 | budget_exhausted | 0.000 | provider_or_infrastructure_failure | type_or_binding_mismatch, join_verification_rejection, join_expressivity_failure, final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `4hop1__152146_5274_458768_33637` | 4 | budget_exhausted | 0.000 | provider_or_infrastructure_failure | retrieval_miss, claim_extraction_miss, type_or_binding_mismatch, evc_misallocation, budget_exhaustion |
| `4hop1__51465_53706_795904_580996` | 4 | infrastructure_failure | 0.000 | provider_or_infrastructure_failure | retrieval_miss, type_or_binding_mismatch, join_expressivity_failure |
| `4hop2__103790_39078_8987_8529` | 4 | budget_exhausted | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_verification_rejection, join_expressivity_failure, evc_misallocation, budget_exhaustion |
| `4hop2__161602_474028_88460_21057` | 4 | answer | 0.400 | claim_extraction_miss | type_or_binding_mismatch |

The companion JSON contains every inference rule and its supporting trace fields.
