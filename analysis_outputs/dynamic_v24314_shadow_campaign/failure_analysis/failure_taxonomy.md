# Dynamic Hypergraph TDCA v2 pre-change failure taxonomy

- Source run: `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787817590041791173`
- Frozen baseline commit: `a6c53189`
- Cases: 20
- Method: deterministic rules over frozen official metrics and reasoning/retrieval traces.
- Caution: query/EVC labels are trace-grounded hypotheses, not claims of unique causality.

## Counts

| Label | Main cause | Any label |
|---|---:|---:|
| infrastructure_failure | 0 | 0 |
| provider_refusal | 0 | 0 |
| structured_output_failure | 0 | 0 |
| retrieval_miss | 3 | 3 |
| query_formulation_or_missing_binding_miss | 0 | 0 |
| claim_extraction_miss | 3 | 5 |
| type_or_binding_mismatch | 4 | 8 |
| join_verification_rejection | 0 | 0 |
| join_expressivity_failure | 0 | 7 |
| candidate_commit_or_survival_failure | 0 | 0 |
| final_synthesis_failure | 5 | 7 |
| premature_stop | 0 | 5 |
| evc_misallocation | 0 | 3 |
| budget_exhaustion | 0 | 3 |
| correct_or_no_observed_failure | 5 | 5 |

## Per-example attribution

| QID | Hop | Status | F1 | Main cause | Secondary causes |
|---|---:|---|---:|---|---|
| `2hop__142130_67668` | 2 | abstain | 0.000 | final_synthesis_failure | premature_stop |
| `2hop__15650_77173` | 2 | answer | 0.400 | final_synthesis_failure | - |
| `2hop__25797_25855` | 2 | abstain | 0.000 | final_synthesis_failure | premature_stop |
| `2hop__711946_269414` | 2 | answer | 0.000 | final_synthesis_failure | - |
| `2hop__829358_754719` | 2 | answer | 0.000 | final_synthesis_failure | - |
| `2hop__88906_55840` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__228_237521_291682` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__465684_160545_34751` | 3 | abstain | 0.000 | retrieval_miss | claim_extraction_miss, join_expressivity_failure, premature_stop |
| `3hop1__504362_443779_52195` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__773623_87694_124169` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__816887_127905_80286` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__838590_831637_91775` | 3 | answer | 1.000 | type_or_binding_mismatch | - |
| `3hop1__857_846_7888` | 3 | budget_exhausted | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_expressivity_failure, evc_misallocation, budget_exhaustion |
| `3hop2__127483_19639_10557` | 3 | budget_exhausted | 0.000 | type_or_binding_mismatch | join_expressivity_failure, final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `3hop2__89048_228_66294` | 3 | abstain | 0.000 | retrieval_miss | join_expressivity_failure |
| `4hop1__236903_153080_159767_81096` | 4 | answer | 0.000 | claim_extraction_miss | type_or_binding_mismatch |
| `4hop1__711307_49925_13759_736921` | 4 | abstain | 0.000 | type_or_binding_mismatch | join_expressivity_failure, final_synthesis_failure, premature_stop |
| `4hop2__161602_474028_88460_63559` | 4 | budget_exhausted | 0.000 | retrieval_miss | claim_extraction_miss, type_or_binding_mismatch, join_expressivity_failure, evc_misallocation, budget_exhaustion |
| `4hop3__275416_24325_125104_10557` | 4 | answer | 1.000 | type_or_binding_mismatch | - |
| `4hop3__316459_41402_146281_13584` | 4 | abstain | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_expressivity_failure, premature_stop |

The companion JSON contains every inference rule and its supporting trace fields.
