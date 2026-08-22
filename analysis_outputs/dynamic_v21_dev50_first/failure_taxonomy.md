# Dynamic Hypergraph TDCA v2 pre-change failure taxonomy

- Source run: `research_outputs/musique_distractor_development_dynamic_hypergraph_tdca_v2_1787422603606942232`
- Frozen baseline commit: `97198bef951f88726b213bf199ca53ce8226d13c`
- Cases: 50
- Method: deterministic rules over frozen official metrics and reasoning/retrieval traces.
- Caution: query/EVC labels are trace-grounded hypotheses, not claims of unique causality.

## Counts

| Label | Main cause | Any label |
|---|---:|---:|
| infrastructure_failure | 0 | 0 |
| provider_refusal | 0 | 0 |
| structured_output_failure | 1 | 1 |
| retrieval_miss | 11 | 11 |
| query_formulation_or_missing_binding_miss | 0 | 0 |
| claim_extraction_miss | 19 | 22 |
| type_or_binding_mismatch | 3 | 17 |
| join_verification_rejection | 2 | 13 |
| join_expressivity_failure | 0 | 4 |
| candidate_commit_or_survival_failure | 0 | 0 |
| final_synthesis_failure | 1 | 4 |
| premature_stop | 0 | 11 |
| evc_misallocation | 0 | 2 |
| budget_exhaustion | 0 | 2 |
| correct_or_no_observed_failure | 13 | 13 |

## Per-example attribution

| QID | Hop | Status | F1 | Main cause | Secondary causes |
|---|---:|---|---:|---|---|
| `2hop__107238_64918` | 2 | answer | 0.000 | claim_extraction_miss | - |
| `2hop__131818_161450` | 2 | abstain | 0.000 | structured_output_failure | claim_extraction_miss, type_or_binding_mismatch |
| `2hop__132472_684936` | 2 | answer | 0.000 | claim_extraction_miss | - |
| `2hop__152229_604644` | 2 | abstain | 0.000 | claim_extraction_miss | premature_stop |
| `2hop__196348_150107` | 2 | abstain | 0.000 | retrieval_miss | claim_extraction_miss, premature_stop |
| `2hop__217011_80026` | 2 | infrastructure_failure | 0.000 | claim_extraction_miss | join_verification_rejection |
| `2hop__250030_7298` | 2 | budget_exhausted | 0.000 | type_or_binding_mismatch | final_synthesis_failure, evc_misallocation, budget_exhaustion |
| `2hop__263728_705527` | 2 | answer | 0.000 | claim_extraction_miss | - |
| `2hop__279729_20057` | 2 | answer | 0.000 | retrieval_miss | type_or_binding_mismatch |
| `2hop__29339_40482` | 2 | abstain | 0.000 | retrieval_miss | - |
| `2hop__29349_92763` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__326404_120065` | 2 | answer | 0.000 | final_synthesis_failure | - |
| `2hop__36269_73244` | 2 | answer | 0.000 | claim_extraction_miss | - |
| `2hop__3739_13529` | 2 | answer | 1.000 | join_verification_rejection | - |
| `2hop__388467_110882` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__476927_31270` | 2 | abstain | 0.000 | claim_extraction_miss | join_verification_rejection, premature_stop |
| `2hop__507528_139339` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__557284_160249` | 2 | abstain | 0.000 | claim_extraction_miss | join_verification_rejection, premature_stop |
| `2hop__563476_61845` | 2 | abstain | 0.000 | retrieval_miss | type_or_binding_mismatch |
| `2hop__58168_1783` | 2 | answer | 0.000 | retrieval_miss | type_or_binding_mismatch, join_verification_rejection |
| `2hop__608902_653666` | 2 | answer | 1.000 | type_or_binding_mismatch | - |
| `2hop__66167_72467` | 2 | abstain | 0.000 | claim_extraction_miss | join_verification_rejection, premature_stop |
| `2hop__744055_408817` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__752214_639679` | 2 | answer | 0.000 | retrieval_miss | join_verification_rejection |
| `2hop__809096_491515` | 2 | abstain | 0.000 | claim_extraction_miss | premature_stop |
| `2hop__816977_6455` | 2 | abstain | 0.000 | claim_extraction_miss | join_verification_rejection, premature_stop |
| `2hop__827755_156034` | 2 | answer | 0.000 | claim_extraction_miss | - |
| `2hop__84172_198548` | 2 | abstain | 0.000 | join_verification_rejection | final_synthesis_failure, premature_stop |
| `2hop__92385_2072` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__92590_43786` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__95172_152907` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `2hop__95687_2684` | 2 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__109422_720914_41132` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__161433_33952_34099` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__251085_831637_91775` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__491648_339990_15538` | 3 | abstain | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_expressivity_failure, premature_stop |
| `3hop1__559908_42197_18397` | 3 | answer | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_verification_rejection |
| `3hop1__584774_131926_87157` | 3 | answer | 0.222 | claim_extraction_miss | - |
| `3hop1__765847_831637_91775` | 3 | answer | 1.000 | retrieval_miss | type_or_binding_mismatch |
| `3hop1__786067_228453_10972` | 3 | answer | 1.000 | type_or_binding_mismatch | join_verification_rejection |
| `3hop1__79462_91850_685675` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__857_846_7795` | 3 | answer | 1.000 | correct_or_no_observed_failure | - |
| `3hop1__865037_214799_259594` | 3 | answer | 1.000 | retrieval_miss | type_or_binding_mismatch |
| `3hop2__106842_30645_84681` | 3 | abstain | 0.000 | claim_extraction_miss | type_or_binding_mismatch, join_verification_rejection, join_expressivity_failure, premature_stop |
| `3hop2__668732_223623_162182` | 3 | answer | 0.000 | retrieval_miss | type_or_binding_mismatch, final_synthesis_failure |
| `3hop2__89048_860687_66294` | 3 | abstain | 0.000 | claim_extraction_miss | join_expressivity_failure, premature_stop |
| `4hop1__638988_17130_70784_79935` | 4 | abstain | 0.000 | retrieval_miss | type_or_binding_mismatch, join_expressivity_failure |
| `4hop1__860115_798482_131926_89261` | 4 | budget_exhausted | 0.000 | retrieval_miss | claim_extraction_miss, type_or_binding_mismatch, join_verification_rejection, evc_misallocation, budget_exhaustion |
| `4hop3__130276_29339_508306_70744` | 4 | answer | 0.667 | claim_extraction_miss | type_or_binding_mismatch |
| `4hop3__324979_836463_161616_77103` | 4 | answer | 0.000 | claim_extraction_miss | type_or_binding_mismatch |

The companion JSON contains every inference rule and its supporting trace fields.
