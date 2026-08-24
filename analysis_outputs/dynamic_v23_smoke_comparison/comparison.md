# Dynamic Hypergraph TDCA v2.3 smoke Go/No-Go

- Decision: **NO_GO_FIX_BEFORE_CONTROLS**
- Selected candidate: `v2.3.3`
- Provider/LLM calls made by this comparison: 0
- Failed checks: `['full_chain_non_regression', 'bounded_llm_call_growth']`

| Run | EM | F1 | Candidate | Full chain | EVC↔utility | Calls | Tokens | Retrieval | Budget exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2.2 | 0.300 | 0.355 | 0.500 | 0.650 | -0.0553 | 159 | 179035 | 63 | 0.100 |
| v2.3.0 | 0.300 | 0.409 | 0.500 | 0.600 | 0.1137 | 155 | 179537 | 60 | 0.150 |
| v2.3.1 | 0.300 | 0.384 | 0.450 | 0.550 | 0.1260 | 166 | 176907 | 65 | 0.150 |
| v2.3.2 | 0.400 | 0.484 | 0.700 | 0.600 | 0.2109 | 160 | 177655 | 60 | 0.200 |
| v2.3.3 | 0.400 | 0.434 | 0.700 | 0.550 | 0.2845 | 188 | 207940 | 77 | 0.200 |

## Gate checks

- [x] artifact_complete
- [x] zero_infrastructure_failure
- [x] zero_unsupported_answer
- [x] candidate_presence_plus_0_10
- [ ] full_chain_non_regression
- [x] f1_non_regression
- [x] positive_evc_calibration
- [ ] bounded_llm_call_growth
- [x] bounded_graph_growth

## Selected candidate paired transitions

- Chain gained/lost: 3/5
- Candidate gained/lost: 5/1
- Exact match gained/lost: 4/2
- Chain-lost qids: `['2hop__89764_827343', '3hop1__105767_443779_52195', '3hop1__132795_40769_64047', '3hop2__90327_87184_76291', '4hop1__51465_53706_795904_580996']`
- Budget-exhausted qids: `['2hop__62951_64006', '3hop1__105767_443779_52195', '3hop2__90327_87184_76291', '4hop1__152146_5274_458768_33637']`
- `2hop__89764_827343`: status answer -> abstain; calls 9 -> 9; JOIN attempts/accepted/charged 1/1/1 -> 7/0/0
- `3hop1__105767_443779_52195`: status answer -> budget_exhausted; calls 7 -> 8; JOIN attempts/accepted/charged 2/2/2 -> 1/1/1
- `3hop1__132795_40769_64047`: status answer -> abstain; calls 8 -> 8; JOIN attempts/accepted/charged 2/2/2 -> 0/0/0
- `3hop2__90327_87184_76291`: status answer -> budget_exhausted; calls 11 -> 15; JOIN attempts/accepted/charged 6/6/6 -> 4/3/4
- `4hop1__51465_53706_795904_580996`: status answer -> abstain; calls 7 -> 4; JOIN attempts/accepted/charged 2/2/2 -> 0/0/0

The smoke split is development-only and too small for a paper claim. A failed check blocks matched controls and larger runs; it does not invalidate the mechanism-level improvements recorded above.
