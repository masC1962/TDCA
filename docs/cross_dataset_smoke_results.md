# Frozen cross-dataset smoke-20 results

These runs use the pre-registered seed-520, disjoint smoke splits and the unchanged
MuSiQue-frozen thresholds/budgets. They are reliability checks and low-power transfer
diagnostics, not final quality estimates or a basis for parameter selection.

| Dataset | Method | EM | F1 | Infrastructure failures |
|---|---|---:|---:|---:|
| HotpotQA | Structured-TDCA + Dense | 0.450 | 0.545 | 0/20 |
| HotpotQA | Controlled IRCoT + BM25 | 0.450 | **0.668** | 0/20 |
| 2WikiMultiHopQA | Structured-TDCA + Dense | 0.650 | 0.683 | 0/20 |
| 2WikiMultiHopQA | Controlled IRCoT + BM25 | **0.700** | **0.812** | 0/20 |

On HotpotQA, Structured-minus-IRCoT is 0.000 EM (95% paired bootstrap CI
[-0.150, 0.150]) and -0.124 F1 (CI [-0.267, -0.006]). On 2Wiki it is -0.050 EM
(CI [-0.350, 0.250]) and -0.129 F1 (CI [-0.378, 0.107]). These 20-row intervals
should not be overinterpreted, but neither dataset provides evidence of a Structured
quality advantage.

All four current-schema artifacts passed checksum, ordered-ID, metric-count and
completion audits. The resulting `research_outputs/cross_dataset_smoke_gate.json`
opens the pre-registered, disjoint tuning-50 reliability gate; it does not authorize
configuration changes.
