# Frozen cross-dataset tuning-50 results

These results use the pre-registered seed-520 splits disjoint from smoke-20, with the
MuSiQue-frozen algorithm, thresholds and budgets. All four current-schema artifacts
passed checksum, ordered-ID, completion and count audits with zero infrastructure
failures. Metrics were recomputed independently from saved predictions.

| Dataset | Method | EM | F1 | Answered | Selective accuracy | Support recall | Tokens | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Structured-TDCA + Dense | 0.480 | 0.630 | 0.760 | 0.632 | 1.000 | 442,495 | 311 |
| HotpotQA | Controlled IRCoT + BM25 | **0.520** | **0.722** | **0.920** | 0.565 | 1.000 | 123,372 | 74 |
| 2WikiMultiHopQA | Structured-TDCA + Dense | 0.520 | 0.650 | 0.860 | 0.605 | 1.000 | 348,071 | 299 |
| 2WikiMultiHopQA | Controlled IRCoT + BM25 | **0.620** | **0.774** | **0.940** | **0.660** | 1.000 | 143,792 | 99 |

Hotpot Structured-minus-IRCoT is -0.040 EM (95% paired bootstrap CI
[-0.160, 0.080]) and -0.093 F1 (CI [-0.206, 0.014]). 2Wiki is -0.100 EM
(CI [-0.240, 0.020]) and -0.124 F1 (CI [-0.253, -0.001]); only the 2Wiki F1 interval
excludes zero, in IRCoT's favor. Structured uses 3.59x/2.42x as many logical tokens
and 4.20x/3.02x as many calls on Hotpot/2Wiki. There is no quality-cost Pareto
advantage.

Hotpot tuning contains only two-hop labels. On 2Wiki, Structured EM is 0.523 on 44
two-hop questions and 0.500 on six four-hop questions; IRCoT is 0.568 and 1.000.
The four-hop slice is too small for a standalone significance claim, but its direction
matches MuSiQue validation's long-chain weakness.

## Failure concentration

Both datasets have support recall 1.000, all-gold recall 1.000 and answer-in-context
near 1.0, so these controlled runs do not identify initial retrieval as the bottleneck.
Post-hoc deterministic categories are:

- Hotpot: 24 correct; 14 final-synthesis errors; eight verifier false-reject/missing
  terminal; three budget exhaustion; one incomplete chain.
- 2Wiki: 26 correct; 17 final-synthesis errors; five verifier false-reject/missing
  terminal; one budget exhaustion; one incomplete chain.

The next algorithmic work should therefore focus on generic evidence-to-answer
compression, terminal candidate coverage and verifier recall, while preserving strict
grounding. Entity/question patches are not justified. Scheduler attribution also
remains weak: only 28% of Hotpot and 18% of 2Wiki plans expose more than one ready
slot, so most examples cannot distinguish scheduler policies.

Per the staged protocol, cross-dataset validation-200 is not expanded: tuning-50 does
not show high accuracy relative to IRCoT. This prevents unnecessary API spend and
post-hoc scaling after a failed quality gate.
The machine-readable decision is frozen in
`configs/cross_dataset_validation_gate.json`; the guarded validation launcher exits
before testing or API access while that gate remains closed.
