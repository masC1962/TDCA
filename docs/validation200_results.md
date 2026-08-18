# Frozen MuSiQue validation-200 results

This is the disjoint validation split from `configs/splits/musique_dev_seed520.json`.
The main configuration was frozen before any validation labels were inspected. All 200
examples remain in metric denominators, including provider failures. No SOTA claim is made.

## Independently verified main result

| Method | EM | F1 | Answered | Selective accuracy | Support recall | All gold recalled | Infrastructure failures | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Structured-TDCA + Dense | 0.400 | 0.453 | 0.610 | 0.656 | 0.955 | 0.885 | 1/200 | 1,629,908 |

The one failure is a DashScope `data_inspection_failed` response. It is retained as an
infrastructure failure, not relabeled as abstention and not removed from the denominator.
It was triggered while sending retrieved source text; no entity- or question-specific
content-filter workaround is added to the frozen run.

## By hop

| Gold hop count | n | EM | F1 | Answered | Support recall | All gold recalled | Mean tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 108 | 0.454 | 0.534 | 0.731 | 0.972 | 0.954 | 7,390 |
| 3 | 56 | 0.482 | 0.510 | 0.643 | 0.946 | 0.857 | 8,816 |
| 4 | 36 | 0.111 | 0.122 | 0.194 | 0.917 | 0.722 | 9,391 |

The 4-hop collapse despite high support recall and answer-in-context rate identifies
long-chain decomposition, state propagation and terminal construction as the dominant
held-out weakness. It is not explained by initial retrieval alone.

Post-hoc categories over all 200 rows are: 80 correct, 54 decomposition errors, 21
verification false-reject/missing-terminal cases, 17 retrieval misses, 11 wrong claim
extractions, nine final-synthesis errors, seven budget-exhaustion cases and one provider
failure. Of the 36 four-hop questions, only four are exact; 12 are categorized as
decomposition errors, nine retrieval misses and six budget cases.

Predicted plan length equals the gold hop label on 63.0% of two-hop, 58.9% of three-hop
and 50.0% of four-hop rows. Four-hop plans are under-decomposed 27.8% and over-decomposed
22.2%. These labels are used only post hoc and never enter prompts.

Calibration also degrades out of sample: ECE is 0.187. The 56 rows reported near 0.95
confidence are only 55.4% exact, and the 47 rows reported as 1.00 are 72.3% exact.
Therefore `calibrated_confidence` should currently be read as a ranking/selection score,
not a calibrated probability. Thresholds are not retuned on validation labels.

## Paired baseline and significance

| Method | EM | F1 | Answered | Selective accuracy | Support recall | All gold recalled | Failures | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled IRCoT + BM25 | 0.380 | **0.473** | **0.775** | 0.490 | 0.913 | 0.790 | 0/200 | 5,263 |
| Structured-TDCA + Dense | **0.400** | 0.453 | 0.610 | **0.656** | **0.955** | **0.885** | 1/200 | 8,150 |

The paired EM difference is +0.020 with 95% bootstrap CI [-0.050, 0.090]. The paired
F1 difference is -0.020 with 95% CI [-0.089, 0.048]. Neither metric demonstrates a
significant advantage. Structured uses 1.55x the tokens, so it also does not establish
a quality/cost Pareto advantage over IRCoT at this checkpoint. Its defensible advantage
is higher selective accuracy and evidence recall at lower coverage.

By hop, Structured and IRCoT have identical two-hop EM (0.454). Structured is better
on three-hop EM (0.482 vs 0.304) but much worse on four-hop EM (0.111 vs 0.278). This
interaction is the main reason an overall point estimate alone is misleading.

Paired outcomes reinforce the interaction. Overall, Structured changes 28 IRCoT errors
to correct answers but loses 24 IRCoT-correct rows. Two-hop is balanced at 13 wins/13
losses; three-hop is 13/3 in Structured's favor; four-hop reverses to 2/8. The evidence
supports a useful middle-depth working-memory effect, but the current plan/state/token
design does not scale reliably to longer chains.

## Official-code HippoRAG 2 controlled comparison

| Method | EM | F1 | Answered | LLM calls | Prompt tokens | Completion tokens | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structured-TDCA + Dense | 0.400 | 0.453 | 0.610 | 1,219 | 1,478,934 | 150,974 | 3,330 |
| HippoRAG 2 + MiniLM | **0.480** | **0.552** | **1.000** | 6,913 | 3,855,325 | 1,270,375 | 6,101 |

The HippoRAG row is an official-code controlled run at repository commit
`c617143f01477243992a63b2e2151cc003dd3b21`, using Qwen-plus and the pinned MiniLM
embedding backend. It is not the paper-default NV-Embed-v2 reproduction. Its upstream
EM/F1 (0.480/0.552218) match the independent scorer (0.480/0.552217, rounding only),
with zero parser recoveries or failures.

Structured-minus-HippoRAG is -0.080 EM with 95% paired bootstrap CI
[-0.150, -0.010], and -0.099 F1 with CI [-0.168, -0.030]. Both intervals exclude
zero: HippoRAG significantly outperforms the frozen Structured method on this split.
HippoRAG is more expensive, using 5.67x as many Qwen calls, 2.61x prompt tokens,
8.42x completion tokens and 1.83x summed wall time. This is a quality/cost tradeoff,
not a Structured quality or SOTA claim.

By hop, HippoRAG EM is 0.509/0.518/0.333 for 2/3/4-hop questions, compared with
Structured 0.454/0.482/0.111. Its largest practical gain is on four-hop questions,
which reinforces the diagnosed long-chain planning/state-propagation failure.

Stage 5 (1,000 examples) remains closed: Structured is statistically worse than the
strongest completed controlled baseline and has no quality/cost Pareto advantage.
