# Paper completion manifest

This file maps every provisional part of the draft to the experiment or artifact needed to replace it. Update the paper from generated summaries, never by hand-copying untracked numbers without provenance.

| Paper location | Required artifact | Minimum contents |
|---|---|---|
| Abstract | Frozen aggregate summary | Main quality delta, matched cost, Pareto claim, reliability result |
| Table 1 | Cross-dataset matched-compute report | MuSiQue, HotpotQA, 2Wiki; same IDs; EM/F1/candidate/chain/calls/tokens |
| Figure 2 | Budget sweep export | At least four budget points per method; calls, tokens, retrievals, latency, cost |
| Table 2 | Paired ablation report | Full method, no diffusion, no joins, no revision, no feedback, uniform, fixed |
| Revision paragraph | Natural and adversarial revision audit | Precision, recall, case counts, false-positive categories |
| EVC paragraph | Packet-level calibration export | Spearman correlation, sign accuracy, MAE, observable regret, coverage |
| Termination appendix | Terminal-state audit | Three-way confusion matrix and unsupported-answer count |
| Per-hop appendix figure | Dataset stratification export | Quality and cost by 2/3/4-hop group |
| Trace appendix figure | Auditable trace renderer | At least one successful 3-hop and one 4-hop case, plus one revision case |
| Limitations/conclusion | Frozen results and failure taxonomy | Explicit negative results and boundary conditions |

## Required primary baselines

- Direct prompting and fixed top-k RAG
- IRCoT
- Adaptive-RAG
- HippoRAG
- HippoRAG 2
- TDCA v1
- Uniform and fixed allocation controls

If a baseline cannot share the same index or reader due to method assumptions, report both the native setting and the nearest controlled setting, and state the mismatch.

## Statistical reporting

- Pair examples by immutable question ID.
- Report bootstrap 95% confidence intervals.
- Use a paired randomization or permutation test for the primary quality comparison.
- Correct for multiple comparisons in the ablation family.
- Report effect sizes and counts, not only p-values.
- Do not select a budget point using final test outcomes.

## Stop conditions for paper claims

Do not claim graph-state adaptive allocation is successful unless all infrastructure and termination gates pass, candidate presence and full-chain improvements meet the frozen targets, allocation is non-uniform under counterfactual checks, EVC traces are complete, and at least one matched-compute Pareto improvement exists.
