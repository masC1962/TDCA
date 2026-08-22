# Dynamic Hypergraph TDCA v2.1 — Safe-Failure Closure

## Outcome

The development campaign stopped under the preregistered safe-failure rule. The campaign used 1,421 uncached Qwen provider calls and 1,571,736 provider-reported tokens. The 1.5M-token cap was crossed during the first matched-compute control, so that run was not resumed, the fixed-order control was not started, and the heldout split remains sealed.

This is not a successful hard-gate result. The machine-readable record is `analysis_outputs/dynamic_v21_safe_failure/stop_report.json`.

## Implemented research system

The repository now contains the intended training-free Dynamic Reasoning Hypergraph core:

- three distinct state layers: relation-light corpus memory, question-local activated evidence/entity graph, and controller-owned proof hypergraph;
- a query-graph compiler with typed variables and dependency bindings;
- typed claim extraction, canonical values, provenance-preserving source triples, and independent raw verification scores;
- deterministic goal-path JOIN, multi-premise JOIN, and numeric comparison JOIN without question-ID rules;
- query-aware hypergraph diffusion over uncertainty, answer impact, evidence gap, contradiction pressure, and unlock value;
- operation × fidelity EVC allocation with normalized components, predicted value, actual cost, state delta, realized utility, and feedback into later allocation;
- event-triggered graph editing, cycle-safe preflight, controller-only mutation, belief supersession, and natural revision;
- explicit `ANSWER`, `ABSTAIN`, and `BUDGET_EXHAUSTED` terminal states with unsupported-answer prevention.

## Development evidence

On the frozen MuSiQue development-50 split, v2.1 raised candidate presence from 0.42 to 0.52 and full-chain completion from 0.48 to 0.60. It produced 15 auditable 3/4-hop JOIN cases, including 6 cases where an n-ary JOIN was used downstream. All 50 examples had complete EVC and feedback traces; all 50 showed non-uniform allocation, and feedback affected a later allocation in 15 cases. Infrastructure failure rate and unsupported-answer count were both zero.

The sealed 60-case VitaminC revision evaluation passed its preregistered gate: precision 0.926, recall 0.833, FPR 0.067, with 60/60 complete predictions and zero invariant violations. Its decision rule was frozen after development scoring; no evaluation-driven tuning occurred.

The result also exposes two important weaknesses. First, answer F1 fell from v1's 0.424 to 0.384 even though candidate presence and chain completion improved, indicating that final candidate selection/answer projection remains weaker than state construction. Second, the candidate-presence gain is exactly the +0.10 boundary and is therefore not robust to ordinary sampling variance.

## Why the hard gate remains closed

The adaptive run is complete, but the uniform control reached only 37/50 examples before the campaign token cap was crossed. It alone added 192 provider calls and 206,791 tokens. The fixed-order control was consequently not started. Without both complete, identity-matched controls, a Pareto improvement point cannot be established. The heldout gate therefore remains closed by construction.

## Next research direction

A future campaign should be separately preregistered rather than extending this one. Its first goal should be completing matched controls with an enforced campaign-wide per-response budget guard and isolated cache accounting. Algorithmically, work should target the gap between “correct candidate/chain exists” and “correct answer is selected”: graph-level belief calibration, answer-slot projection, contradiction-aware candidate competition, and EVC policies that prioritize unresolved terminal bindings. Only after development quality and the matched Pareto gate pass should a new heldout run be authorized.
