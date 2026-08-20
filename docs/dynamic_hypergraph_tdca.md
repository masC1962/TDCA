# Dynamic Hypergraph TDCA

This is a new method, `dynamic_hypergraph_tdca`, isolated from the frozen
`structured_tdca` implementation. It is training-free. The initial 16-call,
16,000-token and 8-retrieval limits are development safety caps, not the formal
experimental budget.

## State and invariants

The structural belief state may contain contradiction and revision links. A separate
execution dependency graph is always a DAG. Derivation hyperedges are also acyclic
within each committed graph version. The transactional `GraphController` is the only
component allowed to mutate state; an invalid proposal is rejected without changing
the input graph.

The typed nodes are `SubgoalNode`, `ClaimNode`, `EvidenceNode`, and `AnswerNode`.
Every node and hyperedge has operation-level provenance. Hard limits cover nodes,
hyperedges, graph operations, graph revisions, per-candidate revisions, active
branches, candidates per subgoal, retrievals, and execution depth.

The controller supports eight explicit operations: `EXPAND`, `BRANCH`, `RETRIEVE`,
`VERIFY`, `MERGE`, `PRUNE`, `COMMIT`, and `REVISE`. Each accepted operation records
before/after state hashes and the exact created, updated, or pruned IDs.

## Candidate reasoning

Candidate generation is batched but produces independently plausible, evidence-
grounded candidates. Verification returns independent raw components: grounding,
entailment, answer-type match, dependency consistency, retrieval-rank support,
contradiction risk, and raw model confidence.

Only after all raw scores are fixed does deterministic fusion compute absolute
support and relative softmax weight. Set entropy and evidence gap remain separate
state variables. They are not prematurely compressed into a single probability.

High support with low entropy and a sufficient margin permits commit. Ambiguous
candidate sets branch lazily only when downstream queries can differ. A new retrieval
round reactivates preserved candidates for a fresh independent raw-score pass.
Pruning is an explicit reversible-history operation, while reactivation requires an
explicit `REVISE`, a cooldown, and revision budget.

## Planning, scheduling, and answers

The initial planner creates at most two obvious subgoals plus the root objective. The
event-triggered graph editor runs only on named failure or high-uncertainty events and
may propose structural operations, never answers. All proposals pass through the
controller.

When multiple operations are ready, each utility component is min-max normalized
within that ready set. The scheduler then applies a deterministic additive utility
over uncertainty reduction, dependency unlock, answer impact, evidence novelty,
recovery value, expected cost, and graph-growth risk. A singleton ready set is logged
but is not counted as scheduler activation.

An answer can only be emitted from an `AnswerNode` with supporting claim IDs,
evidence IDs, and a derivation edge. Rendering is deterministic. There is no
free-form finalizer that can introduce an unsupported answer. Before A3, the required
answer-provenance edge is unary; A3 activates genuine multi-premise hyperedges.

## Leakage boundary

The runtime passes only `QAExample.inference_view()` to this method. It contains the
question, qid, and public passages. Gold answers, decompositions, support IDs, titles,
hop labels, and metadata are unavailable to planner, retriever, generator, verifier,
editor, scheduler, and terminal reasoner. Gold fields are used only after prediction
for evaluation and mechanism diagnostics.

## Cumulative ablations

The same implementation is used throughout:

| Level | Added mechanism |
|---|---|
| A0 | frozen `structured_tdca` control |
| A1 | adaptive coarse planning and event-triggered editing |
| A2 | CandidateSet preservation and lazy branching |
| A3 | multi-premise derivation hyperedges |
| A4 | learned soft verification with independent raw scoring |
| A5 | explicit bounded revision/reactivation |
| A6 | normalized deterministic operation scheduler |

## Artifacts and metrics

Dynamic runs retain all standard artifacts and additionally write
`dynamic_graphs.jsonl`, `dynamic_per_example_metrics.jsonl`,
`dynamic_metrics.json`, and `dynamic_metrics_by_hop.json`. These cover operation
counts, candidate survival, branch activation, pruning, commit reversals, revisions,
scheduler activation, ready-operation count, graph peaks, terminal grounding, gold-
candidate survival/false prune (evaluation only), and attributed calls/tokens.

## Development protocol

`configs/splits/musique_dynamic_seed20260820.json` contains disjoint 20/50/200
splits drawn only from MuSiQue IDs unused by the old seed-520 manifest. Thresholds may
be changed on smoke/development only. The heldout gate is closed by default.

```bash
bash scripts/run_dynamic_research.sh tests
bash scripts/run_dynamic_research.sh smoke20
bash scripts/run_dynamic_research.sh development50
bash scripts/run_dynamic_research.sh ablations50
bash scripts/run_dynamic_research.sh budget_curve50
```

Formal comparisons must use the same model, retriever, corpus, IDs, `top_k`, and cache
policy where the method permits it, and must report realized calls/tokens/retrievals.
HippoRAG remains an official-code controlled baseline; no claim of paper-default
reproduction or SOTA superiority is made until the corresponding protocol is run.

The completed seed-20260820 development results and failure analysis are reported in
[`dynamic_hypergraph_results_20260820.md`](dynamic_hypergraph_results_20260820.md).
They do not currently justify a superiority claim or opening the heldout split.
