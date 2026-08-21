# Dynamic Reasoning Hypergraph TDCA v2

## Scope

Dynamic Hypergraph TDCA v2 is a training-free multi-hop QA method whose reasoning
state controls where computation is spent. It is isolated from the frozen v1 path.
The graph is not only a trace container: typed graph state produces the computation
heat, expected value of computation (EVC), operation choice, local budget packet,
revision trigger, and stopping decision.

No inference component receives answers, supporting-document labels, oracle
decompositions, hop counts, or persistent cross-question memory. Qwen-plus proposes
typed claims, validates relational joins, independently scores raw verification
channels, and may propose a structural expansion after a named event. Every state
mutation is deterministic and transactional inside `V2GraphController`.

## State and invariants

The state contains typed subgoals, evidence, canonicalized claims, answers,
multi-source hyperedges, branch assignments, and four histories: diffusion,
allocation, supersession, and termination. Each claim retains separate:

- absolute support;
- relative weight among actual answer alternatives;
- candidate-set entropy;
- evidence gap;
- contradiction pressure;
- downstream answer impact;
- dependency-unlock value;
- computation heat.

These channels are never prematurely collapsed into one probability. Each claim is
grounded to a source span. When the answer occurs in the source position of a raw
triple, extraction canonically reverses the tuple so `ClaimNode.value` is always the
slot projection while preserving the source triple in provenance. This removes the
common subject-versus-object terminal error without question-specific rules.

The serialized state is sealed by a controller hash. Validation rejects external
mutation, unknown citations, untyped claims, active joined claims without a
multi-premise hyperedge, duplicate allocation IDs, incomplete EVC records, and
accepted answers supported by invalidated claims or edges.

## Belief update and typed diffusion

After operation \(o_t\), the controller recomputes the affected graph closure and
updates local belief channels. Typed directional messages are then constructed for
evidence support, claim-to-subgoal resolution, subgoal demand, contradiction links,
execution dependencies, and hyperedge premise/result relations.

For channel \(c\), node \(v\), restart \(r\), and propagation decay \(d\):

\[
x^{(k+1)}_{v,c}=\operatorname{clip}_{[0,1]}\left[
r x^{(0)}_{v,c}+(1-r)\left((1-d)x^{(k)}_{v,c}
+d\,\operatorname{mean}_{u\rightarrow v}(w_{uvc}x^{(k)}_{u,c})\right)\right].
\]

Support flows forward; answer impact and heat flow backward toward unresolved
premises. The complete typed message list and each iteration result are serialized.

Initial heat is a normalized additive utility over uncertainty, answer impact,
evidence gap, contradiction, and unlock value. Component channels remain available
for audit and revision.

## Explicit JOIN and revision

JOIN discovery is deterministic: compatible paths may unify value-to-subject,
subject-to-value, a shared subject, or a shared value. Candidate ranking prioritizes
dependency-lineage coverage and the shortest sufficient chain, and a per-question
attempt cap bounds graph growth. There are no entity-, relation-, dataset-, or
question-ID rules. Qwen validates open-endpoint compositions using cited grounded
spans. A valid result creates both a new typed claim and a multi-premise hyperedge.

For a pure variable-binding projection, Qwen has already independently scored both
premises and the target premise's dependency consistency. Exact endpoint/type
matching plus the normal support gate therefore materializes the projection
hyperedge deterministically without another generation call. Sequential
decompositions are explicit dependency-claim plus current-relation joins before slot
commit, so three/four-hop paths form auditable nested state rather than implicit
string substitution.

Contradiction pressure, support collapse, entropy rise, or evidence-gap rise can
trigger revision. Revision never deletes history. It invalidates a versioned
downstream closure, rejects dependent answers, records the evidence and trigger, and
reopens affected subgoals through the controller.

## Adaptive computation allocation

For every executable operation, the allocator records raw and min-max-normalized
components and uses the deterministic additive policy:

\[
\widehat{\mathrm{EVC}}(o)=\max\{0,
w_h h+w_u u+w_a a+w_d d+w_n n+w_r r-w_c c-w_g g\}.
\]

The terms are graph heat, expected uncertainty reduction, downstream answer impact,
dependency unlock, evidence novelty, recovery value, expected cost, and graph-growth
risk. The selected packet controls completion tokens, retrieval top-k, claim cap,
verification samples, branch width, and revision allowance within global caps.
Structured calls have schema-safe token floors coupled to output cardinality: low
heat reduces candidate count but cannot request an unserializable JSON budget.

Every selected allocation, including a rejected proposal or malformed model output,
gets a globally unique ledger row with predicted EVC, requested resources, measured
LLM/token/retrieval deltas, completion state, and failure reason. Thus failed calls
cannot disappear from matched-compute or Pareto accounting.

## Meta-stop and protocol

The deterministic meta-policy returns exactly one of:

- `ANSWER`: an accepted answer has active supporting claims, evidence, and edge;
- `ABSTAIN`: no executable positive-value computation remains;
- `BUDGET_EXHAUSTED`: useful computation exists but violates a global safety cap.

Budget exhaustion can never emit an unsupported answer. The development safety cap
is 16 calls, 16,000 logical tokens, and 8 retrievals; formal reporting additionally
uses matched-compute controls and a budget curve. Thresholds may be adjusted only on
MuSiQue smoke-20/development-50, then are frozen for heldout and cross-dataset runs.
The fixed seed is `20260820`.

The machine-readable gate is `configs/dynamic_v2_hard_gate.json`. Heldout launchers
remain closed until all infrastructure, reasoning, dynamic-revision, allocation,
Pareto, and termination requirements pass. `scripts/evaluate_dynamic_v2_gate.py`
produces the auditable decision report; it does not silently open the gate.

The first mechanism-complete smoke run is an explicit negative result; see
[`dynamic_hypergraph_tdca_v2_smoke_results_20260821.md`](dynamic_hypergraph_tdca_v2_smoke_results_20260821.md).
