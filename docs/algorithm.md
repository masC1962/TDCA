# Algorithm

Let the reasoning plan be a DAG `G=(V,E)`. Each slot `v` is ready only when all
predecessors have a verified claim. For a ready slot, the expected-utility scheduler
uses

```text
U(v) = I(v) * D(v) * E(v) * C(v) / K(v)
```

where `I` is expected information gain, `D` is dependency-unlock value, `E` is the
evidence gap, `C` is confidence need, and `K` is expected cost. The current reference
implementation uses deterministic, bounded proxies for these values so that each term
can be ablated and logged.

## Diffusion scheduler

The simplified TDCA ablation is a stable lazy random walk:

```text
T_(t+1) = decay * ((1-alpha) I + alpha P^T) T_t
```

`P` is row-stochastic over dependency edges; sinks retain their propagated mass.
For non-negative `T`, temperatures remain non-negative and total heat after one step
is at most `decay * sum(T)` up to floating error. This replaces the legacy collection
of task-specific reheating bonuses with one testable operator.

## Complexity

With `n` slots, `m` dependency edges, `k` retrieved passages and at most `S` steps:

- DAG validation: `O(n+m)`.
- One diffusion step: `O(n+m)`.
- BM25 query: `O(N * |q|)` in the reference in-memory implementation.
- Working-memory selection: `O(c)` for `c` claims; this can later be indexed by slot.
- LLM work is bounded by explicit call and total-token budgets with a final reserve.

## Distinction from related methods

- IRCoT interleaves free-form reasoning and retrieval; this method additionally
  enforces a typed dependency DAG and verified variable bindings.
- HippoRAG 2 primarily strengthens global non-parametric retrieval and associativity;
  this method focuses on the online working-memory reasoning bottleneck and can use a
  HippoRAG retriever through an external adapter.
- Generic GraphRAG constructs graph context; this method treats claims and dependency
  completion as the execution state and explicitly optimizes which unresolved slot to
  execute under cost.

These distinctions are hypotheses to test, not evidence of superiority.

