# Dynamic Hypergraph TDCA v2.1: preregistered design

## Research claim

TDCA v2.1 tests whether a training-free, question-local proof hypergraph can turn
unresolved belief structure into marginal computation decisions.  It does not claim
that graph storage alone improves QA and it does not use answer labels during
inference.

## Published-work mapping

The implementation borrows mechanisms, not task-specific rules or trained weights:

| TDCA failure | Published reference | Adopted mechanism | Deliberate difference |
|---|---|---|---|
| OpenIE/NER errors erase paths | HippoRAG, NeurIPS 2024; LinearRAG, ICLR 2026 | persistent passage/entity associations; relation-light memory | typed relations enter only the question-local proof layer |
| graph traversal is query-agnostic | QAFD-RAG, ICLR 2026 | query-seeded activation and typed directional flow | heat allocates reasoning operations, not only retrieved passages |
| graph evidence is too large/noisy | G-Retriever, NeurIPS 2024; Superposition Prompting, ICML 2024 | bounded activated subgraphs and explicit low/medium/high fidelity | no soft-prompt training and no answer-conditioned graph optimizer |
| paths are ungrounded | RoG, ICLR 2024 | query constraints and grounded path plans | no relation-path fine-tuning; canonical unification is deterministic |
| extraction loses intermediate facts | TRACE, EMNLP 2024 Findings | evidence-span-first atomic claims and proof chains | claims retain independent raw scores and controller-owned provenance |
| iterative retrieval lacks a stopping signal | S2G-RAG, ACL 2026 | explicit sufficiency/evidence-gap state | gap participates in EVC with support, entropy and cost kept separate |

Primary sources:

- HippoRAG: https://papers.nips.cc/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html
- LinearRAG: https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee1955739b91db042e659b6f782a5e79-Abstract-Conference.html
- QAFD-RAG: https://proceedings.iclr.cc/paper_files/paper/2026/hash/584f32ccf76d73b85fa2053e795df697-Abstract-Conference.html
- G-Retriever: https://papers.nips.cc/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html
- Superposition Prompting: https://proceedings.mlr.press/v235/merth24a.html
- RoG: https://proceedings.iclr.cc/paper/2024/hash/3e2aeb66481dd63a32421bf032b70384-Abstract-Conference.html
- TRACE: https://aclanthology.org/2024.findings-emnlp.496/
- S2G-RAG: https://aclanthology.org/2026.acl-long.1185/

## Three lifetimes

1. **Corpus memory** is an immutable passage/entity/alias association index. It has
   no answer labels and no generated semantic relations. A global-corpus index is
   shared across questions; distractor settings build the same structure over each
   question's supplied corpus.
2. **Activated graph** is question-local. Retrieval materializes only passages and
   entities relevant to current query constraints and records cross-layer grounding
   edges.
3. **Proof hypergraph** is question-local. It stores typed claims, independent raw
   verification, variable bindings, JOINs, beliefs, revisions and terminal proofs.

## Computation policy

The ready frontier exposes competing operations where safe. Adjustable operations
have explicit low, medium and high fidelity packets.  The additive EVC uses
component-wise normalized signals and predicts marginal gain under diminishing
returns. Every selected packet records its fidelity, predicted EVC, measured cost,
pre/post state, actual utility components and exact-context feedback.

## Evaluation discipline

`configs/dynamic_v21_preregistration.json` freezes the seed, API cap and stop policy.
The old smoke-20 is regression-only. The disjoint development-50 selects the final
configuration. Heldout remains fail-closed until the machine hard gate passes; it is
then executed exactly once with no post-heldout tuning.
