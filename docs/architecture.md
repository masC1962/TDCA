# Architecture

## Boundary

The research package targets per-question structured working memory. Episodic memory
is disabled by configuration, preventing cross-test contamination. Long-term agent
memory is a separate future project.

## Dependency DAG

`ReasoningPlan` contains validated `ReasoningSlot` nodes. Edges point from prerequisite
slots to consumers. A consumer is executable only when every predecessor has a
verified claim. Variables such as `$bridge_1` are declared as predecessor outputs and
are substituted into the successor question with an auditable claim-ID list.

Normal planning sees only the root question. Official decomposition fields remain in
the evaluation object and are accessible only when `oracle_decomposition=true`.

## Working memory

Each `Claim` contains subject, relation, object, answer type, target slot, evidence
documents/spans, claim dependencies and independent scores. Its lifecycle is:

`proposed -> verified | rejected -> superseded`

Conflicting verified claims link to each other; the stronger claim may supersede the
weaker one. Rejected and superseded claims cannot unlock downstream slots or become
final answers.

## Retrieval

Retrieval operates on the current bound subquestion. BM25 stores absolute raw scores.
Dense retrieval requires `sentence-transformers`; missing dependencies raise an error
unless `explicit_tfidf` is deliberately configured, in which case every trace labels
the fallback. Hybrid retrieval uses reciprocal-rank fusion without pretending its
score is an absolute probability.

Entity-aware retrieval is a generic surface-form reranker over the current bound query;
it never sees supporting labels. In distractor mode, each question has its own
retriever. In global mode, one retriever is constructed once over a hashed corpus and
reused. Dense/TF-IDF diagnostic indices can be materialized and are rejected at load
time when the corpus fingerprint or encoder differs.

Evidence rendering is shared by claim extraction, independent verification and final
synthesis. The frozen experiments use `evidence_compaction: none`. An optional
`query_sentence` mode ranks verbatim sentences by generic lexical overlap with the
current bound query, retains passage IDs/titles, and falls back deterministically when
there is no overlap. It is a future token/policy-exposure ablation, not a post-hoc
repair of validation-200 and not an entity- or question-specific content filter.

## Verification and final synthesis

The claim generator cannot verify itself implicitly. The verifier checks grounded
source spans, evidence relevance, relation entailment, answer type, dependency
consistency and contradiction. Confidence is conservatively calibrated from absolute
component scores.

Finalization has two stages only: select a verified terminal claim, then run an
evidence-constrained final verification/synthesis call. Results distinguish answer,
abstention and infrastructure failure; an empty string is never overloaded.
