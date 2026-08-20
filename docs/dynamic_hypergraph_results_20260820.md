# Dynamic Hypergraph TDCA: development results (seed 20260820)

This report records development evidence, not a held-out or SOTA claim. All runs use
Qwen-plus. The dynamic method is training-free, and the 200-example held-out split
remains sealed.

## Protocol

- Dataset: MuSiQue distractor, deterministic disjoint smoke-20/development-50/
  heldout-200 split.
- Development methods: `dynamic_hypergraph_tdca` A1--A6, frozen
  `structured_tdca`, and IRCoT.
- Official HippoRAG comparison: smoke-20 only, pinned official repository with
  Qwen-plus and MiniLM in the per-question distractor setting. This is a controlled
  implementation comparison, not a reproduction of the paper-default NV-Embed
  configuration.
- Paired intervals: 10,000 bootstrap samples, seed 20260820.
- Safety caps are not treated as a formal compute budget; realized calls, tokens and
  retrievals are reported.

All selected artifacts passed the repository artifact auditor and had zero
infrastructure failures. Every emitted dynamic answer was graph-grounded.

## Selected results

| Method / split | EM | F1 | Answered | LLM calls | Tokens | Retrievals |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic A6 / smoke-20 | 0.350 | 0.410 | 0.600 | 140 | 125,010 | 74 |
| Official HippoRAG / smoke-20 | 0.400 | 0.543 | - | - | - | - |
| Dynamic A6 / development-50 | 0.420 | 0.424 | 0.540 | 349 | 328,702 | 160 |
| Structured TDCA / development-50 | 0.460 | 0.487 | 0.540 | 277 | 352,737 | 109 |
| IRCoT / development-50 | 0.420 | 0.489 | 0.840 | 167 | 269,595 | 166 |

Dynamic versus structured TDCA has an EM difference of -0.040 (95% paired
bootstrap CI [-0.160, 0.080]) and an F1 difference of -0.062 (CI
[-0.189, 0.060]). Dynamic versus IRCoT has an EM difference of 0.000 (CI
[-0.120, 0.120]) and an F1 difference of -0.065 (CI [-0.193, 0.063]). The
smoke-20 dynamic-versus-HippoRAG differences are -0.050 EM (CI [-0.350, 0.250])
and -0.133 F1 (CI [-0.417, 0.150]). None supports a superiority claim.

## Cumulative ablation

| Level | EM | F1 | LLM calls | Retrievals | Observed mechanism |
|---|---:|---:|---:|---:|---|
| A1 | 0.400 | 0.413 | 237 | 132 | coarse plan/editor |
| A2 | 0.400 | 0.413 | 253 | 160 | branch activation 0.28 |
| A3 | 0.420 | 0.433 | 256 | 159 | multi-premise derivation |
| A4 | 0.420 | 0.433 | 356 | 163 | independent soft verification |
| A5 | 0.420 | 0.433 | 356 | 163 | zero revisions/reactivations |
| A6 | 0.420 | 0.424 | 349 | 160 | scheduler activation 2.2/example |

A3 is the only observed quality increase. A4 adds roughly 100 model calls without a
quality gain. A5 is a valid implementation but a null mechanism on this split. A6 is
active, yet does not improve EM and slightly reduces F1. Revision is therefore not
credited as an empirical contribution.

## Budget curve

| Safety caps (calls/tokens/retrievals) | EM | F1 | Realized calls | Realized tokens | Realized retrievals |
|---|---:|---:|---:|---:|---:|
| 8 / 8k / 4 | 0.180 | 0.180 | 274 | 242,657 | 143 |
| 12 / 12k / 6 | 0.400 | 0.404 | 336 | 314,892 | 157 |
| 16 / 16k / 8 | 0.420 | 0.424 | 349 | 328,702 | 160 |
| 24 / 24k / 12 | 0.420 | 0.424 | 349 | 328,872 | 164 |

The curve saturates at the 16/16k/8 development setting. These are per-example
safety caps; the realized totals above are dataset totals.

## Diagnosis and next changes

Gold-document recall is 0.86, while gold-answer candidate presence is only 0.42 and
full-chain completion is 0.48. The dominant error is therefore candidate extraction
and multi-hop graph completion, especially at three and four hops, rather than raw
retrieval access.

The next iteration should focus on general mechanisms rather than question-specific
patches:

1. add relation-aware extraction that converts retrieved evidence into typed claims
   before answer candidate generation;
2. make the root template and variable bindings drive explicit multi-hop joins;
3. trigger revision from measurable contradiction, entropy and evidence-gap events,
   and delete A5 if it remains inactive under adversarial tests;
4. calibrate independent verifier components on development data without collapsing
   absolute support, relative weight, entropy and evidence gap into one probability;
5. rerun matched-compute controls and budget curves before considering the sealed
   heldout split.

Relevant current directions include [HippoRAG 2](https://arxiv.org/abs/2502.14802),
[ChainRAG](https://aclanthology.org/2025.acl-long.1089/),
[ToG-2](https://arxiv.org/abs/2407.10805), and
[PER-QA](https://aclanthology.org/2025.acl-long.1142/). They motivate graph-guided
retrieval and path/evidence control, but do not establish that this implementation
matches their reported settings.

## Reproducibility pointers

- Selected dynamic development artifact:
  `research_outputs/musique_distractor_development_dynamic_hypergraph_tdca_1787255517137774629`
- Structured control:
  `research_outputs/musique_distractor_development_structured_tdca_1787255882382568783`
- IRCoT control:
  `research_outputs/musique_distractor_development_ircot_1787256825790596574`
- Paired results: `research_outputs/paired_dynamic_dev50/`
- Gate: `configs/dynamic_heldout_gate.json`
