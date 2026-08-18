# Reproduction Status

| Method | Status | Label |
|---|---|---|
| Closed-book/BM25/Dense/Hybrid | adapters and fixed-split smoke runs implemented; BM25/Dense/Hybrid tuning-50 complete | controlled local implementation |
| IRCoT | controlled interleaving adapter run on smoke-20, tuning-50 and validation-200 | controlled reimplementation, not official |
| Legacy TDCA | frozen snapshot, mock smoke, real Qwen canary and tuning-50 complete | legacy local |
| HippoRAG 2 | official repository pinned and isolated; end-to-end canary, smoke-20, distractor tuning-50 and validation-200 complete | official code, controlled configuration; not paper-default reproduction |
| KiRAG | repository/commit recorded; training-dependent | official reproduction pending |
| KGEIR | repository/commit recorded; isolated dependency audit pending | official reproduction pending |
| HopRAG | official HEAD recorded; Neo4j/global graph build not installed | global-corpus reproduction pending; no license file verified |
| Youtu-GraphRAG | official ICLR-2026 repository HEAD recorded | global-corpus reproduction pending |
| SAG | official benchmark HEAD recorded; requires separate embedding/reranker services and databases | blocked under current authorized endpoints |
| CIRAG | trajectory-distilled/LoRA method | paper-reported only; excluded from training-free controlled track |
| StepChain GraphRAG | no reliable official repository confirmed | paper-reported only |

The active Linux environment is Docker container `mc_env` on a server with eight
24-GiB RTX 3090 GPUs (enumerated as devices 0-7). Qwen-plus health checks and real runs pass. Dense retrieval uses
`sentence-transformers/all-MiniLM-L6-v2`; the server cannot reach the default
Hugging Face endpoint, so `HF_ENDPOINT=https://hf-mirror.com` is required. The model
resolved to repository commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

HippoRAG is installed in `external_repos/HippoRAG/.venv_controlled` so its pinned and
eager-imported dependencies do not overwrite the validated TDCA environment. The
repository HEAD was verified as `c617143f01477243992a63b2e2151cc003dd3b21`.
The official sample, a frozen-ID 20-question mini-global run, and a fair per-question
distractor run completed. See
`docs/smoke20_results.md` and
`research_outputs/hipporag2_controlled_tdca_distractor20.json`.

Verified (97 research tests and five frozen legacy tests at the final checkpoint):

- The complete one-command suite passes, including artifact-integrity and analysis-
  manifest mutation checks.
- Full MuSiQue dev contains 2,417 valid distractor examples.
- Fixed 20/50/200/1,000 IDs are pairwise disjoint.
- Full HotpotQA dev has 7,405 controlled-distractor examples. Pinned 2Wiki dev has
  12,576 examples (9,825 two-hop and 2,751 four-hop), and both have independent
  disjoint 20/50/200/1,000 manifests at seed 520.
- Unified answer EM/F1 matches the pinned official HotpotQA and 2Wiki scorers on nine
  programmatic normalization/type edge cases each, including yes/no/noanswer and Unicode. MuSiQue has
  no categorical gold answers; HotpotQA has 458 and 2Wiki has 1,295, so the official
  categorical no-partial-credit rule matters only on the cross-dataset tracks.
- Paragraph-level MuSiQue gold IDs are exact even when multiple chunks share a title.
- Qwen smoke results have zero infrastructure failures.
- HippoRAG official-code controlled run produces saved per-example responses and
  independently reproducible metrics.
- HippoRAG distractor tuning-50 reached upstream and independently verified EM/F1
  0.44/0.4534, with Recall@10 0.8633 and strict all-gold@10 0.64.
- On frozen tuning-50, Structured-Dense reached EM/F1 0.48/0.5186 versus repaired
  controlled IRCoT 0.34/0.4153. The paired EM difference is +0.14 with a 95%
  bootstrap interval of [-0.02, 0.30], so this is not yet a significance claim.
- Independently rescored frozen legacy TDCA reached EM/F1 0.16/0.2067 and answered
  rate 0.48 despite legacy title-hit 1.00, confirming a post-retrieval bottleneck.
- Frozen MuSiQue validation-200 Structured-Dense and controlled IRCoT are complete and
  independently rescored. Structured-Dense reached EM/F1 0.40/0.4530 versus IRCoT
  0.38/0.4732. The paired Structured-minus-IRCoT differences are +0.020 EM (95% CI
  [-0.050, 0.090]) and -0.020 F1 (95% CI [-0.089, 0.048]); neither is significant.
  Structured-Dense also used 1.55x as many tokens. Stage 5 is therefore not unlocked.
  One Structured-Dense provider-policy failure remains in the denominator.
- Official-code controlled HippoRAG validation-200 reached EM/F1 0.48/0.5522 versus
  Structured 0.40/0.4530. Structured-minus-HippoRAG is -0.080 EM (95% CI
  [-0.150, -0.010]) and -0.099 F1 (CI [-0.168, -0.030]); both significantly favor
  HippoRAG, although HippoRAG uses substantially more Qwen calls/tokens.
- HotpotQA and 2Wiki controlled smoke-20 and disjoint tuning-50 are complete with zero
  infrastructure failures. At tuning-50, Structured/IRCoT EM/F1 are 0.48/0.630 versus
  0.52/0.722 on Hotpot and 0.52/0.650 versus 0.62/0.774 on 2Wiki. See
  `docs/cross_dataset_tuning50_results.md`.
- Cross-dataset validation-200 was intentionally not run because the frozen tuning-50
  gate failed; this is a staged experimental decision, not an infrastructure failure.

The full HippoRAG 2 Hugging Face repository is about 0.360 GiB; the official Git
checkout already includes roughly 41 MB of MuSiQue, HotpotQA and 2Wiki reproduction
data. Paper-default NV-Embed-v2 reproduction remains pending and must not be conflated
with the controlled MiniLM track.
