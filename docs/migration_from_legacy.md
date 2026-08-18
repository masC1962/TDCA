# Migration from Legacy

The root implementation is retained for historical experiments and invoked by
`legacy/tdca_v0/run.py`; the non-algorithmic batch wrapper is
`legacy/tdca_v0/run_batch.py`. The new package never imports `tdca_scheduler.py`.

Legacy concepts map as follows:

| Legacy | Research path |
|---|---|
| mutable heterogeneous node bag | validated dependency DAG plus typed claims |
| normalized context score | raw retriever score plus independent verifier |
| memory/root-memory variants | working claim lifecycle and terminal slot |
| TCC/TMC/promotion layers | one candidate-construction and one finalization stage |
| many reheating bonuses | expected utility or stable diffusion ablation |
| ambiguous empty final string | answer/abstain/infrastructure status |

Old outputs are not deleted or rewritten. New runs use `research_outputs/`.
