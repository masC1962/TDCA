from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .baselines.simple import run_closed_book, run_rag
from .baselines.ircot import run_ircot
from .config import ResearchConfig
from .data import build_split_manifest, load_examples, select_split, validate_dataset_integrity
from .evaluation import evaluate_predictions, grouped_metrics
from .experiments import ArtifactWriter
from .llm import DeterministicMockLLM, OpenAICompatibleLLM
from .models import Passage, prediction_from_dict
from .reasoning import StructuredReasoner
from .retrieval import build_retriever
from .utils import sha256_file, write_json


def build_llm(config: ResearchConfig, mock=None):
    if mock is not None:
        return mock
    return OpenAICompatibleLLM(
        config.llm_base_url, config.llm_model, config.api_cache_dir, config.prompt_version,
        request_timeout_seconds=config.request_timeout_seconds,
        max_api_attempts=config.max_api_attempts,
    )


def _global_passages(path: str) -> list[Passage]:
    if not path:
        raise ValueError("global setting requires global_corpus_path")
    passages = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            passages.append(Passage(
                passage_id=str(row.get("id") or row.get("doc_id") or index),
                title=str(row.get("title") or ""),
                text=str(row.get("text") or row.get("paragraph_text") or ""),
            ))
    identifiers = [passage.passage_id for passage in passages]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("global corpus passage IDs must be unique")
    return passages


def _source_tree_hash(root: Path = Path(".")) -> str:
    digest = hashlib.sha256()
    paths = []
    for relative_root in (Path("src/tdca_research"), Path("scripts"), Path("external_baselines")):
        candidate = root / relative_root
        if candidate.exists():
            paths.extend(path for path in candidate.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        # Experiment outputs and caches are intentionally outside these roots;
        # reject symlink escapes so external mutable files cannot silently enter
        # the purported source version.
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"source hash path escapes repository root: {path}") from exc
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _code_version(root: Path = Path(".")) -> str:
    tree_hash = _source_tree_hash(root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return f"git:{commit}{'+dirty' if dirty else ''};source-tree-sha256:{tree_hash}"
    except Exception:
        return f"source-tree-sha256:{tree_hash}"


def run(
    config: ResearchConfig,
    mock=None,
    split_manifest_path: str | None = None,
    resume_dir: str | Path | None = None,
) -> Path:
    examples = load_examples(config.dataset_path, config.dataset)
    integrity_report = validate_dataset_integrity(examples, config.setting)
    manifest_data = build_split_manifest(examples, config.split_seed)
    manifest_path = split_manifest_path or config.split_manifest_path
    if manifest_path:
        if not Path(manifest_path).exists():
            raise ValueError(f"configured split manifest not found: {manifest_path}")
        manifest_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_hash = manifest_data.get("dataset_sha256")
        if manifest_hash and manifest_hash != sha256_file(config.dataset_path):
            raise ValueError("split manifest dataset SHA-256 does not match configured dataset")
    selected = select_split(examples, config.split, manifest_data, config.split_seed)
    if not selected and examples:
        raise ValueError(
            f"split {config.split!r} is empty for a dataset with {len(examples)} examples; "
            "use a larger source dataset or an explicit non-overlapping split manifest"
        )
    corpus = _global_passages(config.global_corpus_path) if config.setting == "global" else None
    if corpus is not None:
        # Evaluation-only mapping. The corpus and retriever never receive the labels.
        corpus_ids_by_title = {}
        for passage in corpus:
            corpus_ids_by_title.setdefault(passage.title, []).append(passage.passage_id)
        for example in examples:
            if example.gold_titles:
                example.gold_document_ids = [
                    passage_id for title in example.gold_titles
                    for passage_id in corpus_ids_by_title.get(title, [])
                ]
    llm = build_llm(config, mock)
    run_id = f"{config.dataset}_{config.setting}_{config.split}_{config.method}_{time.time_ns()}"
    run_dir = Path(resume_dir) if resume_dir else Path(config.output_root) / run_id
    writer = ArtifactWriter(run_dir)
    manifest = {
        "experiment_id": run_id,
        "code_version": _code_version(),
        "dataset_path": config.dataset_path,
        "dataset_sha256": sha256_file(config.dataset_path),
        "sample_ids": [example.qid for example in selected],
        "split_seed": config.split_seed,
        "provider": "openai_compatible" if mock is None else "mock",
        "model": llm.model_name,
        "prompt_version": config.prompt_version,
        "baseline_source": config.baseline_source,
        "baseline_commit": config.baseline_commit,
        "cache_enabled": mock is None,
        "dataset_integrity": integrity_report,
    }
    if resume_dir:
        predictions, retrieval_rows, reasoning_rows = _load_resume_state(
            run_dir, config, manifest["dataset_sha256"], manifest["sample_ids"], writer,
        )
        run_id = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["experiment_id"]
    else:
        writer.initialize(config, manifest, {
            "os": platform.platform(),
            "is_linux": sys.platform.startswith("linux"),
            "api_key_present": bool(os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")),
        })
        predictions, retrieval_rows, reasoning_rows = [], [], []
    retriever_kind = _resolved_retriever_kind(config)
    if config.setting == "distractor" and config.dense_index_path:
        raise ValueError(
            "dense_index_path is only valid for a shared global corpus; distractor mode builds a per-question index"
        )
    global_retriever = build_retriever(
        retriever_kind, corpus, config.dense_model, config.dense_fallback, config.dense_index_path,
    ) if corpus is not None else None
    for example_index, example in enumerate(selected[len(predictions):], start=len(predictions) + 1):
        passages = corpus if corpus is not None else example.passages
        retriever = global_retriever or build_retriever(
            retriever_kind, passages, config.dense_model, config.dense_fallback, config.dense_index_path,
        )
        if config.method in {"closed_book"}:
            prediction = run_closed_book(example, llm, config.max_llm_calls, config.max_total_tokens, config.temperature)
            retrieval_trace, reasoning_trace = [], []
        elif config.method in {"bm25_rag", "dense_rag", "hybrid_rag"}:
            prediction = run_rag(
                example, llm, retriever, config.top_k, config.max_llm_calls,
                config.max_total_tokens, config.temperature, config.evidence_char_budget,
            )
            retrieval_trace, reasoning_trace = [], []
        elif config.method == "ircot":
            prediction = run_ircot(
                example, llm, retriever, config.top_k, config.max_steps, config.max_llm_calls,
                config.max_total_tokens, config.final_reserve_tokens, config.temperature,
                config.evidence_char_budget,
            )
            retrieval_trace, reasoning_trace = [], []
        elif config.method == "dynamic_hypergraph_tdca":
            from .dynamic.config import DynamicResearchConfig
            from .dynamic.engine import DynamicHypergraphReasoner

            if not isinstance(config, DynamicResearchConfig):
                raise TypeError("dynamic_hypergraph_tdca requires DynamicResearchConfig")
            prediction, retrieval_trace, reasoning_trace = DynamicHypergraphReasoner(
                llm, retriever, config,
            ).solve(example.inference_view())
        else:
            prediction, retrieval_trace, reasoning_trace = StructuredReasoner(llm, retriever, config).solve(example)
        predictions.append(prediction)
        current_retrieval_rows = [{"qid": example.qid, **row} for row in retrieval_trace]
        current_reasoning_rows = [{"qid": example.qid, **row} for row in reasoning_trace]
        retrieval_rows.extend(current_retrieval_rows)
        reasoning_rows.extend(current_reasoning_rows)
        writer.checkpoint_prediction(
            prediction, current_retrieval_rows, current_reasoning_rows,
            completed=example_index, total=len(selected),
        )
    metrics, metric_rows = evaluate_predictions(selected, predictions)
    by_hop = grouped_metrics(metric_rows, "hop_count")
    by_type = grouped_metrics(metric_rows, "question_type")
    writer.write_predictions(predictions)
    writer.write_rows("retrieval_traces.jsonl", retrieval_rows)
    writer.write_rows("reasoning_traces.jsonl", reasoning_rows)
    writer.write_rows("per_example_metrics.jsonl", metric_rows)
    writer.write_metrics(metrics, by_hop, predictions, by_type)
    if config.method == "dynamic_hypergraph_tdca":
        from .dynamic.metrics import dynamic_mechanism_metrics

        dynamic_metrics, dynamic_by_hop, graph_rows, dynamic_rows = dynamic_mechanism_metrics(
            selected, reasoning_rows,
        )
        writer.write_rows("dynamic_graphs.jsonl", graph_rows)
        writer.write_rows("dynamic_per_example_metrics.jsonl", dynamic_rows)
        write_json(run_dir / "dynamic_metrics.json", dynamic_metrics)
        write_json(run_dir / "dynamic_metrics_by_hop.json", dynamic_by_hop)
    write_json(run_dir / "split_manifest.json", manifest_data)
    writer.finalize_manifest()
    writer.checksums()
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_resume_state(
    run_dir: Path,
    config: ResearchConfig,
    dataset_sha256: str,
    sample_ids: list[str],
    writer: ArtifactWriter,
):
    """Load only the checkpoint prefix declared durable by partial_progress.json."""
    required = ("run_manifest.json", "resolved_config.yaml", "partial_progress.json", "predictions.jsonl")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"resume directory is missing required artifacts: {missing}")
    import yaml

    previous_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    if previous_config != config.to_dict():
        raise ValueError("resume config does not exactly match resolved_config.yaml")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != dataset_sha256 or manifest.get("sample_ids") != sample_ids:
        raise ValueError("resume dataset hash or ordered sample IDs do not match")
    progress = json.loads((run_dir / "partial_progress.json").read_text(encoding="utf-8")) or {}
    completed = int(progress.get("completed", 0))
    if not 0 <= completed <= len(sample_ids):
        raise ValueError("resume checkpoint completed count is invalid")
    prediction_rows = _read_jsonl(run_dir / "predictions.jsonl")
    if len(prediction_rows) < completed:
        raise ValueError("resume checkpoint claims more completed rows than predictions.jsonl contains")
    predictions = [prediction_from_dict(row) for row in prediction_rows[:completed]]
    if [prediction.qid for prediction in predictions] != sample_ids[:completed]:
        raise ValueError("resume predictions are not the expected ordered sample prefix")
    completed_qids = set(sample_ids[:completed])
    retrieval_rows = [row for row in _read_jsonl(run_dir / "retrieval_traces.jsonl") if row.get("qid") in completed_qids]
    reasoning_rows = [row for row in _read_jsonl(run_dir / "reasoning_traces.jsonl") if row.get("qid") in completed_qids]
    # Remove any torn append after the last durable checkpoint before continuing.
    writer.write_rows("predictions.jsonl", [prediction.to_dict() for prediction in predictions])
    writer.write_rows("retrieval_traces.jsonl", retrieval_rows)
    writer.write_rows("reasoning_traces.jsonl", reasoning_rows)
    writer.write_rows("failures.jsonl", [
        prediction.to_dict() for prediction in predictions
        if prediction.status.value == "infrastructure_failure"
    ])
    return predictions, retrieval_rows, reasoning_rows


def _resolved_retriever_kind(config: ResearchConfig) -> str:
    forced = {
        "bm25_rag": "bm25",
        "dense_rag": "dense",
        "hybrid_rag": "hybrid",
    }
    return forced.get(config.method, config.retriever)
