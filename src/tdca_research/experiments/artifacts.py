from __future__ import annotations

import json
import platform
import sys
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import ResearchConfig
from ..models import Prediction
from ..utils import append_jsonl, sha256_file, write_json


class ArtifactWriter:
    FILES = [
        "run_manifest.json", "resolved_config.yaml", "environment.json", "predictions.jsonl",
        "retrieval_traces.jsonl", "reasoning_traces.jsonl", "metrics.json", "metrics_by_hop.json",
        "metrics_by_type.json",
        "cost_summary.json", "failures.jsonl",
        "per_example_metrics.jsonl", "estimated_cost.json", "partial_progress.json",
    ]

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

    def initialize(self, config: ResearchConfig, manifest: dict[str, Any], environment: dict[str, Any]) -> None:
        write_json(self.run_dir / "run_manifest.json", {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            **manifest,
        })
        (self.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8")
        write_json(self.run_dir / "environment.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            **environment,
        })
        for name in self.FILES[3:]:
            path = self.run_dir / name
            if not path.exists():
                path.write_text("{}" if name.endswith(".json") else "", encoding="utf-8")

    def write_rows(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_rows(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        append_jsonl(self.run_dir / name, rows)

    def checkpoint_prediction(
        self,
        prediction: Prediction,
        retrieval_rows: list[dict[str, Any]],
        reasoning_rows: list[dict[str, Any]],
        completed: int,
        total: int,
    ) -> None:
        """Persist enough state to audit a terminated long run.

        A completed run rewrites the same JSONL files canonically, so checkpointing
        does not alter final artifacts or evaluation semantics.
        """
        self.append_rows("predictions.jsonl", [prediction.to_dict()])
        self.append_rows("retrieval_traces.jsonl", retrieval_rows)
        self.append_rows("reasoning_traces.jsonl", reasoning_rows)
        if prediction.status.value == "infrastructure_failure":
            self.append_rows("failures.jsonl", [prediction.to_dict()])
        write_json(self.run_dir / "partial_progress.json", {
            "status": "running",
            "completed": completed,
            "total": total,
            "last_qid": prediction.qid,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    def write_predictions(self, predictions: list[Prediction]) -> None:
        self.write_rows("predictions.jsonl", [prediction.to_dict() for prediction in predictions])
        self.write_rows("failures.jsonl", [prediction.to_dict() for prediction in predictions if prediction.status.value == "infrastructure_failure"])

    def write_metrics(
        self, metrics: dict[str, Any], by_hop: dict[str, Any], predictions: list[Prediction],
        by_type: dict[str, Any] | None = None,
    ) -> None:
        write_json(self.run_dir / "metrics.json", metrics)
        write_json(self.run_dir / "metrics_by_hop.json", by_hop)
        write_json(self.run_dir / "metrics_by_type.json", by_type or {})
        cost_summary = {
            "llm_calls": sum(prediction.usage.llm_calls for prediction in predictions),
            "provider_calls": sum(prediction.usage.provider_calls for prediction in predictions),
            "provider_attempts": sum(prediction.usage.provider_attempts for prediction in predictions),
            "retrieval_calls": sum(prediction.usage.retrieval_calls for prediction in predictions),
            "prompt_tokens": sum(prediction.usage.prompt_tokens for prediction in predictions),
            "completion_tokens": sum(prediction.usage.completion_tokens for prediction in predictions),
            "provider_prompt_tokens": sum(prediction.usage.provider_prompt_tokens for prediction in predictions),
            "provider_completion_tokens": sum(prediction.usage.provider_completion_tokens for prediction in predictions),
            "wall_seconds": sum(prediction.usage.wall_seconds for prediction in predictions),
            "cache_hits": sum(prediction.usage.cache_hits for prediction in predictions),
            "estimated_cost": None,
            "estimated_cost_note": "No versioned Qwen-plus price schedule is configured; tokens are reported without inventing a monetary cost.",
            "token_semantics": {
                "prompt_tokens_and_completion_tokens": "logical model-equivalent usage including cache hits",
                "provider_prompt_tokens_and_provider_completion_tokens": "uncached successful provider generations only",
                "provider_attempts": "actual HTTP attempts including retry attempts; failed attempts may not expose token usage",
            },
        }
        write_json(self.run_dir / "cost_summary.json", cost_summary)
        write_json(self.run_dir / "estimated_cost.json", {
            "currency": None,
            "amount": None,
            "status": "not_computed",
            "reason": cost_summary["estimated_cost_note"],
            "prompt_tokens": cost_summary["prompt_tokens"],
            "completion_tokens": cost_summary["completion_tokens"],
            "provider_prompt_tokens": cost_summary["provider_prompt_tokens"],
            "provider_completion_tokens": cost_summary["provider_completion_tokens"],
        })

    def finalize_manifest(self) -> None:
        path = self.run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, manifest)
        progress_path = self.run_dir / "partial_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        progress.update({"status": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(progress_path, progress)

    def checksums(self) -> None:
        checksums = {}
        for path in self.run_dir.rglob("*"):
            if path.is_file() and path.name != "artifact_checksums.json":
                checksums[str(path.relative_to(self.run_dir))] = sha256_file(path)
        write_json(self.run_dir / "artifact_checksums.json", checksums)


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "networkx", "openai", "pydantic", "PyYAML", "scikit-learn", "sentence-transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions
