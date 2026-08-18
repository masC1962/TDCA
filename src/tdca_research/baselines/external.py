from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExternalBaseline:
    name: str
    source_url: str
    commit: str
    license: str
    implementation: str
    installation_status: str
    adapter: str
    failure_reason: str = ""
    original_config: str = ""
    normalized_config: str = ""


def load_manifest(path: str | Path = "external_baselines/manifest.yaml") -> list[ExternalBaseline]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [ExternalBaseline(**item) for item in data.get("baselines", [])]

