from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocRecord:
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    qid: str
    question: str
    gold_answers: List[str]
    docs: List[DocRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)
