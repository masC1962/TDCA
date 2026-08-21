from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class SlotStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return {"passage_id": self.passage_id, "title": self.title, "text": self.text}


@dataclass
class QAExample:
    qid: str
    question: str
    passages: list[Passage]
    answers: list[str] = field(default_factory=list, repr=False)
    gold_document_ids: list[str] = field(default_factory=list, repr=False)
    gold_titles: list[str] = field(default_factory=list, repr=False)
    oracle_decomposition: list[dict[str, Any]] = field(default_factory=list, repr=False)
    answerable: bool = True
    hop_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def inference_view(self) -> dict[str, Any]:
        """Return only fields that a normal method is allowed to observe."""
        return {
            "qid": self.qid,
            "question": self.question,
            "passages": [p.public_dict() for p in self.passages],
        }


@dataclass
class VariableBinding:
    variable: str
    source_slot: str


@dataclass
class ReasoningSlot:
    slot_id: str
    subquestion_template: str
    answer_type: str = "entity"
    dependencies: list[str] = field(default_factory=list)
    variable_bindings: list[VariableBinding] = field(default_factory=list)
    output_variable: str = "$answer"
    terminal: bool = False
    status: SlotStatus = SlotStatus.PENDING
    confidence: float = 0.0
    bound_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class ReasoningPlan:
    question: str
    slots: list[ReasoningSlot]
    plan_type: str = "chain"
    source: str = "model"

    def by_id(self) -> dict[str, ReasoningSlot]:
        return {slot.slot_id: slot for slot in self.slots}


@dataclass
class Claim:
    claim_id: str
    subject: str
    relation: str
    object: str
    answer_type: str
    target_slot: str
    source_document_ids: list[str] = field(default_factory=list)
    source_spans: list[str] = field(default_factory=list)
    depends_on_claim_ids: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    entailment_score: float = 0.0
    type_score: float = 0.0
    calibrated_confidence: float = 0.0
    status: ClaimStatus = ClaimStatus.PROPOSED
    contradiction_ids: list[str] = field(default_factory=list)
    created_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class RetrievalHit:
    passage: Passage
    raw_score: float
    rank: int
    retriever: str
    query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage.passage_id,
            "title": self.passage.title,
            "text": self.passage.text,
            "raw_score": self.raw_score,
            "rank": self.rank,
            "retriever": self.retriever,
            "query": self.query,
        }


@dataclass
class Usage:
    llm_calls: int = 0
    provider_attempts: int = 0
    retrieval_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider_prompt_tokens: int = 0
    provider_completion_tokens: int = 0
    wall_seconds: float = 0.0
    cache_hits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def provider_calls(self) -> int:
        return self.provider_attempts


@dataclass
class Prediction:
    qid: str
    question: str
    status: RunStatus
    answer: str | None
    confidence: float
    stop_reason: str
    best_unverified_candidate: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    plan: ReasoningPlan | None = None
    retrieved: list[RetrievalHit] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
            "best_unverified_candidate": self.best_unverified_candidate,
            "rejection_reasons": self.rejection_reasons,
            "claims": [c.to_dict() for c in self.claims],
            "plan": None if self.plan is None else {
                "question": self.plan.question,
                "plan_type": self.plan.plan_type,
                "source": self.plan.source,
                "slots": [s.to_dict() for s in self.plan.slots],
            },
            "retrieved": [h.to_dict() for h in self.retrieved],
            "usage": asdict(self.usage) | {"total_tokens": self.usage.total_tokens},
            "error": self.error,
        }


def prediction_from_dict(row: dict[str, Any]) -> Prediction:
    """Rehydrate a persisted prediction for independent scoring or safe resume."""
    plan_row = row.get("plan")
    plan = None
    if isinstance(plan_row, dict):
        slots = [ReasoningSlot(
            slot_id=str(slot["slot_id"]),
            subquestion_template=str(slot.get("subquestion_template", "")),
            answer_type=str(slot.get("answer_type", "entity")),
            dependencies=[str(value) for value in slot.get("dependencies", [])],
            variable_bindings=[VariableBinding(**binding) for binding in slot.get("variable_bindings", [])],
            output_variable=str(slot.get("output_variable", "$answer")),
            terminal=bool(slot.get("terminal", False)),
            status=SlotStatus(slot.get("status", "pending")),
            confidence=float(slot.get("confidence", 0)),
            bound_question=str(slot.get("bound_question", "")),
        ) for slot in plan_row.get("slots", [])]
        plan = ReasoningPlan(
            question=str(plan_row.get("question", "")), slots=slots,
            plan_type=str(plan_row.get("plan_type", "chain")),
            source=str(plan_row.get("source", "model")),
        )
    claims = []
    for claim_row in row.get("claims", []):
        claim_data = dict(claim_row)
        claim_data["status"] = ClaimStatus(claim_data.get("status", "proposed"))
        claims.append(Claim(**claim_data))
    retrieved = [RetrievalHit(
        Passage(str(hit["passage_id"]), str(hit.get("title", "")), str(hit.get("text", ""))),
        float(hit.get("raw_score", 0)), int(hit.get("rank", 0)),
        str(hit.get("retriever", "")), str(hit.get("query", "")),
    ) for hit in row.get("retrieved", [])]
    usage_row = row.get("usage", {})
    usage_values = {
        key: usage_row[key]
        for key in (
            "llm_calls", "provider_attempts", "retrieval_calls", "prompt_tokens", "completion_tokens",
            "provider_prompt_tokens", "provider_completion_tokens", "wall_seconds", "cache_hits",
        )
        if key in usage_row
    }
    if "provider_attempts" not in usage_values:
        # Backward-compatible interpretation of artifacts written before actual
        # HTTP attempts were tracked separately.
        usage_values["provider_attempts"] = max(
            0, int(usage_values.get("llm_calls", 0)) - int(usage_values.get("cache_hits", 0)),
        )
    usage = Usage(**usage_values)
    return Prediction(
        qid=str(row["qid"]), question=str(row.get("question", "")), status=RunStatus(row["status"]),
        answer=row.get("answer"), confidence=float(row.get("confidence", 0)),
        stop_reason=str(row.get("stop_reason", "")),
        best_unverified_candidate=row.get("best_unverified_candidate"),
        rejection_reasons=list(row.get("rejection_reasons", [])), claims=claims, plan=plan,
        retrieved=retrieved, usage=usage, error=row.get("error"),
    )
