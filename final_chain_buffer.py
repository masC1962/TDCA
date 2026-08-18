from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils import canonicalize_state_text, lexical_jaccard, normalize_text, relation_signature


def _simple_answer_type(question: str) -> str:
    q = canonicalize_state_text(question).lower()
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
        return "yesno"
    if "what year" in q or q.startswith("when ") or re.search(r"\b(date|born|died|founded)\b", q):
        return "date"
    if q.startswith("who ") or " who " in q or "director" in q or "author" in q:
        return "person"
    if "what country" in q or "which country" in q:
        return "country"
    if "where " in q or "what city" in q or "located in" in q:
        return "location"
    if q.startswith("how many") or "number of" in q:
        return "quantity"
    return "generic"


def canonical_slot_key(
    slot_question: str,
    slot_type: Optional[str] = None,
    slot_role: Optional[str] = None,
    anchor_entity: Optional[str] = None,
) -> str:
    normalized = normalize_text(canonicalize_state_text(slot_question))
    rel = relation_signature(slot_question) or "generic"
    answer_type = normalize_text(slot_type or "") or _simple_answer_type(slot_question)
    role = normalize_text(slot_role or "") or "generic"
    anchor = normalize_text(anchor_entity or "")
    if not anchor:
        entities = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5}", canonicalize_state_text(slot_question))
        anchor = normalize_text(entities[-1]) if entities else ""
    return "|".join([rel, answer_type, role, anchor, normalized])


@dataclass
class FinalChainRecord:
    target_question: str
    slot_key: str
    slot_role: str
    answer_text: str
    answer_type: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    support_score: float = 0.0
    derived_from_state: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_question": self.target_question,
            "slot_key": self.slot_key,
            "slot_role": self.slot_role,
            "answer_text": self.answer_text,
            "answer_type": self.answer_type,
            "evidence_ids": list(self.evidence_ids),
            "support_score": self.support_score,
            "derived_from_state": self.derived_from_state,
            "depends_on": list(self.depends_on),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


class FinalChainBuffer:
    def __init__(self) -> None:
        self.records: List[FinalChainRecord] = []

    def add_candidate(
        self,
        target_question: str,
        slot_key: str,
        slot_role: str,
        answer_text: str,
        answer_type: str | None = None,
        evidence_ids: list[str] | None = None,
        support_score: float = 0.0,
        derived_from_state: str | None = None,
        depends_on: list[str] | None = None,
        source: str = "unknown",
        metadata: dict | None = None,
    ) -> None:
        answer = str(answer_text or "").strip()
        if not answer:
            return
        record = FinalChainRecord(
            target_question=str(target_question or ""),
            slot_key=str(slot_key or canonical_slot_key(target_question, answer_type, slot_role)),
            slot_role=str(slot_role or "generic"),
            answer_text=answer,
            answer_type=answer_type,
            evidence_ids=list(evidence_ids or []),
            support_score=float(support_score or 0.0),
            derived_from_state=derived_from_state,
            depends_on=list(depends_on or []),
            source=str(source or "unknown"),
            metadata=dict(metadata or {}),
        )
        key = (record.slot_key, normalize_text(record.answer_text), record.source, record.metadata.get("node_id", ""))
        for idx, existing in enumerate(self.records):
            existing_key = (
                existing.slot_key,
                normalize_text(existing.answer_text),
                existing.source,
                existing.metadata.get("node_id", ""),
            )
            if existing_key == key:
                if record.support_score > existing.support_score:
                    self.records[idx] = record
                return
        self.records.append(record)

    def group_by_slot(self) -> Dict[str, List[FinalChainRecord]]:
        grouped: Dict[str, List[FinalChainRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.slot_key].append(record)
        return dict(grouped)

    def best_for_slot(self, slot_key: str) -> Optional[FinalChainRecord]:
        records = self.group_by_slot().get(slot_key, [])
        if not records:
            return None
        return max(records, key=lambda r: (r.support_score, len(r.evidence_ids), r.metadata.get("node_value", 0.0)))

    def consolidate(self, question: str, goal_plan: Dict[str, Any]) -> Dict[str, Any]:
        grouped = self.group_by_slot()
        required_slots = [
            s for s in goal_plan.get("slots", [])
            if str(s.get("question", "")).strip() and str(s.get("slot_role", "")).strip().lower() != "bridge_entity"
        ]
        required_keys = [
            canonical_slot_key(
                str(s.get("question", "")),
                str(s.get("slot_type", "") or ""),
                str(s.get("slot_role", "") or ""),
            )
            for s in required_slots
        ]
        covered = sum(1 for key in required_keys if key in grouped)
        root_key = canonical_slot_key(question, slot_role="root_answer")
        root_record = self.best_for_slot(root_key)
        if root_record is None:
            root_like = [
                r for r in self.records
                if normalize_text(r.target_question) == normalize_text(question)
                or r.slot_role in {"root_answer", "target_attribute", "final_boolean"}
            ]
            if root_like:
                root_record = max(root_like, key=lambda r: (r.support_score, len(r.evidence_ids)))
        return {
            "required_slot_count": len(required_keys),
            "covered_slot_count": covered,
            "slot_coverage": covered / max(1, len(required_keys)) if required_keys else 1.0,
            "root_record": root_record.as_dict() if root_record else None,
            "record_count": len(self.records),
            "group_count": len(grouped),
        }


def score_final_chain_candidate_old(candidate: Dict[str, Any], question: str, goal_plan: Dict[str, Any], buffer: FinalChainBuffer) -> tuple[float, Dict[str, float]]:
    root_alignment = 1.0 if bool(candidate.get("root_aligned", False)) else 0.0
    if not root_alignment:
        target = str(candidate.get("target_question", "") or candidate.get("target_text", ""))
        root_alignment = lexical_jaccard(canonicalize_state_text(question), canonicalize_state_text(target))
    consolidation = buffer.consolidate(question, goal_plan)
    slot_coverage = max(float(candidate.get("coverage_ratio", 0.0) or 0.0), float(consolidation.get("slot_coverage", 0.0) or 0.0))
    evidence_support = max(
        float(candidate.get("support_score", 0.0) or 0.0),
        float(candidate.get("span_support", 0.0) or 0.0),
        float(candidate.get("node_value", 0.0) or 0.0),
    )
    dependency_satisfaction = float(candidate.get("dependency_satisfaction", 0.0) or 0.0)
    if candidate.get("depends_on"):
        dependency_satisfaction = max(dependency_satisfaction, 0.65)
    if int(candidate.get("composed_from_count", 0) or 0) >= 2:
        dependency_satisfaction = 1.0
    answer_type_match = float(candidate.get("type_score", 0.0) or 0.0)
    chain_len = max(1, int(candidate.get("composed_from_count", 0) or 0) + 1)
    chain_compactness = max(0.0, 1.0 - 0.12 * (chain_len - 1))
    parts = {
        "root_alignment": max(0.0, min(1.0, root_alignment)),
        "slot_coverage": max(0.0, min(1.0, slot_coverage)),
        "evidence_support": max(0.0, min(1.0, evidence_support)),
        "dependency_satisfaction": max(0.0, min(1.0, dependency_satisfaction)),
        "answer_type_match": max(0.0, min(1.0, answer_type_match)),
        "chain_compactness": max(0.0, min(1.0, chain_compactness)),
    }
    score = (
        0.25 * parts["root_alignment"]
        + 0.20 * parts["slot_coverage"]
        + 0.20 * parts["evidence_support"]
        + 0.15 * parts["dependency_satisfaction"]
        + 0.10 * parts["answer_type_match"]
        + 0.10 * parts["chain_compactness"]
    )
    return max(0.0, min(1.0, score)), parts


def passes_final_admission_preconditions(
    candidate: Dict[str, Any],
    question: str,
    goal_plan: Dict[str, Any],
    buffer: FinalChainBuffer,
) -> tuple[bool, Dict[str, Any]]:
    reasons: List[str] = []
    answer = str(candidate.get("answer", "") or "").strip()
    if not answer:
        reasons.append("precondition_failed_empty_answer")
    answer_norm = normalize_text(answer)
    question_norm = normalize_text(canonicalize_state_text(question))
    if answer_norm and (answer_norm == question_norm or len(answer_norm) > 20 and answer_norm in question_norm):
        reasons.append("precondition_failed_question_echo")
    if bool(candidate.get("title_only", False)):
        reasons.append("precondition_failed_title_only")
    if bool(candidate.get("is_bridge_entity", False)):
        reasons.append("precondition_failed_bridge_entity")
    if float(candidate.get("answer_type_match", candidate.get("type_score", 0.0)) or 0.0) <= 0.2:
        reasons.append("precondition_failed_answer_type")
    floor_reasons: List[str] = []
    min_last_hop = float(candidate.get("min_last_hop_support", 0.50) or 0.50)
    if float(candidate.get("last_hop_support", 0.0) or 0.0) < min_last_hop:
        reasons.append("precondition_failed_last_hop_support")
        floor_reasons.append("last_hop_support_below_active_floor")
    min_dep = float(candidate.get("min_dependency_satisfaction", 0.40) or 0.40)
    if float(candidate.get("dependency_satisfaction", 0.0) or 0.0) < min_dep:
        reasons.append("precondition_failed_dependency")
        floor_reasons.append("dependency_satisfaction_below_active_floor")
    min_root = float(candidate.get("min_root_alignment", 0.55) or 0.55)
    if float(candidate.get("root_alignment", 0.0) or 0.0) < min_root:
        reasons.append("precondition_failed_root_alignment")
        floor_reasons.append("root_alignment_below_active_floor")
    floor_check_passed = not floor_reasons
    return not reasons, {
        "score_admission_precondition_passed": not reasons,
        "score_admission_precondition_fail_reasons": list(dict.fromkeys(reasons)),
        "inferred_hop_count": int(float(candidate.get("inferred_hop_count", candidate.get("hop_count", 1)) or 1)),
        "is_longhop": bool(candidate.get("is_longhop", False)),
        "active_dependency_floor": min_dep,
        "active_last_hop_floor": min_last_hop,
        "floor_check_passed": floor_check_passed,
        "floor_check_fail_reasons": list(dict.fromkeys(floor_reasons)),
    }


def score_final_chain_candidate(candidate: Dict[str, Any], question: str, goal_plan: Dict[str, Any], buffer: FinalChainBuffer) -> tuple[float, Dict[str, float]]:
    root_alignment = float(candidate.get("root_alignment", 0.0) or 0.0)
    if root_alignment <= 0.0:
        root_alignment = 1.0 if bool(candidate.get("root_aligned", False)) else 0.0
    if not root_alignment:
        target = str(candidate.get("target_question", "") or candidate.get("target_text", ""))
        root_alignment = lexical_jaccard(canonicalize_state_text(question), canonicalize_state_text(target))
    consolidation = buffer.consolidate(question, goal_plan)
    slot_coverage = max(float(candidate.get("coverage_ratio", 0.0) or 0.0), float(consolidation.get("slot_coverage", 0.0) or 0.0))
    evidence_support = max(
        float(candidate.get("support_score", 0.0) or 0.0),
        float(candidate.get("span_support", 0.0) or 0.0),
        float(candidate.get("node_value", 0.0) or 0.0),
    )
    dependency_satisfaction = float(candidate.get("dependency_satisfaction", 0.0) or 0.0)
    last_hop_support = float(candidate.get("last_hop_support", 0.0) or 0.0)
    answer_type_match = float(candidate.get("answer_type_match", candidate.get("type_score", 0.0)) or 0.0)
    chain_len = max(1, int(candidate.get("composed_from_count", 0) or 0) + 1)
    chain_compactness = max(0.0, 1.0 - 0.12 * (chain_len - 1))
    parts = {
        "root_alignment": max(0.0, min(1.0, root_alignment)),
        "dependency_satisfaction": max(0.0, min(1.0, dependency_satisfaction)),
        "last_hop_support": max(0.0, min(1.0, last_hop_support)),
        "answer_type_match": max(0.0, min(1.0, answer_type_match)),
        "evidence_support": max(0.0, min(1.0, evidence_support)),
        "chain_compactness": max(0.0, min(1.0, chain_compactness)),
        "slot_coverage": max(0.0, min(1.0, slot_coverage)),
    }
    score = (
        0.30 * parts["root_alignment"]
        + 0.25 * parts["dependency_satisfaction"]
        + 0.25 * parts["last_hop_support"]
        + 0.10 * parts["answer_type_match"]
        + 0.05 * parts["evidence_support"]
        + 0.05 * parts["chain_compactness"]
    )
    return max(0.0, min(1.0, score)), parts
