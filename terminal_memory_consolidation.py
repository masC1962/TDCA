from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from final_chain_buffer import FinalChainBuffer, FinalChainRecord
from terminal_chain_closure import evaluate_terminal_chain_closure
from utils import canonicalize_state_text, lexical_jaccard, normalize_text


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return max(lo, min(hi, out))


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _metadata(node: Any) -> Dict[str, Any]:
    meta = getattr(node, "metadata", {}) if node is not None else {}
    return meta if isinstance(meta, dict) else {}


def _node_answer(node: Any) -> str:
    meta = _metadata(node)
    for key in ["answer_text", "answer", "candidate_answer"]:
        text = str(meta.get(key, "") or "").strip()
        if text:
            return text
    return ""


def _target_text(node: Any, fallback: str = "") -> str:
    meta = _metadata(node)
    return str(meta.get("target_question", "") or getattr(node, "content", "") or fallback)


@dataclass
class TerminalMemoryUnit:
    unit_id: str
    answer: str
    source: str
    node_id: str = ""
    target_question: str = ""
    slot_role: str = "generic"
    support_score: float = 0.0
    root_contribution: float = 0.0
    dependency_completeness: float = 0.0
    terminal_confidence: float = 0.0
    depends_on: List[str] = field(default_factory=list)
    composed_from: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "answer": self.answer,
            "source": self.source,
            "node_id": self.node_id,
            "target_question": self.target_question,
            "slot_role": self.slot_role,
            "support_score": self.support_score,
            "root_contribution": self.root_contribution,
            "dependency_completeness": self.dependency_completeness,
            "terminal_confidence": self.terminal_confidence,
            "depends_on": list(self.depends_on),
            "composed_from": list(self.composed_from),
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class ConsolidatedTerminalMemory:
    terminal_id: str
    answer: str
    consolidated_from: List[str]
    source_distribution: Dict[str, int]
    dependency_coverage: float
    root_support: float
    last_hop_support: float
    terminal_confidence: float
    root_contribution: float
    dependency_completeness: float
    evidence_ids: List[str]
    depends_on: List[str]
    composed_from: List[str]
    representative_node_id: str = ""
    target_question: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "terminal_id": self.terminal_id,
            "answer": self.answer,
            "consolidated_from": list(self.consolidated_from),
            "source_distribution": dict(self.source_distribution),
            "dependency_coverage": self.dependency_coverage,
            "root_support": self.root_support,
            "last_hop_support": self.last_hop_support,
            "terminal_confidence": self.terminal_confidence,
            "root_contribution": self.root_contribution,
            "dependency_completeness": self.dependency_completeness,
            "evidence_ids": list(self.evidence_ids),
            "depends_on": list(self.depends_on),
            "composed_from": list(self.composed_from),
            "representative_node_id": self.representative_node_id,
            "target_question": self.target_question,
            "metadata": dict(self.metadata),
        }

    def as_tcc_candidate(self, question: str, graph: Any = None) -> Dict[str, Any]:
        memory = graph.get_node(self.representative_node_id) if graph is not None and self.representative_node_id and graph.has_node(self.representative_node_id) else None
        return {
            "answer": self.answer,
            "source": "terminal_memory",
            "node_id": self.representative_node_id,
            "memory": memory,
            "target_question": self.target_question or question,
            "target_text": self.target_question or question,
            "root_aligned": self.root_contribution >= 0.70,
            "root_alignment": self.root_contribution,
            "coverage_ratio": self.dependency_coverage,
            "support_score": self.terminal_confidence,
            "span_support": self.metadata.get("span_support", 0.0),
            "node_value": self.terminal_confidence,
            "dependency_satisfaction": self.dependency_completeness,
            "last_hop_support": self.last_hop_support,
            "last_hop_verification": {
                "last_hop_support": self.last_hop_support,
                "last_hop_reason": "terminal_memory_consolidation",
            },
            "composed_from_count": len(self.composed_from),
            "depends_on": list(self.depends_on),
            "supporting_memory_ids": list(dict.fromkeys([*self.depends_on, *self.composed_from])),
            "title_only": False,
            "root_goal_satisfied": True,
            "base_score": self.terminal_confidence,
            "original_score": self.terminal_confidence,
            "terminal_memory": self.as_dict(),
        }


@dataclass
class MemoryRepairGoal:
    goal_id: str
    repair_type: str
    target_answer: str = ""
    target_question: str = ""
    missing_dependency: str = ""
    reason: str = ""
    priority: float = 0.0
    source_terminal_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "repair_type": self.repair_type,
            "target_answer": self.target_answer,
            "target_question": self.target_question,
            "missing_dependency": self.missing_dependency,
            "reason": self.reason,
            "priority": self.priority,
            "source_terminal_id": self.source_terminal_id,
            "metadata": dict(self.metadata),
        }


def _unit_from_memory(question: str, node: Any, source: str, graph: Any = None) -> Optional[TerminalMemoryUnit]:
    answer = _node_answer(node)
    if not answer:
        return None
    meta = _metadata(node)
    node_id = str(getattr(node, "node_id", "") or "")
    target = _target_text(node, question)
    target_norm = str(meta.get("target_question_norm", "") or normalize_text(canonicalize_state_text(target)))
    root_norm = normalize_text(canonicalize_state_text(question))
    depends_on = [str(x) for x in _as_list(meta.get("depends_on")) if str(x)]
    composed_from = [str(x) for x in _as_list(meta.get("composed_from")) if str(x)]
    support = max(
        _clamp(meta.get("support_score")),
        _clamp(getattr(node, "value", 0.0)),
        _clamp(getattr(node, "temperature", 0.0)),
        _clamp(meta.get("final_chain_score")),
    )
    root_contribution = 1.0 if target_norm == root_norm else lexical_jaccard(question, target)
    if bool(meta.get("path_terminal", False)) or str(meta.get("composition_kind", "") or "").startswith("score_based"):
        root_contribution = max(root_contribution, 0.80)
    dependency_completeness = 0.35
    if target_norm == root_norm:
        dependency_completeness = max(dependency_completeness, 0.55)
    if depends_on:
        dependency_completeness = max(dependency_completeness, 0.70)
    if len(composed_from) >= 2:
        dependency_completeness = max(dependency_completeness, 0.90)
    terminal_confidence = max(
        support,
        _clamp(meta.get("terminal_chain_closure_score")),
        0.70 if bool(meta.get("terminal", False) or meta.get("path_terminal", False)) else 0.0,
    )
    return TerminalMemoryUnit(
        unit_id=f"mem:{node_id}",
        answer=answer,
        source=source,
        node_id=node_id,
        target_question=target,
        slot_role=str(meta.get("slot_role", "") or "generic"),
        support_score=support,
        root_contribution=_clamp(root_contribution),
        dependency_completeness=_clamp(dependency_completeness),
        terminal_confidence=_clamp(terminal_confidence),
        depends_on=depends_on,
        composed_from=composed_from,
        evidence_ids=[str(x) for x in _as_list(meta.get("evidence_ids")) if str(x)],
        metadata={
            "target_question_norm": target_norm,
            "composition_kind": str(meta.get("composition_kind", "") or ""),
            "terminal": bool(meta.get("terminal", False)),
            "path_terminal": bool(meta.get("path_terminal", False)),
        },
    )


def _unit_from_record(question: str, record: FinalChainRecord) -> Optional[TerminalMemoryUnit]:
    answer = str(record.answer_text or "").strip()
    if not answer:
        return None
    meta = dict(record.metadata or {})
    target = str(record.target_question or question)
    root_norm = normalize_text(canonicalize_state_text(question))
    target_norm = str(meta.get("target_question_norm", "") or normalize_text(canonicalize_state_text(target)))
    root_contribution = 1.0 if target_norm == root_norm else lexical_jaccard(question, target)
    depends_on = [str(x) for x in _as_list(record.depends_on) if str(x)]
    composed_from = [str(x) for x in _as_list(meta.get("composed_from")) if str(x)]
    dependency_completeness = 0.55 if target_norm == root_norm else 0.35
    if depends_on:
        dependency_completeness = max(dependency_completeness, 0.70)
    if len(composed_from) >= 2:
        dependency_completeness = max(dependency_completeness, 0.90)
    return TerminalMemoryUnit(
        unit_id=f"buffer:{record.source}:{record.slot_key}:{normalize_text(answer)}",
        answer=answer,
        source=f"buffer:{record.source}",
        node_id=str(meta.get("node_id", "") or record.derived_from_state or ""),
        target_question=target,
        slot_role=str(record.slot_role or "generic"),
        support_score=_clamp(record.support_score),
        root_contribution=_clamp(root_contribution),
        dependency_completeness=_clamp(dependency_completeness),
        terminal_confidence=max(_clamp(record.support_score), 0.60 if bool(meta.get("terminal", False)) else 0.0),
        depends_on=depends_on,
        composed_from=composed_from,
        evidence_ids=[str(x) for x in _as_list(record.evidence_ids) if str(x)],
        metadata=meta,
    )


def consolidate_terminal_memories(
    *,
    question: str,
    goal_plan: Dict[str, Any],
    graph: Any,
    final_chain_buffer: FinalChainBuffer,
    current_run_memory_node_ids: Iterable[str],
    root_memory: Any = None,
    composed_memory: Any = None,
    anytime_answer: str = "",
    anytime_score: float = 0.0,
) -> Dict[str, Any]:
    units: List[TerminalMemoryUnit] = []
    seen_units: set[str] = set()
    current_ids = {str(x) for x in current_run_memory_node_ids if str(x)}

    def add(unit: Optional[TerminalMemoryUnit]) -> None:
        if unit is None or not normalize_text(unit.answer):
            return
        if unit.unit_id in seen_units:
            return
        seen_units.add(unit.unit_id)
        units.append(unit)

    add(_unit_from_memory(question, root_memory, "root_memory", graph=graph))
    add(_unit_from_memory(question, composed_memory, "composed_memory", graph=graph))

    if graph is not None and hasattr(graph, "memory_nodes"):
        for mem in graph.memory_nodes():
            node_id = str(getattr(mem, "node_id", "") or "")
            if current_ids and node_id not in current_ids:
                continue
            source = "graph_memory"
            meta = _metadata(mem)
            if str(meta.get("target_question_norm", "") or "") == normalize_text(canonicalize_state_text(question)):
                source = "root_level_memory"
            if meta.get("composed_from"):
                source = "dependency_memory"
            add(_unit_from_memory(question, mem, source, graph=graph))

    for record in final_chain_buffer.records:
        add(_unit_from_record(question, record))

    if str(anytime_answer or "").strip():
        units.append(TerminalMemoryUnit(
            unit_id=f"anytime:{normalize_text(anytime_answer)}",
            answer=str(anytime_answer).strip(),
            source="anytime",
            target_question=question,
            support_score=_clamp(anytime_score),
            root_contribution=0.55,
            dependency_completeness=0.25,
            terminal_confidence=_clamp(anytime_score),
        ))

    grouped: Dict[str, List[TerminalMemoryUnit]] = defaultdict(list)
    for unit in units:
        grouped[normalize_text(unit.answer)].append(unit)

    terminals: List[ConsolidatedTerminalMemory] = []
    required_slots = [s for s in goal_plan.get("slots", []) or [] if isinstance(s, dict)]
    required_count = len([s for s in required_slots if str(s.get("slot_role", "")).strip().lower() != "bridge_entity"])
    for idx, (answer_norm, group) in enumerate(grouped.items()):
        if not answer_norm:
            continue
        source_counts: Dict[str, int] = defaultdict(int)
        for unit in group:
            source_counts[unit.source] += 1
        evidence_ids = list(dict.fromkeys([eid for unit in group for eid in unit.evidence_ids]))
        depends_on = list(dict.fromkeys([dep for unit in group for dep in unit.depends_on]))
        composed_from = list(dict.fromkeys([dep for unit in group for dep in unit.composed_from]))
        covered_slots = {unit.slot_role for unit in group if unit.slot_role and unit.slot_role != "generic"}
        dependency_coverage = len(covered_slots) / max(1, required_count) if required_count else max(unit.dependency_completeness for unit in group)
        dependency_coverage = max(dependency_coverage, max(unit.dependency_completeness for unit in group))
        root_support = max(unit.root_contribution for unit in group)
        terminal_confidence = max(unit.terminal_confidence for unit in group)
        if len(group) >= 2:
            terminal_confidence = min(1.0, terminal_confidence + 0.04 * min(3, len(group) - 1))
        last_hop_support = max(_clamp(unit.metadata.get("last_hop_support")) for unit in group)
        if last_hop_support <= 0:
            last_hop_support = max(0.40, min(0.75, terminal_confidence))
        representative = max(group, key=lambda u: (u.root_contribution, u.dependency_completeness, u.terminal_confidence, bool(u.node_id)))
        terminals.append(ConsolidatedTerminalMemory(
            terminal_id=f"tmc_{idx}_{answer_norm[:24].replace(' ', '_')}",
            answer=representative.answer,
            consolidated_from=[unit.unit_id for unit in group],
            source_distribution=dict(source_counts),
            dependency_coverage=_clamp(dependency_coverage),
            root_support=_clamp(root_support),
            last_hop_support=_clamp(last_hop_support),
            terminal_confidence=_clamp(terminal_confidence),
            root_contribution=_clamp(root_support),
            dependency_completeness=_clamp(max(unit.dependency_completeness for unit in group)),
            evidence_ids=evidence_ids,
            depends_on=depends_on,
            composed_from=composed_from,
            representative_node_id=representative.node_id,
            target_question=representative.target_question or question,
            metadata={
                "unit_count": len(group),
                "required_slot_count": required_count,
                "covered_slot_roles": sorted(covered_slots),
            },
        ))

    terminals.sort(
        key=lambda terminal: (
            terminal.root_support,
            terminal.dependency_completeness,
            terminal.terminal_confidence,
            len(terminal.consolidated_from),
        ),
        reverse=True,
    )
    return {
        "question": question,
        "unit_count": len(units),
        "terminal_count": len(terminals),
        "units": [unit.as_dict() for unit in units],
        "terminals": [terminal.as_dict() for terminal in terminals],
    }


def diagnose_terminal_feedback(
    *,
    terminal_memory_graph: Dict[str, Any],
    tcc_results: List[Dict[str, Any]],
    goal_plan: Dict[str, Any],
    max_goals: int = 3,
) -> List[Dict[str, Any]]:
    diagnoses: List[MemoryRepairGoal] = []
    terminals = terminal_memory_graph.get("terminals", []) if isinstance(terminal_memory_graph, dict) else []
    bridge_slots = [
        slot for slot in goal_plan.get("slots", []) or []
        if isinstance(slot, dict)
        and (
            str(slot.get("slot_role", "")).strip().lower() == "bridge_entity"
            or str(slot.get("kind", "")).strip().lower() == "bridge"
        )
    ]

    def add(repair_type: str, reason: str, terminal: Dict[str, Any], priority: float, missing_dependency: str = "") -> None:
        diagnoses.append(MemoryRepairGoal(
            goal_id=f"repair_{len(diagnoses)}_{repair_type}",
            repair_type=repair_type,
            target_answer=str(terminal.get("answer", "") or ""),
            target_question=str(terminal.get("target_question", "") or ""),
            missing_dependency=missing_dependency,
            reason=reason,
            priority=_clamp(priority),
            source_terminal_id=str(terminal.get("terminal_id", "") or ""),
            metadata={
                "dependency_coverage": terminal.get("dependency_coverage", 0.0),
                "root_support": terminal.get("root_support", 0.0),
                "last_hop_support": terminal.get("last_hop_support", 0.0),
            },
        ))

    for result in tcc_results:
        terminal = dict(result.get("terminal_memory", {}) or {})
        info = dict(result.get("closure_info", {}) or {})
        fail_reasons = {str(r) for r in result.get("fail_reasons", []) or []}
        if "tcc_dependency_not_closed" in fail_reasons or float(info.get("dependency_closure", 0.0) or 0.0) < 0.55:
            missing = ""
            if bridge_slots:
                missing = str(bridge_slots[0].get("question", "") or bridge_slots[0].get("name", "") or "")
            add("missing_dependency", "dependency_closure_below_floor", terminal, 0.92, missing_dependency=missing)
        if "tcc_last_hop_not_entailed" in fail_reasons or float(info.get("last_hop_entailment", 0.0) or 0.0) < 0.55:
            add("missing_last_hop", "last_hop_support_below_floor", terminal, 0.86)
        if "tcc_root_inconsistent" in fail_reasons or float(info.get("root_consistency", 0.0) or 0.0) < 0.55:
            add("weak_root_alignment", "root_consistency_below_floor", terminal, 0.80)
        chain = info.get("candidate_chain", []) or []
        if len(chain) <= 1 and bool(goal_plan.get("requires_structured_reasoning")):
            add("singleton_chain", "terminal_chain_has_no_dependency_context", terminal, 0.72)

    if len(terminals) > 1:
        top = terminals[0]
        add("multiple_terminal_candidates", "multiple_terminal_memories_compete", top, 0.58)

    dedup: Dict[tuple[str, str, str], MemoryRepairGoal] = {}
    for goal in diagnoses:
        key = (goal.repair_type, normalize_text(goal.target_answer), normalize_text(goal.missing_dependency))
        existing = dedup.get(key)
        if existing is None or goal.priority > existing.priority:
            dedup[key] = goal
    return [goal.as_dict() for goal in sorted(dedup.values(), key=lambda g: g.priority, reverse=True)[:max_goals]]


def evaluate_terminal_memories_with_tcc(
    *,
    terminal_memory_graph: Dict[str, Any],
    question: str,
    goal_plan: Dict[str, Any],
    final_chain_buffer: FinalChainBuffer,
    graph: Any,
    dimension_floors: Optional[Dict[str, float]] = None,
    score_threshold: float = 0.70,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for terminal in terminal_memory_graph.get("terminals", []) if isinstance(terminal_memory_graph, dict) else []:
        terminal_obj = ConsolidatedTerminalMemory(
            terminal_id=str(terminal.get("terminal_id", "") or ""),
            answer=str(terminal.get("answer", "") or ""),
            consolidated_from=[str(x) for x in terminal.get("consolidated_from", []) or []],
            source_distribution=dict(terminal.get("source_distribution", {}) or {}),
            dependency_coverage=_clamp(terminal.get("dependency_coverage")),
            root_support=_clamp(terminal.get("root_support")),
            last_hop_support=_clamp(terminal.get("last_hop_support")),
            terminal_confidence=_clamp(terminal.get("terminal_confidence")),
            root_contribution=_clamp(terminal.get("root_contribution")),
            dependency_completeness=_clamp(terminal.get("dependency_completeness")),
            evidence_ids=[str(x) for x in terminal.get("evidence_ids", []) or []],
            depends_on=[str(x) for x in terminal.get("depends_on", []) or []],
            composed_from=[str(x) for x in terminal.get("composed_from", []) or []],
            representative_node_id=str(terminal.get("representative_node_id", "") or ""),
            target_question=str(terminal.get("target_question", "") or question),
            metadata=dict(terminal.get("metadata", {}) or {}),
        )
        candidate = terminal_obj.as_tcc_candidate(question, graph=graph)
        score, info = evaluate_terminal_chain_closure(
            candidate,
            question,
            goal_plan,
            final_chain_buffer,
            graph=graph,
            dimension_floors=dimension_floors,
        )
        fail_reasons: List[str] = []
        if score < score_threshold:
            fail_reasons.append("tcc_score_below_threshold")
        for key, floor in (dimension_floors or {}).items():
            if float(info.get(key, 0.0) or 0.0) < float(floor):
                fail_reasons.append(f"tcc_{key}_below_floor")
        for reason in info.get("closure_fail_reasons", []) or []:
            fail_reasons.append(str(reason))
        results.append({
            "terminal_id": terminal_obj.terminal_id,
            "answer": terminal_obj.answer,
            "tcc_score": score,
            "tcc_passed": not fail_reasons,
            "fail_reasons": list(dict.fromkeys(fail_reasons)),
            "closure_info": info,
            "terminal_memory": terminal_obj.as_dict(),
            "candidate": {
                key: value
                for key, value in candidate.items()
                if key != "memory"
            },
        })
    results.sort(
        key=lambda item: (
            bool(item.get("tcc_passed", False)),
            float(item.get("tcc_score", 0.0) or 0.0),
            float((item.get("terminal_memory", {}) or {}).get("terminal_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return results
