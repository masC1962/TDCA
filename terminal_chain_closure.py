from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from final_chain_buffer import FinalChainBuffer
from utils import canonicalize_state_text, lexical_jaccard, normalize_text


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    if math.isnan(out):
        out = 0.0
    return max(lo, min(hi, out))


def _node_id(obj: Any) -> str:
    return str(getattr(obj, "node_id", "") or "")


def _metadata(obj: Any) -> Dict[str, Any]:
    meta = getattr(obj, "metadata", {}) if obj is not None else {}
    return meta if isinstance(meta, dict) else {}


def _answer(obj: Any) -> str:
    meta = _metadata(obj)
    for key in ["answer_text", "answer", "candidate_answer"]:
        text = str(meta.get(key, "") or "").strip()
        if text:
            return text
    content = str(getattr(obj, "content", "") or "")
    if "Answer:" in content:
        return content.split("Answer:", 1)[1].strip().splitlines()[0].strip()
    return ""


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _simple_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(canonicalize_state_text(text)).split() if len(t) > 2}


def _chain_nodes(candidate: Dict[str, Any], graph: Any = None) -> List[Dict[str, Any]]:
    mem = candidate.get("memory")
    meta = _metadata(mem)
    seen: set[str] = set()
    chain: List[Dict[str, Any]] = []

    def add(node_id: str, role: str, text: str = "", answer: str = "") -> None:
        node_id = str(node_id or "").strip()
        if not node_id or node_id in seen:
            return
        seen.add(node_id)
        chain.append({"node_id": node_id, "role": role, "text": text, "answer": answer})

    if mem is not None:
        add(_node_id(mem), str(meta.get("slot_role", "candidate") or "candidate"), str(getattr(mem, "content", "") or ""), _answer(mem))
    for role, key in [("depends_on", "depends_on"), ("composed_from", "composed_from")]:
        for dep_id in _as_list(meta.get(key)) + _as_list(candidate.get(key)):
            dep_id = str(dep_id or "").strip()
            if not dep_id:
                continue
            dep_text = ""
            dep_answer = ""
            if graph is not None and hasattr(graph, "has_node") and graph.has_node(dep_id):
                dep = graph.get_node(dep_id)
                dep_text = str(getattr(dep, "content", "") or "")
                dep_answer = _answer(dep)
            add(dep_id, role, dep_text, dep_answer)
    return chain


def _bridge_slot_totals(goal_plan: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    slots = goal_plan.get("slots") or []
    if not isinstance(slots, list):
        return 0, []
    bridge_slots = [
        s for s in slots
        if str(s.get("slot_role", "")).strip().lower() == "bridge_entity"
        or str(s.get("kind", "")).strip().lower() == "bridge"
    ]
    return len(bridge_slots), bridge_slots


def _bridge_slots_covered(candidate: Dict[str, Any], buffer: FinalChainBuffer, bridge_slots: Iterable[Dict[str, Any]]) -> int:
    answer_norms = {
        normalize_text(str(record.answer_text or ""))
        for record in buffer.records
        if str(record.slot_role or "").strip().lower() == "bridge_entity"
    }
    depends = {str(x) for x in _as_list(candidate.get("depends_on")) if str(x)}
    covered = 0
    for slot in bridge_slots:
        slot_q = normalize_text(str(slot.get("question", "")))
        slot_name = normalize_text(str(slot.get("name", "")))
        if any(ans and ans in slot_q for ans in answer_norms):
            covered += 1
        elif any(slot_name and slot_name in normalize_text(dep) for dep in depends):
            covered += 1
    return covered


def _root_anchor_found(candidate: Dict[str, Any], question: str) -> bool:
    evidence_items = candidate.get("evidence_items") or []
    answer_norm = normalize_text(str(candidate.get("answer", "") or ""))
    anchors = _simple_tokens(question)
    target_tokens = _simple_tokens(str(candidate.get("target_question", "") or candidate.get("target_text", "")))
    if anchors & target_tokens:
        return True
    for item in evidence_items if isinstance(evidence_items, list) else []:
        text_norm = normalize_text(str(getattr(item, "text", "") or ""))
        if answer_norm and answer_norm in text_norm and any(anchor in text_norm for anchor in anchors):
            return True
    return False


def compute_closure_score(closure_info: Dict[str, Any]) -> float:
    core_keys = [
        "path_completeness",
        "dependency_closure",
        "last_hop_entailment",
        "terminality",
        "root_consistency",
    ]
    core_values = [_clamp(closure_info.get(key)) for key in core_keys]
    product = 1.0
    for value in core_values:
        product *= max(value, 1e-6)
    core = product ** (1.0 / len(core_values))
    evidence = _clamp(closure_info.get("evidence_grounding"))
    return _clamp(0.85 * core + 0.15 * evidence)


def evaluate_terminal_chain_closure(
    candidate: Dict[str, Any],
    question: str,
    goal_plan: Dict[str, Any],
    buffer: FinalChainBuffer,
    graph: Any = None,
    dimension_floors: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    chain = _chain_nodes(candidate, graph=graph)
    chain_ids = {str(item.get("node_id", "")) for item in chain}
    bridge_total, bridge_slots = _bridge_slot_totals(goal_plan)
    bridge_covered = _bridge_slots_covered(candidate, buffer, bridge_slots)
    is_multi = bool(goal_plan.get("requires_structured_reasoning")) or bridge_total > 0 or int(candidate.get("inferred_hop_count", 1) or 1) >= 3

    root_aligned = bool(candidate.get("root_aligned", False))
    root_alignment = _clamp(candidate.get("root_alignment"))
    composed_count = int(candidate.get("composed_from_count", 0) or 0)
    depends = [str(x) for x in _as_list(candidate.get("depends_on")) if str(x)]
    has_dependency = bool(depends) or composed_count >= 2
    root_anchor = _root_anchor_found(candidate, question)

    path_completeness = 0.25
    if chain:
        path_completeness = 0.45
    if root_aligned or root_alignment >= 0.70:
        path_completeness = max(path_completeness, 0.70)
    if has_dependency:
        path_completeness = max(path_completeness, 0.78)
    if is_multi and (has_dependency or bridge_covered > 0):
        path_completeness = max(path_completeness, 0.82)
    if not is_multi and (root_aligned or root_alignment >= 0.55):
        path_completeness = max(path_completeness, 0.78)

    dependency_closure = _clamp(candidate.get("dependency_satisfaction"))
    if has_dependency:
        dependency_closure = max(dependency_closure, 0.70)
    if bridge_total > 0:
        dependency_closure = max(dependency_closure, bridge_covered / max(1, bridge_total))
    if is_multi and not has_dependency and bridge_covered <= 0:
        dependency_closure = min(dependency_closure, 0.35)

    last_hop_info = candidate.get("last_hop_verification")
    if not isinstance(last_hop_info, dict):
        last_hop_info = {}
    last_hop_entailment = _clamp(last_hop_info.get("last_hop_support", candidate.get("last_hop_support")))
    if root_anchor:
        last_hop_entailment = max(last_hop_entailment, 0.58)
    if str(last_hop_info.get("last_hop_reason", "")) in {"root_aligned_memory", "no_last_hop_support"} and not root_anchor and not has_dependency:
        last_hop_entailment = min(last_hop_entailment, 0.45)

    consumed = bool(candidate.get("is_bridge_entity", False) or candidate.get("candidate_is_consumed_as_bridge", False))
    mem = candidate.get("memory")
    meta = _metadata(mem)
    answer_norm = normalize_text(str(candidate.get("answer", "") or ""))
    for record in buffer.records:
        if normalize_text(record.answer_text) == answer_norm and str(record.slot_role).lower() == "bridge_entity":
            consumed = True
    candidate_leaf = not consumed
    terminality = 0.82 if candidate_leaf else 0.25
    if bool(candidate.get("title_only", False)):
        terminality = min(terminality, 0.45)
    if str(meta.get("slot_role", "")).lower() == "bridge_entity":
        terminality = min(terminality, 0.25)

    target_text = str(candidate.get("target_question", "") or candidate.get("target_text", ""))
    root_consistency = max(root_alignment, lexical_jaccard(canonicalize_state_text(question), canonicalize_state_text(target_text)))
    if root_aligned:
        root_consistency = max(root_consistency, 0.82)
    if is_multi and not root_aligned and not has_dependency:
        root_consistency = min(root_consistency, 0.45)
    if has_dependency and root_anchor:
        root_consistency = max(root_consistency, 0.72)

    evidence_grounding = max(
        _clamp(candidate.get("support_score")),
        _clamp(candidate.get("span_support")),
        _clamp(candidate.get("node_value")),
    )

    floors = dict(dimension_floors or {})
    path_floor = _clamp(floors.get("path_completeness", 0.45))
    dep_floor = _clamp(floors.get("dependency_closure", 0.45))
    last_floor = _clamp(floors.get("last_hop_entailment", 0.50))
    terminal_floor = _clamp(floors.get("terminality", 0.60))
    root_floor = _clamp(floors.get("root_consistency", 0.55))

    fail_reasons: List[str] = []
    if path_completeness < path_floor:
        fail_reasons.append("tcc_path_incomplete")
    if dependency_closure < dep_floor:
        fail_reasons.append("tcc_dependency_not_closed")
    if last_hop_entailment < last_floor:
        fail_reasons.append("tcc_last_hop_not_entailed")
    if terminality < terminal_floor:
        fail_reasons.append("tcc_candidate_not_terminal")
    if root_consistency < root_floor:
        fail_reasons.append("tcc_root_inconsistent")

    info = {
        "path_completeness": _clamp(path_completeness),
        "dependency_closure": _clamp(dependency_closure),
        "last_hop_entailment": _clamp(last_hop_entailment),
        "terminality": _clamp(terminality),
        "root_consistency": _clamp(root_consistency),
        "evidence_grounding": _clamp(evidence_grounding),
        "closure_fail_reasons": list(dict.fromkeys(fail_reasons)),
        "candidate_chain": chain,
        "root_anchor_found": bool(root_anchor),
        "bridge_slots_covered": int(bridge_covered),
        "bridge_slots_total": int(bridge_total),
        "candidate_is_consumed_as_bridge": bool(consumed),
        "candidate_is_terminal_leaf": bool(candidate_leaf),
        "chain_node_ids": list(chain_ids),
        "tcc_dimension_floors": {
            "path_completeness": path_floor,
            "dependency_closure": dep_floor,
            "last_hop_entailment": last_floor,
            "terminality": terminal_floor,
            "root_consistency": root_floor,
        },
    }
    return compute_closure_score(info), info
