from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from ..models import QAExample
from ..utils import normalize_text
from ..dynamic.graph import CandidateStatus, ClaimNode
from .graph import DynamicReasoningHypergraphV2
from .proof import audit_graph_proof


def dynamic_v2_metrics(
    examples: list[QAExample], reasoning_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reasoning_rows:
        if row.get("qid"):
            rows_by_qid[str(row["qid"])].append(row)
    examples_by_qid = {example.qid: example for example in examples}
    per_example = []
    graph_rows = []
    for qid, traces in sorted(rows_by_qid.items()):
        snapshots = [row["graph_snapshot"] for row in traces if row.get("graph_snapshot")]
        if not snapshots:
            continue
        graph = snapshots[-1]
        graph_rows.append({"qid": qid, "graph": graph})
        example = examples_by_qid.get(qid)
        nodes = graph.get("nodes", {})
        claims = {key: value for key, value in nodes.items() if value.get("kind") == "claim"}
        answers = {key: value for key, value in nodes.items() if value.get("kind") == "answer"}
        semantics = graph.get("claim_semantics", {})
        joined = {
            node_id: value for node_id, value in semantics.items()
            if int(value.get("join_depth", 0)) > 0
        }
        hyperedges = graph.get("hyperedges", {})
        auditable_joins = [
            node_id for node_id in joined
            if any(
                edge.get("target_node") == node_id and len(edge.get("source_node_set", [])) >= 2
                for edge in hyperedges.values()
            )
        ]
        gold = {
            normalize_text(value) for value in (example.answers if example else []) if normalize_text(value)
        }
        gold_candidates = {
            node_id for node_id, value in claims.items()
            if normalize_text(str(value.get("value", ""))) in gold
        }
        active = {
            node_id for node_id, value in claims.items()
            if value.get("status") not in {"invalid", "archived"}
        }
        allocations = graph.get("allocation_history", [])
        join_attempts = graph.get("join_attempt_history", [])
        outcomes = graph.get("operation_outcome_history", [])
        budget_shapes = {
            tuple(sorted((row.get("requested_budget") or {}).items())) for row in allocations
        }
        fidelity_levels = {
            str(row.get("fidelity_level", "medium")) for row in allocations
        }
        memory_messages = [
            message
            for snapshot in graph.get("diffusion_history", [])
            for message in snapshot.get("typed_messages", [])
            if message.get("edge_type") == "memory_query_activation"
        ]
        complete_evc = bool(allocations) and all(
            row.get("evc_components_raw") and row.get("evc_components_normalized")
            and row.get("requested_budget") and row.get("actual_cost")
            and row.get("predicted_evc") is not None
            for row in allocations
        )
        complete_outcome_feedback = bool(allocations) and len(outcomes) == len(allocations) and all(
            row.get("feedback_applied")
            and row.get("pre_state_summary") and row.get("post_state_summary")
            and row.get("state_delta")
            and row.get("actual_utility_components_raw")
            and row.get("actual_utility_components_normalized")
            and row.get("actual_utility") is not None
            for row in allocations
        )
        accepted_nary = [
            row for row in join_attempts
            if row.get("accepted") and len(
                row.get("proof_leaf_ids") or row.get("premise_ids", [])
            ) >= 3
        ]
        answer_support_closure = set()
        for answer in answers.values():
            answer_support_closure.update(answer.get("supporting_claims", []))
            queue = list(answer.get("supporting_claims", []))
            while queue:
                claim = claims.get(str(queue.pop()), {})
                for dependency_id in claim.get("dependency_claim_ids", []):
                    if dependency_id not in answer_support_closure:
                        answer_support_closure.add(dependency_id)
                        queue.append(dependency_id)
        downstream_nary = [
            row for row in accepted_nary
            if row.get("conclusion_node_id") in answer_support_closure
            or claims.get(str(row.get("conclusion_node_id")), {}).get("status") == "committed"
        ]
        revisions = [row for row in graph.get("supersession_history", []) if row.get("natural")]
        revision_labels = [_revision_label(row, claims, example) for row in revisions]
        known_labels = [value for value in revision_labels if value in {"correct", "wrong"}]
        accepted_answers = [value for value in answers.values() if value.get("status") == "accepted"]
        terminal_beliefs = graph.get("terminal_beliefs", {})
        invalid_edges = set(graph.get("invalidated_hyperedges", []))
        unsupported = [
            value for value in accepted_answers
            if not value.get("supporting_claims") or not value.get("supporting_evidence")
            or value.get("derivation_edge") not in hyperedges
            or value.get("derivation_edge") in invalid_edges
            or any(claim_id not in active for claim_id in value.get("supporting_claims", []))
            or value.get("node_id") not in terminal_beliefs
            or not terminal_beliefs.get(value.get("node_id"), {}).get("accepted")
            or terminal_beliefs.get(value.get("node_id"), {}).get("rejection_reasons")
            or not terminal_beliefs.get(value.get("node_id"), {}).get("sufficient_chain")
            or set(terminal_beliefs.get(value.get("node_id"), {}).get("supporting_claims", []))
            != set(value.get("supporting_claims", []))
            or set(terminal_beliefs.get(value.get("node_id"), {}).get("supporting_evidence", []))
            != set(value.get("supporting_evidence", []))
        ]
        complete_terminal_readout = all(
            value.get("node_id") in terminal_beliefs
            and terminal_beliefs[value.get("node_id")].get("accepted")
            and terminal_beliefs[value.get("node_id")].get("sufficient_chain")
            and not terminal_beliefs[value.get("node_id")].get("rejection_reasons")
            for value in accepted_answers
        )
        terminal_gap_trace = bool(allocations) and all(
            "terminal_gap" in (allocation.get("evc_components_raw") or {})
            and "terminal_proximity" in (allocation.get("evc_components_raw") or {})
            and "terminal_gap" in (allocation.get("pre_state_summary") or {})
            for allocation in allocations
        ) and len(outcomes) == len(allocations) and all(
            "terminal_gap_reduction" in (outcome.get("actual_utility_components_raw") or {})
            and "terminal_gap" in (outcome.get("post_state_summary") or {})
            for outcome in outcomes
        )
        terminations = graph.get("termination_history", [])
        proof_metrics = _best_graph_proof(graph)
        extraction_rows = [
            trace for trace in traces
            if trace.get("event") == "typed_extraction_diagnostic"
        ]
        extraction_fingerprints = [
            (
                str(trace.get("target_id", "")),
                str(trace.get("branch_id", "")),
                str(trace.get("extraction_state_fingerprint", "")),
                str(trace.get("focus_mode", "coverage")),
            )
            for trace in extraction_rows
            if trace.get("extraction_state_fingerprint")
        ]
        retrieval_gates = [
            trace for trace in traces if trace.get("event") == "region_retrieval_gate"
        ]
        row = {
            "qid": qid,
            "hop_count": None if example is None else example.hop_count,
            "claim_count": len(claims),
            "typed_claim_rate": len(semantics) / max(1, len(claims)),
            "join_count": len(joined),
            "auditable_join_count": len(auditable_joins),
            "auditable_three_or_four_hop_join_case": bool(
                example and (example.hop_count or 0) >= 3 and auditable_joins
            ),
            "candidate_presence": bool(gold_candidates),
            "candidate_survival": bool(gold_candidates & active),
            "diffusion_count": len(graph.get("diffusion_history", [])),
            "typed_message_count": sum(
                len(value.get("typed_messages", [])) for value in graph.get("diffusion_history", [])
            ),
            "query_graph_present": bool(graph.get("query_graph")),
            "activated_passage_count": len(graph.get("activated_passages", {})),
            "activated_entity_count": len(graph.get("activated_entities", {})),
            "cross_layer_edge_count": len(graph.get("cross_layer_edges", [])),
            "memory_activation_message_count": len(memory_messages),
            "allocation_count": len(allocations),
            "selected_fidelity_level_count": len(fidelity_levels),
            "non_uniform_allocation": len(budget_shapes) > 1,
            "complete_evc_trace": complete_evc,
            "complete_outcome_feedback_trace": complete_outcome_feedback,
            "feedback_influenced_allocation": any(
                float((row.get("feedback_prior") or {}).get("observations", 0.0)) > 0.0
                for row in allocations
            ),
            "nary_join_attempt_count": sum(
                len(row.get("proof_leaf_ids") or row.get("premise_ids", [])) >= 3
                for row in join_attempts
            ),
            "nary_join_accepted_count": len(accepted_nary),
            "nary_join_downstream_used_count": len(downstream_nary),
            "natural_revision_count": len(revisions),
            "natural_revision_correct": revision_labels.count("correct"),
            "natural_revision_wrong": revision_labels.count("wrong"),
            "natural_revision_unknown": revision_labels.count("unknown"),
            "natural_revision_precision": (
                revision_labels.count("correct") / len(known_labels) if known_labels else 0.0
            ),
            "unsupported_answer_count": len(unsupported),
            "terminal_belief_count": len(terminal_beliefs),
            "complete_terminal_belief_readout": complete_terminal_readout,
            "complete_terminal_gap_trace": terminal_gap_trace,
            "termination_outcome": terminations[-1].get("outcome") if terminations else "MISSING",
            "controller_state_hash_present": bool(graph.get("controller_state_hash")),
            "join_preallocation_filtered_count": sum(
                len(trace.get("filtered", [])) for trace in traces
                if trace.get("event") == "join_preallocation_filtered"
            ),
            "selected_infeasible_join_count": sum(
                bool((trace.get("diagnostics") or {}).get("preallocation_feasible") is False)
                for trace in traces if trace.get("event") == "join_rejected"
            ),
            "repeated_same_fingerprint_extraction_count": (
                len(extraction_fingerprints) - len(set(extraction_fingerprints))
            ),
            "region_retrieval_gate_count": len(retrieval_gates),
            "region_retrieval_allowed_count": sum(
                bool(trace.get("allowed")) for trace in retrieval_gates
            ),
            **proof_metrics,
        }
        per_example.append(row)
    return _aggregate(per_example), _group_by_hop(per_example), graph_rows, per_example


def _revision_label(row: dict[str, Any], claims: dict[str, Any], example: QAExample | None) -> str:
    explicit = str(row.get("correctness_label", "pending"))
    if explicit in {"correct", "wrong"}:
        return explicit
    target = claims.get(str(row.get("target_claim_id", "")), {})
    value = normalize_text(str(target.get("value", "")))
    if not value or example is None:
        return "unknown"
    gold_values = {normalize_text(item) for item in example.answers}
    for step in example.oracle_decomposition:
        if isinstance(step, dict):
            for key in ("answer", "answers", "output", "value"):
                raw = step.get(key)
                if isinstance(raw, list):
                    gold_values.update(normalize_text(item) for item in raw)
                elif raw is not None:
                    gold_values.add(normalize_text(str(raw)))
    if value in gold_values:
        return "wrong"
    if row.get("evidence_ids") and str(row.get("trigger", "")) == "contradiction_pressure":
        return "correct"
    return "unknown"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric = (
        "claim_count", "typed_claim_rate", "join_count", "auditable_join_count",
        "diffusion_count", "typed_message_count", "allocation_count",
        "activated_passage_count", "activated_entity_count", "cross_layer_edge_count",
        "memory_activation_message_count", "selected_fidelity_level_count",
        "nary_join_attempt_count", "nary_join_accepted_count", "nary_join_downstream_used_count",
        "natural_revision_count", "natural_revision_correct", "natural_revision_wrong",
        "natural_revision_unknown", "unsupported_answer_count",
        "terminal_belief_count",
        "join_preallocation_filtered_count", "selected_infeasible_join_count",
        "repeated_same_fingerprint_extraction_count", "region_retrieval_gate_count",
        "region_retrieval_allowed_count", "dependency_coverage",
        "evidence_leaf_coverage", "distinct_evidence_leaf_ratio", "proof_depth",
    )
    result = {"count": len(rows)}
    result.update({f"mean_{key}": mean(float(row[key]) for row in rows) for key in numeric})
    for key in (
        "auditable_three_or_four_hop_join_case", "candidate_presence", "candidate_survival",
        "non_uniform_allocation", "complete_evc_trace", "controller_state_hash_present",
        "complete_outcome_feedback_trace", "feedback_influenced_allocation",
        "query_graph_present",
        "complete_terminal_belief_readout", "complete_terminal_gap_trace",
        "graph_proof_completion", "proof_connected",
    ):
        result[f"{key}_rate"] = mean(float(bool(row[key])) for row in rows)
    correct = sum(int(row["natural_revision_correct"]) for row in rows)
    wrong = sum(int(row["natural_revision_wrong"]) for row in rows)
    result["natural_revision_precision"] = correct / max(1, correct + wrong)
    result["termination_outcomes"] = dict(sorted(Counter(row["termination_outcome"] for row in rows).items()))
    result["unsupported_answer_count"] = sum(int(row["unsupported_answer_count"]) for row in rows)
    return result


def _best_graph_proof(
    graph_payload: dict[str, Any], *, allow_historical_unseal: bool = False,
) -> dict[str, Any]:
    """Recompute a gold-free proof score from the final serialized graph."""
    try:
        payload = graph_payload
        if allow_historical_unseal:
            payload = {**graph_payload, "controller_state_hash": ""}
        graph = DynamicReasoningHypergraphV2.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "graph_proof_completion": False,
            "dependency_coverage": 0.0,
            "evidence_leaf_coverage": 0.0,
            "distinct_evidence_leaf_ratio": 0.0,
            "proof_connected": False,
            "proof_depth": 0,
            "graph_proof_reason_codes": [
                f"unreadable_graph_snapshot:{type(exc).__name__}:{exc}"
            ],
        }
    roots = [
        row.node_id for row in graph.subgoals() if row.terminal
    ]
    if "subgoal_root" in graph.execution_graph.dependencies:
        roots = ["subgoal_root"]
    candidates = []
    for root_id in roots:
        for branch in graph.branches.values():
            for claim in graph.claims(root_id, branch.branch_id):
                if claim.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}:
                    continue
                candidates.append(audit_graph_proof(
                    graph, root_id, branch.branch_id, [claim.node_id],
                ))
    if not candidates:
        return {
            "graph_proof_completion": False,
            "dependency_coverage": 0.0,
            "evidence_leaf_coverage": 0.0,
            "distinct_evidence_leaf_ratio": 0.0,
            "proof_connected": False,
            "proof_depth": 0,
            "graph_proof_reason_codes": ["no_root_claim_candidate"],
        }
    best = max(candidates, key=lambda value: (
        int(value.graph_proof_completion),
        value.dependency_coverage,
        value.evidence_leaf_coverage,
        int(value.proof_connected),
        value.proof_depth,
        value.claim_ids,
    ))
    return {
        "graph_proof_completion": best.graph_proof_completion,
        "dependency_coverage": best.dependency_coverage,
        "evidence_leaf_coverage": best.evidence_leaf_coverage,
        "distinct_evidence_leaf_ratio": best.distinct_evidence_leaf_ratio,
        "proof_connected": best.proof_connected,
        "proof_depth": best.proof_depth,
        "graph_proof_reason_codes": list(best.reason_codes),
    }


def _group_by_hop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("hop_count") if row.get("hop_count") is not None else "unknown")].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}
