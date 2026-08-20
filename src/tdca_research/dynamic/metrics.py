from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from ..models import QAExample
from ..utils import normalize_text


def dynamic_mechanism_metrics(
    examples: list[QAExample], reasoning_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute mechanism diagnostics from durable graph snapshots only."""
    rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reasoning_rows:
        if row.get("graph_snapshot") and row.get("qid"):
            rows_by_qid[str(row["qid"])].append(row)
    example_by_qid = {example.qid: example for example in examples}
    per_example = []
    graph_rows = []
    for qid, steps in sorted(rows_by_qid.items()):
        steps = sorted(steps, key=lambda value: int(value.get("step_id", 0)))
        graph = steps[-1]["graph_snapshot"]
        graph_rows.append({"qid": qid, "graph": graph})
        nodes = graph.get("nodes", {})
        claims = [value for value in nodes.values() if value.get("kind") == "claim"]
        answers = [value for value in nodes.values() if value.get("kind") == "answer"]
        operations = Counter(str(value.get("operation", "")) for value in steps)
        active_claims = [value for value in claims if value.get("status") not in {"archived", "invalid"}]
        pruned = {node_id for step in steps for node_id in step.get("pruned_nodes", [])}
        created_claims = {
            node_id for step in steps for node_id in step.get("created_nodes", [])
            if nodes.get(node_id, {}).get("kind") == "claim"
        }
        example = example_by_qid.get(qid)
        gold = {normalize_text(value) for value in (example.answers if example else []) if normalize_text(value)}
        gold_candidates = {
            node_id for node_id, value in nodes.items()
            if value.get("kind") == "claim" and normalize_text(str(value.get("value", ""))) in gold
        }
        scheduler_multi = [
            step for step in steps
            if int((step.get("scheduler") or {}).get("ready_operation_count", 0)) > 1
        ]
        grounded_answers = [
            answer for answer in answers
            if answer.get("supporting_claims") and answer.get("supporting_evidence")
            and answer.get("derivation_edge") in graph.get("hyperedges", {})
        ]
        branch_count = len(graph.get("branches", {}))
        peak_nodes = max(len(step["graph_snapshot"].get("nodes", {})) for step in steps)
        peak_edges = max(len(step["graph_snapshot"].get("hyperedges", {})) for step in steps)
        row = {
            "qid": qid,
            "hop_count": None if example is None else example.hop_count,
            "operation_count": len(steps),
            "operation_counts": dict(sorted(operations.items())),
            "candidate_count": len(created_claims),
            "candidate_survival_rate": len([value for value in active_claims if value.get("node_id") in created_claims]) / max(1, len(created_claims)),
            "branch_activated": branch_count > 1,
            "branch_count": branch_count,
            "revision_count": len(graph.get("revision_history", [])),
            "pruned_candidate_count": len(pruned & created_claims),
            "prune_rate": len(pruned & created_claims) / max(1, len(created_claims)),
            "commit_reversal_count": sum(
                1 for value in graph.get("revision_history", [])
                if (value.get("before") or {}).get("status") == "committed"
            ),
            "scheduler_multi_ready_steps": len(scheduler_multi),
            "scheduler_activation_steps": sum(bool((step.get("scheduler") or {}).get("scheduler_active")) for step in steps),
            "mean_ready_operations": mean(
                int((step.get("scheduler") or {}).get("ready_operation_count", 0)) for step in steps
            ),
            "peak_nodes": peak_nodes,
            "peak_hyperedges": peak_edges,
            "terminal_candidate_count": len(answers),
            "grounded_terminal_count": len(grounded_answers),
            "answer_emitted": bool(answers),
            "grounded_terminal_rate": len(grounded_answers) / max(1, len(answers)),
            "gold_candidate_generated": bool(gold_candidates),
            "gold_candidate_survived": bool(gold_candidates & {value.get("node_id") for value in active_claims}),
            "false_prune": bool(gold_candidates & pruned),
            "llm_calls_attributed": sum(float(step.get("llm_call_count", 0.0)) for step in steps),
            "tokens_attributed": sum(float(step.get("token_cost", 0.0)) for step in steps),
        }
        per_example.append(row)
    return _aggregate(per_example), _group_by_hop(per_example), graph_rows, per_example


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric = [
        "operation_count", "candidate_count", "candidate_survival_rate", "branch_count",
        "revision_count", "pruned_candidate_count", "prune_rate", "commit_reversal_count",
        "scheduler_multi_ready_steps", "scheduler_activation_steps", "mean_ready_operations",
        "peak_nodes", "peak_hyperedges", "terminal_candidate_count", "grounded_terminal_rate",
        "grounded_terminal_count",
        "llm_calls_attributed", "tokens_attributed",
    ]
    result: dict[str, Any] = {"count": len(rows)}
    result.update({f"mean_{key}": mean(float(row[key]) for row in rows) for key in numeric})
    for key in ("branch_activated", "answer_emitted", "gold_candidate_generated", "gold_candidate_survived", "false_prune"):
        result[f"{key}_rate"] = mean(float(bool(row[key])) for row in rows)
    result["emitted_answer_grounding_rate"] = (
        sum(int(row["grounded_terminal_count"]) for row in rows)
        / max(1, sum(int(row["terminal_candidate_count"]) for row in rows))
    )
    totals = Counter()
    for row in rows:
        totals.update(row["operation_counts"])
    result["operation_counts"] = dict(sorted(totals.items()))
    return result


def _group_by_hop(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("hop_count") if row.get("hop_count") is not None else "unknown")].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}
