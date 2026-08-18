from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..models import ClaimStatus, Prediction, QAExample, RunStatus
from ..utils import normalize_text


def exact_match(prediction: str | None, answers: list[str]) -> float:
    normalized = normalize_text(prediction or "")
    # Match the official MuSiQue/HotpotQA/2Wiki equality semantics exactly.
    # Dataset validation, rather than the scorer, is responsible for rejecting
    # invalid empty gold answers.
    return float(any(normalized == normalize_text(answer) for answer in answers))


def token_f1(prediction: str | None, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction or "")
    predicted = normalized_prediction.split()
    best = 0.0
    for answer in answers:
        normalized_gold = normalize_text(answer)
        # HotpotQA and 2Wiki official scorers assign zero partial credit when
        # yes/no/noanswer types disagree. This is a general answer-type rule.
        categorical = {"yes", "no", "noanswer"}
        if (normalized_prediction in categorical or normalized_gold in categorical) and normalized_prediction != normalized_gold:
            continue
        gold = normalized_gold.split()
        if not predicted or not gold:
            continue
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        if not overlap:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def retrieval_scores(retrieved_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    retrieved = set(retrieved_ids)
    gold = set(gold_ids)
    overlap = len(retrieved & gold)
    precision = overlap / len(retrieved) if retrieved else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores = {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "all_gold_recalled": float(bool(gold) and gold.issubset(retrieved)),
        "ordered_evidence_path_recall": _ordered_subsequence_recall(retrieved_ids, gold_ids),
    }
    for cutoff in (1, 2, 5, 10):
        prefix = set(retrieved_ids[:cutoff])
        scores[f"recall_at_{cutoff}"] = len(prefix & gold) / len(gold) if gold else 0.0
    return scores


def _ordered_subsequence_recall(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of the gold evidence order recovered as a subsequence."""
    retrieved_ids = list(dict.fromkeys(retrieved_ids))
    gold_ids = list(dict.fromkeys(gold_ids))
    if not gold_ids:
        return 0.0
    width = len(gold_ids) + 1
    previous = [0] * width
    for retrieved in retrieved_ids:
        current = previous[:]
        for index, gold in enumerate(gold_ids, start=1):
            if retrieved == gold:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(current[index], current[index - 1])
        previous = current
    return previous[-1] / len(gold_ids)


def _oracle_edges(example: QAExample) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for target, step in enumerate(example.oracle_decomposition, start=1):
        question = str(step.get("question", ""))
        for source in range(1, target):
            if f"#{source}" in question:
                edges.add((source, target))
    return edges


def _predicted_edges(prediction: Prediction) -> set[tuple[int, int]]:
    if prediction.plan is None:
        return set()
    positions = {slot.slot_id: index for index, slot in enumerate(prediction.plan.slots, start=1)}
    return {
        (positions[source], positions[slot.slot_id])
        for slot in prediction.plan.slots for source in slot.dependencies
        if source in positions
    }


def reasoning_scores(example: QAExample, prediction: Prediction, answer_em: float) -> dict[str, float | None]:
    """Gold-dependent diagnostics; never consumed by inference code."""
    oracle = example.oracle_decomposition
    if not oracle or prediction.plan is None:
        return {
            "decomposition_node_accuracy": None,
            "decomposition_edge_f1": None,
            "variable_binding_accuracy": None,
            "verified_claim_precision": None,
            "grounded_claim_precision": None,
            "full_chain_correct": None,
            "terminal_slot_accuracy": None,
            "grounded_answer": None,
        }
    slots = prediction.plan.slots
    node_scores = []
    for index, step in enumerate(oracle):
        if index >= len(slots):
            node_scores.append(0.0)
            continue
        gold_question = str(step.get("question", ""))
        predicted_question = slots[index].subquestion_template
        # Variable names are implementation details; compare the actual words.
        import re
        gold_question = re.sub(r"#\d+", "bridge", gold_question)
        predicted_question = re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*", "bridge", predicted_question)
        node_scores.append(token_f1(predicted_question, [gold_question]))
    if len(slots) > len(oracle):
        node_scores.extend([0.0] * (len(slots) - len(oracle)))
    gold_edges = _oracle_edges(example)
    predicted_edges = _predicted_edges(prediction)
    overlap = len(gold_edges & predicted_edges)
    edge_precision = overlap / len(predicted_edges) if predicted_edges else float(not gold_edges)
    edge_recall = overlap / len(gold_edges) if gold_edges else float(not predicted_edges)
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if edge_precision + edge_recall else 0.0
    binding_pairs = set()
    positions = {slot.slot_id: index for index, slot in enumerate(slots, start=1)}
    for target, slot in enumerate(slots, start=1):
        for binding in slot.variable_bindings:
            if binding.source_slot in positions:
                binding_pairs.add((positions[binding.source_slot], target))
    binding_accuracy = len(binding_pairs & gold_edges) / len(gold_edges) if gold_edges else float(not binding_pairs)
    verified = [claim for claim in prediction.claims if claim.status == ClaimStatus.VERIFIED]
    correct_verified = 0
    for claim in verified:
        try:
            slot_index = positions[claim.target_slot] - 1
        except KeyError:
            continue
        if slot_index < len(oracle):
            gold_answer = str(oracle[slot_index].get("answer", "")).strip()
            correct_verified += int(bool(gold_answer) and exact_match(claim.object, [gold_answer]) == 1.0)
    grounded = sum(bool(claim.source_document_ids and claim.source_spans) for claim in verified)
    chain_complete = bool(slots) and all(slot.status.value == "complete" for slot in slots)
    return {
        "decomposition_node_accuracy": sum(node_scores) / max(1, len(node_scores)),
        "decomposition_edge_f1": edge_f1,
        "variable_binding_accuracy": binding_accuracy,
        "verified_claim_precision": correct_verified / len(verified) if verified else 0.0,
        "grounded_claim_precision": grounded / len(verified) if verified else 0.0,
        "full_chain_correct": float(chain_complete and bool(answer_em)),
        "terminal_slot_accuracy": float(bool(answer_em) and any(slot.terminal and slot.status.value == "complete" for slot in slots)),
        "grounded_answer": float(bool(answer_em) and any(
            claim.status == ClaimStatus.VERIFIED and claim.source_document_ids and claim.source_spans
            for claim in prediction.claims if claim.target_slot in {slot.slot_id for slot in slots if slot.terminal}
        )),
    }


def risk_coverage_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    ranked = sorted(rows, key=lambda row: (-float(row["confidence"]), str(row["qid"])))
    curve = []
    correct = 0.0
    for index, row in enumerate(ranked, start=1):
        correct += float(row["exact_match"])
        curve.append({
            "coverage": index / max(1, len(ranked)),
            "risk": 1.0 - correct / index,
            "threshold": float(row["confidence"]),
        })
    return curve


def expected_calibration_error(rows: list[dict[str, Any]], bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    bucketed: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        confidence = max(0.0, min(1.0, float(row["confidence"])))
        bucket = min(bins - 1, int(confidence * bins))
        bucketed[bucket].append((confidence, float(row["exact_match"])))
    total = max(1, len(rows))
    ece = 0.0
    report = []
    for bucket in range(bins):
        values = bucketed.get(bucket, [])
        if not values:
            continue
        mean_confidence = sum(value[0] for value in values) / len(values)
        accuracy = sum(value[1] for value in values) / len(values)
        ece += len(values) / total * abs(mean_confidence - accuracy)
        report.append({"bin": bucket, "count": len(values), "confidence": mean_confidence, "accuracy": accuracy})
    return ece, report


def evaluate_predictions(examples: list[QAExample], predictions: list[Prediction]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {example.qid: example for example in examples}
    rows = []
    for prediction in predictions:
        if prediction.qid not in by_id:
            raise ValueError(f"prediction qid not found in dataset: {prediction.qid}")
        example = by_id[prediction.qid]
        retrieved_ids = list(dict.fromkeys(hit.passage.passage_id for hit in prediction.retrieved))
        retrieval = retrieval_scores(retrieved_ids, example.gold_document_ids)
        em = exact_match(prediction.answer, example.answers)
        f1 = token_f1(prediction.answer, example.answers)
        answer_in_context = float(any(
            bool(normalize_text(answer)) and normalize_text(answer) in normalize_text(hit.passage.text)
            for answer in example.answers for hit in prediction.retrieved
        ))
        reasoning = reasoning_scores(example, prediction, em)
        rows.append({
            "qid": prediction.qid,
            "hop_count": example.hop_count,
            "question_type": example.metadata.get("question_type"),
            "status": prediction.status.value,
            "answer": prediction.answer,
            "confidence": prediction.confidence,
            "exact_match": em,
            "f1": f1,
            "answer_in_context": answer_in_context,
            "full_chain_complete": float(bool(prediction.plan) and all(slot.status.value == "complete" for slot in prediction.plan.slots)),
            "verified_claim_count": sum(claim.status.value == "verified" for claim in prediction.claims),
            "llm_calls": prediction.usage.llm_calls,
            "provider_calls": prediction.usage.provider_calls,
            "retrieval_calls": prediction.usage.retrieval_calls,
            "prompt_tokens": prediction.usage.prompt_tokens,
            "completion_tokens": prediction.usage.completion_tokens,
            "provider_prompt_tokens": prediction.usage.provider_prompt_tokens,
            "provider_completion_tokens": prediction.usage.provider_completion_tokens,
            "total_tokens": prediction.usage.total_tokens,
            "wall_seconds": prediction.usage.wall_seconds,
            **retrieval,
            **reasoning,
        })
    ece, calibration = expected_calibration_error(rows)
    answered = [row for row in rows if row["status"] == RunStatus.ANSWER.value]
    infrastructure = [row for row in rows if row["status"] == RunStatus.INFRASTRUCTURE_FAILURE.value]
    aggregate = {
        "count": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / max(1, len(rows)),
        "f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
        "answered_rate": len(answered) / max(1, len(rows)),
        "abstention_rate": sum(row["status"] == RunStatus.ABSTAIN.value for row in rows) / max(1, len(rows)),
        "infrastructure_failure_rate": len(infrastructure) / max(1, len(rows)),
        "selective_accuracy": sum(row["exact_match"] for row in answered) / max(1, len(answered)),
        "support_precision": sum(row["support_precision"] for row in rows) / max(1, len(rows)),
        "support_recall": sum(row["support_recall"] for row in rows) / max(1, len(rows)),
        "support_f1": sum(row["support_f1"] for row in rows) / max(1, len(rows)),
        "all_gold_document_recall": sum(row["all_gold_recalled"] for row in rows) / max(1, len(rows)),
        "ordered_evidence_path_recall": sum(row["ordered_evidence_path_recall"] for row in rows) / max(1, len(rows)),
        "answer_in_context_rate": sum(row["answer_in_context"] for row in rows) / max(1, len(rows)),
        "full_chain_completion_rate": sum(row["full_chain_complete"] for row in rows) / max(1, len(rows)),
        "ece": ece,
        "calibration_bins": calibration,
        "risk_coverage_curve": risk_coverage_curve(rows),
    }
    for cutoff in (1, 2, 5, 10):
        aggregate[f"recall_at_{cutoff}"] = sum(row[f"recall_at_{cutoff}"] for row in rows) / max(1, len(rows))
    for key in (
        "decomposition_node_accuracy", "decomposition_edge_f1", "variable_binding_accuracy",
        "verified_claim_precision", "grounded_claim_precision", "full_chain_correct",
        "terminal_slot_accuracy", "grounded_answer",
    ):
        values = [float(row[key]) for row in rows if row[key] is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    return aggregate, rows


def grouped_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int | None]]:
    """Aggregate the same per-example metrics by an evaluation-only label."""
    groups: dict[str, dict[str, float | int | None]] = {}
    labels = sorted({str(row[field]) for row in rows if row.get(field) is not None})
    numeric_keys = (
        "exact_match", "f1", "support_precision", "support_recall", "support_f1",
        "all_gold_recalled", "ordered_evidence_path_recall", "answer_in_context",
        "full_chain_complete", "verified_claim_precision", "grounded_claim_precision",
        "full_chain_correct", "terminal_slot_accuracy", "grounded_answer", "llm_calls",
        "provider_calls", "retrieval_calls", "total_tokens", "provider_prompt_tokens",
        "provider_completion_tokens", "wall_seconds",
    )
    for label in labels:
        subset = [row for row in rows if str(row.get(field)) == label]
        group: dict[str, float | int | None] = {
            "count": len(subset),
            "answered_rate": sum(row["status"] == "answer" for row in subset) / len(subset),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in subset if row.get(key) is not None]
            group[key] = sum(values) / len(values) if values else None
        groups[label] = group
    return groups
