from __future__ import annotations

from collections import Counter
from typing import Any

from ..models import QAExample


class DatasetIntegrityError(ValueError):
    pass


def dataset_integrity_report(examples: list[QAExample], setting: str) -> dict[str, Any]:
    passage_counts = [len(example.passages) for example in examples]
    gold_counts = [len(example.gold_document_ids) for example in examples]
    missing_passages = [example.qid for example in examples if not example.passages]
    missing_gold = [example.qid for example in examples if not example.gold_document_ids]
    missing_answers = [example.qid for example in examples if not example.answers]
    duplicate_qids = sorted(qid for qid, count in Counter(example.qid for example in examples).items() if count > 1)
    duplicate_passage_ids = [
        example.qid for example in examples
        if len({passage.passage_id for passage in example.passages}) != len(example.passages)
    ]
    gold_outside_context = [
        example.qid for example in examples
        if set(example.gold_document_ids) - {passage.passage_id for passage in example.passages}
    ]
    no_distractors = [
        example.qid for example in examples
        if example.gold_document_ids and len(example.passages) <= len(example.gold_document_ids)
    ]
    return {
        "count": len(examples),
        "setting": setting,
        "passage_count_histogram": dict(sorted(Counter(passage_counts).items())),
        "gold_count_histogram": dict(sorted(Counter(gold_counts).items())),
        "missing_passage_ids": missing_passages,
        "missing_gold_evidence_ids": missing_gold,
        "missing_answer_ids": missing_answers,
        "duplicate_qids": duplicate_qids,
        "duplicate_passage_id_qids": duplicate_passage_ids,
        "gold_outside_context_ids": gold_outside_context,
        "no_distractor_ids": no_distractors,
    }


def validate_dataset_integrity(examples: list[QAExample], setting: str) -> dict[str, Any]:
    report = dataset_integrity_report(examples, setting)
    if not examples:
        raise DatasetIntegrityError("dataset contains no examples")
    if report["duplicate_qids"]:
        raise DatasetIntegrityError(f"duplicate question IDs: {report['duplicate_qids'][:5]}")
    if report["duplicate_passage_id_qids"]:
        raise DatasetIntegrityError(
            f"examples contain duplicate passage IDs: {report['duplicate_passage_id_qids'][:5]}"
        )
    if report["missing_answer_ids"]:
        raise DatasetIntegrityError(
            f"examples have no gold answer for evaluation: {report['missing_answer_ids'][:5]}"
        )
    if setting == "distractor" and report["missing_passage_ids"]:
        raise DatasetIntegrityError(
            f"{len(report['missing_passage_ids'])} examples have no passages; first ids: "
            f"{report['missing_passage_ids'][:5]}"
        )
    if setting == "distractor" and report["missing_gold_evidence_ids"]:
        raise DatasetIntegrityError(
            f"examples have no gold evidence IDs in distractor evaluation: "
            f"{report['missing_gold_evidence_ids'][:5]}"
        )
    if setting == "distractor" and report["gold_outside_context_ids"]:
        raise DatasetIntegrityError(
            f"gold evidence IDs are absent from distractor context: {report['gold_outside_context_ids'][:5]}"
        )
    if setting == "distractor" and len(report["no_distractor_ids"]) == len(examples):
        raise DatasetIntegrityError(
            "every example has only gold passages; this is a support-only diagnostic set, not a distractor setting"
        )
    return report
