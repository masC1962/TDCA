from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..dynamic.graph import (
    BranchState,
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    GraphLimits,
    GraphOperation,
    OperationType,
)
from ..llm import BaseLLM
from ..utils import stable_hash
from .config import DynamicV2ResearchConfig
from .controller import V2GraphController
from .graph import DynamicReasoningHypergraphV2
from .revision import BeliefRevisionDetector


REVISION_RELATION_SYSTEM = """You independently assess one existing belief against one newly arrived evidence passage.
Do not assume either text is correct merely because it is provided. Judge only the semantic relation of the evidence
to the exact claim. Return one JSON object with: relation (supports|refutes|insufficient), grounding,
support_entailment, contradiction_entailment, insufficiency, confidence (all independent raw numbers in [0,1]),
evidence_span (a short exact span from the evidence), and reason_codes (a short list). A changed number, date,
entity, polarity, or relation can refute a claim. Related text that cannot establish or contradict the exact claim
is insufficient. Do not output a final revision action and do not collapse the raw scores into one probability."""


@dataclass(frozen=True)
class RevisionInput:
    item_id: str
    case_id: str
    page: str
    claim: str
    evidence: str
    content_sha256: str


@dataclass(frozen=True)
class RelationAssessment:
    relation: str
    grounding: float
    support_entailment: float
    contradiction_entailment: float
    insufficiency: float
    confidence: float
    evidence_span: str
    reason_codes: tuple[str, ...]


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _content_hash(claim: str, evidence: str) -> str:
    payload = json.dumps(
        {"claim": claim, "evidence": evidence}, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "during", "for", "from", "had", "has", "have", "he", "her", "his",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "will", "with",
}
_DIRECTIONAL_TERMS = {"after", "before", "less", "more", "under", "over", "until", "since"}


def _lexical_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.casefold())
        if token not in _LEXICAL_STOPWORDS
    ]


def _direct_support_audit(item: RevisionInput) -> dict[str, Any]:
    """Conservative, label-free guard against revising a belief its evidence restates.

    It only suppresses an LLM refutation when the evidence covers nearly all claim
    content. Missing claim-side numbers or directional comparators are treated as
    explicit conflicts, so a superficially similar changed date/quantity does not
    pass the guard. Page-title tokens are excluded because the page context already
    establishes the subject and need not be repeated in every evidence sentence.
    """
    page_tokens = set(_lexical_tokens(item.page))
    claim_tokens = [token for token in _lexical_tokens(item.claim) if token not in page_tokens]
    evidence_tokens = set(_lexical_tokens(item.evidence))
    covered = sum(token in evidence_tokens for token in claim_tokens)
    coverage = covered / max(1, len(claim_tokens))
    claim_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", item.claim.casefold()))
    evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", item.evidence.casefold()))
    missing_numbers = sorted(claim_numbers - evidence_numbers)
    claim_directionals = set(_lexical_tokens(item.claim)) & _DIRECTIONAL_TERMS
    evidence_directionals = set(_lexical_tokens(item.evidence)) & _DIRECTIONAL_TERMS
    missing_directionals = sorted(claim_directionals - evidence_directionals)
    return {
        "coverage": coverage,
        "claim_content_token_count": len(claim_tokens),
        "missing_claim_numbers": missing_numbers,
        "missing_directional_terms": missing_directionals,
        "no_explicit_value_conflict": not missing_numbers and not missing_directionals,
    }


def load_unlabeled_inputs(
    input_manifest_path: str | Path,
    split: str,
    *,
    unlabeled_cache_path: str | Path | None = None,
) -> tuple[list[RevisionInput], dict[str, Any]]:
    """Load prediction inputs without ever opening the sealed label manifest."""
    manifest_path = Path(input_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in {"development", "evaluation"}:
        raise ValueError("revision split must be development or evaluation")
    expected_rows = manifest.get("splits", {}).get(split)
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError(f"missing revision input split {split}")
    provenance = manifest.get("provenance", {})
    cache_path = Path(unlabeled_cache_path or provenance.get("unlabeled_cache", ""))
    if not cache_path.exists():
        raise FileNotFoundError(
            f"unlabeled revision cache missing: {cache_path}; run scripts/build_revision_suite.py"
        )
    observed_cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if observed_cache_hash != provenance.get("unlabeled_cache_sha256"):
        raise ValueError("unlabeled revision cache checksum mismatch")
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cache[str(row["item_id"])] = row
    resolved: list[RevisionInput] = []
    for expected in expected_rows:
        item_id = str(expected["item_id"])
        row = cache.get(item_id)
        if row is None:
            raise ValueError(f"unlabeled cache lacks item {item_id}")
        observed_content = _content_hash(str(row["claim"]), str(row["evidence"]))
        if observed_content != expected.get("content_sha256") or observed_content != row.get("content_sha256"):
            raise ValueError(f"content checksum mismatch for {item_id}")
        if str(row["case_id"]) != str(expected["case_id"]):
            raise ValueError(f"case mismatch for {item_id}")
        resolved.append(RevisionInput(
            item_id=item_id,
            case_id=str(row["case_id"]),
            page=str(row.get("page", "")),
            claim=str(row["claim"]),
            evidence=str(row["evidence"]),
            content_sha256=observed_content,
        ))
    return resolved, manifest


class IndependentRevisionScorer:
    def __init__(self, llm: BaseLLM, *, max_tokens: int = 450) -> None:
        self.llm = llm
        self.max_tokens = int(max_tokens)

    def assess(self, item: RevisionInput) -> tuple[RelationAssessment, Any]:
        data, generation = self.llm.generate_json([
            {"role": "system", "content": REVISION_RELATION_SYSTEM},
            {"role": "user", "content": json.dumps({
                "claim": item.claim,
                "new_evidence": item.evidence,
                "page_context": item.page,
            }, ensure_ascii=False)},
        ], "independent_belief_revision_relation_v1", self.max_tokens, temperature=0.0)
        relation = str(data.get("relation", "insufficient")).strip().casefold().replace(" ", "_")
        aliases = {"support": "supports", "contradicts": "refutes", "contradict": "refutes",
                   "not_enough_info": "insufficient", "neutral": "insufficient"}
        relation = aliases.get(relation, relation)
        if relation not in {"supports", "refutes", "insufficient"}:
            relation = "insufficient"
        evidence_span = str(data.get("evidence_span", "")).strip()[:500]
        # A claimed quotation must be auditable against the supplied passage.
        if evidence_span and evidence_span.casefold() not in item.evidence.casefold():
            evidence_span = ""
        assessment = RelationAssessment(
            relation=relation,
            grounding=_unit(data.get("grounding")),
            support_entailment=_unit(data.get("support_entailment")),
            contradiction_entailment=_unit(data.get("contradiction_entailment")),
            insufficiency=_unit(data.get("insufficiency")),
            confidence=_unit(data.get("confidence")),
            evidence_span=evidence_span,
            reason_codes=tuple(str(value)[:100] for value in data.get("reason_codes", [])[:6]),
        )
        return assessment, generation


def _operation(
    item_id: str,
    number: int,
    kind: OperationType,
    *,
    payload: dict[str, Any],
    sources: Iterable[str] = (),
) -> GraphOperation:
    suffix = stable_hash(item_id)[:10]
    return GraphOperation(
        operation_id=f"revision_{suffix}_{number:02d}",
        operation_type=kind,
        target_id="revision_subgoal",
        source_ids=list(sources),
        branch_id="branch_root",
        payload=payload,
        reason="public_natural_revision_suite",
        proposed_by="revision_suite_v1",
        estimated_cost={"llm_calls": 0.0, "tokens": 0.0, "retrieval_calls": 0.0},
    )


def _graph_limits(config: DynamicV2ResearchConfig) -> GraphLimits:
    return GraphLimits(
        config.max_candidates_per_subgoal,
        config.max_active_branches,
        config.max_graph_nodes,
        config.max_hyperedges,
        config.max_graph_revisions,
        config.max_revision_per_candidate,
        config.max_graph_depth,
        config.max_graph_operations,
        config.max_retrieval_calls,
    )


def apply_natural_revision_episode(
    item: RevisionInput,
    assessment: RelationAssessment,
    config: DynamicV2ResearchConfig,
) -> tuple[DynamicReasoningHypergraphV2, dict[str, Any]]:
    """Materialize a prior belief, introduce evidence, and let graph events decide revision."""
    controller = V2GraphController(config)
    graph = DynamicReasoningHypergraphV2(
        question=f"Should the stored belief about {item.page or 'this subject'} be revised?",
        limits=_graph_limits(config),
    )
    graph.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    graph.seal_controller_state()
    graph = controller.apply(graph, _operation(item.item_id, 1, OperationType.EXPAND, payload={
        "subgoals": [{
            "node_id": "revision_subgoal",
            "question_template": "Assess whether a stored belief remains supported",
            "instantiated_question": "Assess whether a stored belief remains supported",
            "dependencies": [],
            "variable_bindings": {},
            "answer_type": "revision_decision",
            "terminal": True,
        }],
    }))
    graph = controller.apply(graph, _operation(item.item_id, 2, OperationType.RETRIEVE, payload={
        "query": "stored belief",
        "evidence": [{
            "node_id": "prior_memory",
            "document_id": f"memory:{item.case_id}",
            "passage_id": f"memory:{item.item_id}",
            "title": item.page,
            "source_span": item.claim,
            "retrieval_rank": 1,
            "retrieval_score": 1.0,
            "retrieval_query": "stored belief",
            "retriever_identity": "ephemeral_question_memory",
        }],
    }))
    graph = controller.apply(graph, _operation(item.item_id, 3, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "prior_claim",
            "subject": item.page or "stored_subject",
            "relation": "stored_belief",
            "value": item.claim,
            "subject_type": "entity",
            "value_type": "proposition",
            "answer_type": "proposition",
            "evidence_refs": ["prior_memory"],
            "source_spans": [item.claim],
            "dependency_claim_ids": [],
            "extraction_confidence": 1.0,
        }],
    }))
    strong = {
        "grounding": 1.0, "entailment": 1.0, "type_match": 1.0,
        "dependency_consistency": 1.0, "retrieval_support": 1.0,
        "contradiction_risk": 0.0, "raw_model_confidence": 1.0,
        "absolute_support": 0.95, "relative_weight": 1.0,
        "set_entropy": 0.05, "evidence_gap": 0.05, "status": "scored",
        "scoring_audit": {"source": "ephemeral_prior_belief", "gold_label_used": False},
    }
    graph = controller.apply(graph, _operation(item.item_id, 4, OperationType.VERIFY, payload={
        "scores": {"prior_claim": strong},
    }, sources=["prior_claim", "prior_memory"]))
    graph = controller.apply(graph, _operation(item.item_id, 5, OperationType.COMMIT, payload={
        "candidate_id": "prior_claim",
    }, sources=["prior_claim"]))
    graph = controller.apply(graph, _operation(item.item_id, 6, OperationType.RETRIEVE, payload={
        "query": "new external evidence",
        "evidence": [{
            "node_id": "incoming_evidence",
            "document_id": f"vitaminc:{item.case_id}",
            "passage_id": item.item_id,
            "title": item.page,
            "source_span": item.evidence,
            "retrieval_rank": 1,
            "retrieval_score": 1.0,
            "retrieval_query": "new external evidence",
            "retriever_identity": "vitaminc_public_suite",
        }],
    }))
    graph = controller.apply(graph, _operation(item.item_id, 7, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "incoming_claim",
            "subject": item.page or "incoming_subject",
            "relation": "incoming_evidence_states",
            "value": item.evidence,
            "subject_type": "entity",
            "value_type": "proposition",
            "answer_type": "proposition",
            "evidence_refs": ["incoming_evidence"],
            "source_spans": [assessment.evidence_span or item.evidence[:500]],
            "dependency_claim_ids": [],
            "extraction_confidence": assessment.confidence,
        }],
    }))
    direct_support = _direct_support_audit(item)
    direct_support_override = (
        direct_support["coverage"] >= config.revision_lexical_support_override_threshold
        and direct_support["claim_content_token_count"] >= 3
        and direct_support["no_explicit_value_conflict"]
    )
    contradiction = (
        assessment.relation == "refutes"
        and assessment.contradiction_entailment >= config.contradiction_threshold
        and assessment.grounding >= config.retain_support_threshold
        and not direct_support_override
    )
    incoming_support = max(assessment.grounding, assessment.confidence)
    prior_support = max(0.0, 0.95 - 0.70 * assessment.contradiction_entailment) if contradiction else 0.95
    prior_score = dict(strong)
    prior_score.update({
        "contradiction_risk": assessment.contradiction_entailment if contradiction else 0.0,
        "absolute_support": prior_support,
        "evidence_gap": max(0.05, assessment.contradiction_entailment) if contradiction else 0.05,
        "set_entropy": max(0.05, assessment.contradiction_entailment) if contradiction else 0.05,
        "status": "committed",
        "contradiction_links": ["incoming_claim"] if contradiction else [],
        "scoring_audit": {"source": "independent_raw_relation_scorer", "gold_label_used": False},
    })
    incoming_score = {
        "grounding": assessment.grounding,
        "entailment": max(assessment.support_entailment, assessment.contradiction_entailment),
        "type_match": 1.0,
        "dependency_consistency": 1.0,
        "retrieval_support": assessment.grounding,
        "contradiction_risk": 0.0,
        "raw_model_confidence": assessment.confidence,
        "absolute_support": incoming_support,
        "relative_weight": 1.0,
        "set_entropy": 1.0 - assessment.confidence,
        "evidence_gap": 1.0 - assessment.grounding,
        "status": "scored",
        "contradiction_links": ["prior_claim"] if contradiction else [],
        "scoring_audit": {"source": "independent_raw_relation_scorer", "gold_label_used": False},
    }
    graph = controller.apply(graph, _operation(item.item_id, 8, OperationType.VERIFY, payload={
        "scores": {"prior_claim": prior_score, "incoming_claim": incoming_score},
    }, sources=["prior_claim", "incoming_claim", "incoming_evidence"]))
    triggers = BeliefRevisionDetector(config).detect(graph)
    if triggers:
        graph = controller.apply(graph, BeliefRevisionDetector.operation(
            graph,
            triggers[0],
            "branch_root",
            f"revision_{stable_hash(item.item_id)[:10]}_09",
            natural=True,
        ))
    graph.validate()
    predicted_revise = any(
        row.target_claim_id == "prior_claim" and row.natural
        for row in graph.supersession_history
    )
    trace = {
        "contradiction_gate_passed": contradiction,
        "direct_support_override": direct_support_override,
        "direct_support_audit": direct_support,
        "trigger_count": len(triggers),
        "trigger": asdict(triggers[0]) if triggers else None,
        "supersession_count": len(graph.supersession_history),
        "prior_status": graph.node("prior_claim", ClaimNode).status.value,
        "controller_state_hash": graph.controller_state_hash,
        "invariant_valid": True,
    }
    return graph, {"predicted_action": "should_revise" if predicted_revise else "should_not_revise", **trace}


def predict_items(
    items: Iterable[RevisionInput],
    llm: BaseLLM,
    config: DynamicV2ResearchConfig,
) -> list[dict[str, Any]]:
    scorer = IndependentRevisionScorer(llm)
    predictions: list[dict[str, Any]] = []
    for item in items:
        assessment, generation = scorer.assess(item)
        graph, decision = apply_natural_revision_episode(item, assessment, config)
        provider_attempts = int(generation.metadata.get("provider_attempts", 1))
        provider_reported_tokens = (
            0 if generation.cached
            else generation.prompt_tokens + generation.completion_tokens
        )
        predictions.append({
            "item_id": item.item_id,
            "case_id": item.case_id,
            "content_sha256": item.content_sha256,
            "assessment": asdict(assessment),
            "decision": decision,
            "usage": {
                "provider_calls": 0 if generation.cached else provider_attempts,
                "logical_calls": 1,
                "provider_reported_tokens": provider_reported_tokens,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "cached": generation.cached,
            },
            "graph_audit": {
                "operation_count": len(graph.operation_history),
                "revision_count": len(graph.supersession_history),
                "state_hash": graph.controller_state_hash,
            },
        })
    return predictions


def score_prediction_rows(
    predictions: list[dict[str, Any]],
    label_manifest: dict[str, Any],
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = label_manifest.get("splits", {}).get(split, [])
    label_by_id = {str(row["item_id"]): row for row in labels}
    if set(label_by_id) != {str(row.get("item_id", "")) for row in predictions}:
        raise ValueError("prediction and sealed-label item sets differ")
    scored: list[dict[str, Any]] = []
    for prediction in predictions:
        item_id = str(prediction["item_id"])
        label = label_by_id[item_id]
        expected = str(label["expected_action"])
        predicted = str(prediction["decision"]["predicted_action"])
        scored.append({
            "item_id": item_id,
            "expected_action": expected,
            "predicted_action": predicted,
            "gold_relation": str(label["gold_relation"]),
            "category": str(label["category"]),
            "correct": predicted == expected,
        })
    metrics = _binary_metrics(scored)
    metrics["no_revision_ablation_accuracy"] = sum(
        row["expected_action"] == "should_not_revise" for row in scored
    ) / max(1, len(scored))
    metrics["revision_gain_over_no_revision"] = (
        metrics["accuracy"] - metrics["no_revision_ablation_accuracy"]
    )
    metrics["provider_calls"] = sum(int(row["usage"]["provider_calls"]) for row in predictions)
    metrics["provider_reported_tokens"] = sum(
        int(row["usage"]["provider_reported_tokens"]) for row in predictions
    )
    metrics["complete_predictions"] = len(predictions) == len(labels)
    metrics["zero_invariant_violations"] = all(
        bool(row["decision"].get("invariant_valid")) for row in predictions
    )
    metrics["thresholds"] = {
        "precision_at_least_0_80": metrics["precision"] >= 0.80,
        "recall_at_least_0_60": metrics["recall"] >= 0.60,
        "false_positive_rate_at_most_0_10": metrics["false_positive_rate"] <= 0.10,
    }
    metrics["passes_natural_revision_gate"] = all(metrics["thresholds"].values())
    metrics["by_category"] = {
        category: _binary_metrics([row for row in scored if row["category"] == category])
        for category in sorted({row["category"] for row in scored})
    }
    return metrics, scored


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["expected_action"] == "should_revise" and row["predicted_action"] == "should_revise" for row in rows)
    fp = sum(row["expected_action"] == "should_not_revise" and row["predicted_action"] == "should_revise" for row in rows)
    fn = sum(row["expected_action"] == "should_revise" and row["predicted_action"] == "should_not_revise" for row in rows)
    tn = sum(row["expected_action"] == "should_not_revise" and row["predicted_action"] == "should_not_revise" for row in rows)
    return {
        "count": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
    }
