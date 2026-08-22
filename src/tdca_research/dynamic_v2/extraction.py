from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from ..budget import Budget
from ..dynamic.graph import ClaimNode, EvidenceNode, GraphOperation, OperationType, SubgoalNode
from ..llm import BaseLLM
from ..utils import bounded_context, estimate_message_tokens, normalize_text
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2
from .query_graph import canonical_type


TYPED_EXTRACTION_SYSTEM = """Extract a small coverage-preserving set of atomic typed relational claims from
only the supplied evidence. Return JSON only as {claims:[...]}. Each compact claim object must contain exactly
subject, relation, value, subject_type, value_type, evidence_ids, extraction_confidence, answer_position,
and quote.
answer_position is subject, value, or none; use subject/value only when that exact field directly answers the
subgoal. Put direct-answer claims first, then relations needed to connect dependency entities. Use short
relation/type strings and evidence IDs. Also include quote: one short verbatim substring (at most 15 words)
from a referenced evidence item that directly states the claim. Include no qualifiers, reasons, prose, or
redundant fields.
Do not use prior knowledge, compare candidates, compose multiple facts, reverse the evidence direction, or
merge sentences. Return an empty list when evidence is insufficient."""


class TypedClaimExtractor:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def propose(
        self,
        graph: DynamicReasoningHypergraphV2,
        subgoal_id: str,
        branch_id: str,
        instantiated_question: str,
        dependency_claim_ids: list[str],
        operation_id: str,
        token_budget: int | None = None,
        candidate_cap: int | None = None,
    ) -> GraphOperation | None:
        evidence = graph.evidence(subgoal_id, branch_id)
        if not evidence:
            self.last_diagnostics = {"raw": 0, "accepted": 0, "rejections": {"no_evidence": 1}}
            return None
        dependencies = "\n".join(
            _claim_line(graph.node(node_id, ClaimNode)) for node_id in dependency_claim_ids
        ) or "(none)"
        focus_terms = _focus_terms(
            graph, subgoal_id, instantiated_question, dependency_claim_ids,
        )
        focused_rows = [
            f"[{node.node_id}] title={node.title}\n"
            f"{_focused_span(node, focus_terms, self.config)}"
            for node in evidence
        ]
        context = bounded_context(focused_rows, self.config.evidence_char_budget)
        query_constraint = next((
            row for row in graph.query_graph.get("constraints", [])
            if str(row.get("subgoal_id")) == subgoal_id
        ), {})
        cap = min(
            self.config.max_extracted_claims_per_round,
            max(1, int(candidate_cap)) if candidate_cap is not None else self.config.max_extracted_claims_per_round,
            max(0, graph.limits.max_candidates_per_subgoal - len([
                claim for claim in graph.claims(subgoal_id, branch_id)
                if claim.status.value not in {"invalid", "archived"}
            ])),
        )
        if cap <= 0:
            self.last_diagnostics = {"raw": 0, "accepted": 0, "rejections": {"claim_cap": 1}}
            return None
        messages = [
            {"role": "system", "content": TYPED_EXTRACTION_SYSTEM},
            {"role": "user", "content": (
                f"Root question: {graph.question}\nSubgoal: {instantiated_question}\n"
                f"Expected answer type: {graph.node(subgoal_id, SubgoalNode).answer_type}\n"
                f"Unresolved query constraint: {query_constraint}\n"
                f"Claim cap: {cap}\nDependency claims:\n{dependencies}\nEvidence:\n{context}"
            )},
        ]
        max_tokens = max(128, min(
            int(token_budget or self.config.typed_extraction_max_tokens),
            self.config.typed_extraction_max_tokens,
        ))
        self.budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages, "dynamic_v2_typed_claim_extraction_v1", max_tokens, self.config.temperature,
        )
        self.budget.record_generation(generation)
        available = {node.node_id: node for node in evidence}
        aliases = {
            alias: node.node_id
            for node in evidence
            for alias in {
                node.node_id, node.document_id, node.passage_id,
                f"[{node.node_id}]", f"[{node.document_id}]", f"[{node.passage_id}]",
            }
            if alias
        }
        existing = {
            (normalize_text(claim.subject), normalize_text(claim.relation), normalize_text(claim.value))
            for claim in graph.claims(subgoal_id, branch_id)
        }
        rows = []
        rejections: dict[str, int] = defaultdict(int)
        raw_rows = data.get("claims", []) if isinstance(data.get("claims"), list) else []
        for index, raw in enumerate(raw_rows[:cap], start=1):
            if not isinstance(raw, dict):
                rejections["non_object"] += 1
                continue
            subject = str(raw.get("subject", "")).strip()
            relation = str(raw.get("relation", "")).strip()
            value = str(raw.get("value", "")).strip()
            source_triple = {"subject": subject, "relation": relation, "value": value}
            subject_type = canonical_type(raw.get("subject_type"))
            value_type = canonical_type(raw.get("value_type"))
            expected_type = canonical_type(graph.node(subgoal_id, SubgoalNode).answer_type)
            value, canonicalization = _canonicalize_typed_value(value, expected_type)
            answer_position = str(raw.get("answer_position", "")).strip().lower()
            if answer_position not in {"subject", "value", "none"}:
                answer_position = "value" if bool(raw.get("answers_subgoal", False)) else "none"
            if answer_position == "subject":
                subject, value = value, subject
                subject_type, value_type = value_type, subject_type
                relation = f"inverse_of:{relation}"
            signature = (normalize_text(subject), normalize_text(relation), normalize_text(value))
            evidence_ids = list(dict.fromkeys(
                aliases[str(item)] for item in raw.get("evidence_ids", []) if str(item) in aliases
            ))
            spans = [str(item).strip() for item in raw.get("source_spans", []) if str(item).strip()]
            quote = str(raw.get("quote", "")).strip()
            if quote:
                spans.append(quote)
            span_matches = {
                node.node_id: [
                    span for span in spans
                    if normalize_text(span) in normalize_text(node.source_span)
                ]
                for node in evidence
            }
            for node_id, matches in span_matches.items():
                if matches and node_id not in evidence_ids:
                    evidence_ids.append(node_id)
            grounded_spans = [span for matches in span_matches.values() for span in matches]
            if not grounded_spans:
                # Deterministic fallback: accept only when the complete subject and
                # value co-occur in one evidence sentence.  This repairs citation
                # formatting, never semantic grounding or missing facts.
                for node in evidence:
                    sentence = _cooccurring_sentence(node.source_span, subject, value)
                    if sentence:
                        evidence_ids = list(dict.fromkeys(evidence_ids + [node.node_id]))
                        grounded_spans.append(sentence)
                        break
            if not all(signature):
                rejections["incomplete_triple"] += 1
                continue
            if signature in existing:
                rejections["duplicate_triple"] += 1
                continue
            if not evidence_ids or not grounded_spans:
                rejections["ungrounded"] += 1
                continue
            existing.add(signature)
            rows.append({
                "node_id": f"claim_v2_{graph.step + 1}_{subgoal_id}_{index}",
                "subject": subject,
                "relation": relation,
                "value": value,
                "subject_type": subject_type,
                "value_type": value_type,
                "answer_type": value_type,
                "qualifiers": {
                    **(raw.get("qualifiers", {}) if isinstance(raw.get("qualifiers"), dict) else {}),
                    **({"value_canonicalization": canonicalization} if canonicalization else {}),
                },
                "evidence_refs": evidence_ids,
                "source_spans": grounded_spans[:3],
                "dependency_claim_ids": dependency_claim_ids,
                "extraction_confidence": _unit(raw.get("extraction_confidence")),
                "answers_subgoal": answer_position in {"subject", "value"},
                "answer_position": answer_position,
                "source_triple": source_triple,
                "extraction_evidence_count": len(evidence),
                "extraction_mode": "typed_evidence_extraction",
            })
        self.last_diagnostics = {
            "raw": len(raw_rows), "accepted": len(rows),
            "focused_evidence_count": len(focused_rows),
            "focus_term_count": len(focus_terms),
            "rejections": dict(sorted(rejections.items())),
        }
        if not rows:
            return None
        return GraphOperation(
            operation_id=operation_id,
            operation_type=OperationType.BRANCH,
            target_id=subgoal_id,
            source_ids=[node.node_id for node in evidence] + dependency_claim_ids,
            branch_id=branch_id,
            payload={"mode": "candidates", "candidates": rows},
            reason="typed_claim_extraction",
            proposed_by="typed_claim_extractor_v2",
            estimated_cost={
                "llm_calls": 1.0,
                "tokens": float(generation.prompt_tokens + generation.completion_tokens),
            },
        )


def _claim_line(claim: ClaimNode) -> str:
    return f"[{claim.node_id}] ({claim.subject}, {claim.relation}, {claim.value})"


def _type(value: Any) -> str:
    text = str(value or "entity").strip().lower().replace("-", "_")
    return text or "entity"


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _cooccurring_sentence(text: str, subject: str, value: str) -> str:
    import re

    normalized_subject = normalize_text(subject)
    normalized_value = normalize_text(value)
    if not normalized_subject or not normalized_value:
        return ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        normalized = normalize_text(sentence)
        if normalized_subject in normalized and normalized_value in normalized:
            return sentence.strip()
    return ""


def _canonicalize_typed_value(value: str, expected_type: str) -> tuple[str, dict[str, str]]:
    """Normalize common structured-evidence scalars without adding semantics.

    The original model tuple remains in ``source_triple`` and the exact evidence
    span remains provenance.  This only removes display attributes from a named
    location or units/context from a typed scalar, so joins operate on an atomic
    endpoint rather than an infobox fragment.
    """
    text = str(value).strip()
    if not text:
        return text, {}
    if expected_type in {
        "location", "city", "country", "state", "province", "region",
        "continent", "river", "mountain",
    }:
        match = re.match(r"^(.+?)\s+-\s+(?:location|located[_ ]in)\b", text, re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip(), {
                "kind": "structured_location_field",
                "original_value": text,
            }
    if expected_type == "distance":
        match = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)\b(?:\s+|$)", text)
        if match and match.group(1) != text:
            return match.group(1), {
                "kind": "typed_scalar",
                "original_value": text,
            }
    return text, {}


def _focus_terms(
    graph: DynamicReasoningHypergraphV2,
    subgoal_id: str,
    question: str,
    dependency_claim_ids: list[str],
) -> set[str]:
    values = [question, graph.question]
    for claim_id in dependency_claim_ids:
        claim = graph.node(claim_id, ClaimNode)
        values.extend([claim.subject, claim.value])
    for row in graph.query_graph.get("constraints", []):
        if str(row.get("subgoal_id")) == subgoal_id:
            values.extend(str(value) for value in row.get("known_entities", []))
    stop = {
        "what", "which", "where", "when", "who", "whose", "how", "the", "a", "an",
        "is", "was", "were", "are", "did", "does", "do", "of", "in", "to", "and",
    }
    return {
        token for value in values for token in re.findall(r"[a-z0-9]+", normalize_text(value))
        if len(token) > 2 and token not in stop
    }


def _focused_span(
    evidence: EvidenceNode,
    focus_terms: set[str],
    config: DynamicV2ResearchConfig,
) -> str:
    sentences = [
        value.strip() for value in re.split(r"(?<=[.!?])\s+|\n+", evidence.source_span)
        if value.strip()
    ]
    if not sentences:
        return evidence.source_span
    scored = []
    title_terms = set(re.findall(r"[a-z0-9]+", normalize_text(evidence.title)))
    for index, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9]+", normalize_text(sentence)))
        score = 2 * len(terms & focus_terms) + len(terms & title_terms)
        scored.append((score, index, sentence))
    cap = min(len(scored), config.extraction_focus_sentences_per_evidence)
    selected_indices = {
        index for _, index, _ in sorted(scored, key=lambda row: (-row[0], row[1]))[:cap]
    }
    selected = " ".join(
        sentence for index, sentence in enumerate(sentences) if index in selected_indices
    )
    if len(selected) < config.extraction_focus_min_chars:
        return evidence.source_span[: max(config.extraction_focus_min_chars, len(selected))]
    return selected
