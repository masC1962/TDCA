from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
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
When one evidence sentence contains an explicit parenthetical or comma-separated
enumeration of same-kind entities, emit a separate atomic claim for every relevant
member up to the claim cap. Shared list membership is evidence, not by itself a
direct answer: set answer_position=none unless the sentence explicitly states the
requested relation.
Do not use prior knowledge, compare candidates, compose multiple facts, reverse the evidence direction, or
merge sentences. Return an empty list when evidence is insufficient."""


DIRECT_ANSWER_EXTRACTION_SYSTEM = """Project only evidence-grounded candidates for the requested subgoal
output variable. Return JSON only as {claims:[...]}, using the same exact compact claim schema: subject,
relation, value, subject_type, value_type, evidence_ids, extraction_confidence, answer_position, and quote.
Every returned row must set answer_position to subject or value, and that exact endpoint must answer the
subgoal after applying the supplied dependency claims. Preserve answer-defining surface modifiers such as
percentages, quantities, ranges, dates, negation, and collection membership; do not shorten a quantified
answer to its head noun. Prefer the shortest evidence substring that is still a complete answer. Do not
return true bridge or context facts that fail to fill the requested output variable. Do not use prior
knowledge, compose unsupported facts, or add prose. Return an empty list when no supplied evidence directly
supports an output candidate."""


CONSTRAINT_AWARE_DIRECT_ANSWER_EXTRACTION_SYSTEM = DIRECT_ANSWER_EXTRACTION_SYSTEM + """
Interpret the wh-slot independently from entities already named in the subgoal. A named destination, source,
comparison object, or other fixed endpoint is an input constraint and must not be returned as the output merely
because it has the expected broad type. For a when question, project the explicit date or year as the output even
when it is written as an adverbial event modifier. For a where question with a named destination, project a
distinct explicitly stated intermediate place only when the evidence states that it fills the requested wh-slot.
Keep the bound dependency entity as the opposite endpoint whenever the evidence permits an atomic binary claim."""


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
        focus_mode: str = "coverage",
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
        system_prompt = (
            (
                CONSTRAINT_AWARE_DIRECT_ANSWER_EXTRACTION_SYSTEM
                if self.config.constraint_aware_direct_projection
                else DIRECT_ANSWER_EXTRACTION_SYSTEM
            )
            if focus_mode == "direct_answer" else TYPED_EXTRACTION_SYSTEM
        )
        user_prefix = (
            f"Root question: {graph.question}\nSubgoal: {instantiated_question}\n"
            f"Expected answer type: {graph.node(subgoal_id, SubgoalNode).answer_type}\n"
            f"Unresolved query constraint: {query_constraint}\n"
            f"Extraction objective: {focus_mode}\n"
            f"Claim cap: {cap}\nDependency claims:\n{dependencies}\nEvidence:\n"
        )
        context = _budget_aware_context(
            focused_rows,
            self.config.evidence_char_budget,
            self.budget,
            system_prompt,
            user_prefix,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prefix + context},
        ]
        max_tokens = max(128, min(
            int(token_budget or self.config.typed_extraction_max_tokens),
            self.config.typed_extraction_max_tokens,
        ))
        estimated_prompt_tokens = estimate_message_tokens(messages)
        max_tokens = _fit_completion_to_remaining_budget(
            self.budget, max_tokens, estimated_prompt_tokens,
        )
        self.budget.require(
            max_tokens, estimated_prompt_tokens=estimated_prompt_tokens,
        )
        data, generation = self.llm.generate_json(
            messages,
            (
                "dynamic_v2_goal_conditioned_answer_projection_v1"
                if focus_mode == "direct_answer" else "dynamic_v2_typed_claim_extraction_v1"
            ),
            max_tokens, self.config.temperature,
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
            if focus_mode == "direct_answer" and answer_position == "none":
                rejections["not_an_output_projection"] += 1
                continue
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
                "extraction_focus_mode": focus_mode,
            })
        if self.config.grounded_numeric_interval_consolidation:
            rows = _consolidate_grounded_numeric_intervals(rows)
        if self.config.deterministic_enumeration_expansion and len(rows) < cap:
            rows.extend(_expand_grounded_enumerations(
                rows, available, existing, cap - len(rows),
                graph.step + 1, subgoal_id,
            ))
        self.last_diagnostics = {
            "raw": len(raw_rows), "accepted": len(rows),
            "focused_evidence_count": len(focused_rows),
            "focus_term_count": len(focus_terms),
            "focused_context_characters": len(context),
            "budget_compacted_context": (
                len(context) < len(bounded_context(
                    focused_rows, self.config.evidence_char_budget,
                ))
            ),
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
            reason=(
                "goal_conditioned_answer_projection"
                if focus_mode == "direct_answer" else "typed_claim_extraction"
            ),
            proposed_by=(
                "goal_conditioned_typed_projector_v22"
                if focus_mode == "direct_answer" else "typed_claim_extractor_v2"
            ),
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


def _fit_completion_to_remaining_budget(
    budget: Budget,
    requested_completion: int,
    estimated_prompt_tokens: int,
    minimum_completion: int = 128,
) -> int:
    """Use the largest schema-safe completion that fits the live token budget.

    The allocator's packet is an upper bound.  Late in a reasoning trajectory,
    the exact focused prompt can be larger than its coarse estimate; failing the
    whole question while a smaller valid JSON completion still fits defeats
    adaptive computation.  If even the schema-safe minimum cannot fit, retain
    the request so ``Budget.require`` raises the normal audited exhaustion.
    """
    reserve = max(0, int(budget.final_reserve_tokens))
    available = (
        int(budget.max_total_tokens)
        - int(budget.usage.total_tokens)
        - max(0, int(estimated_prompt_tokens))
        - reserve
    )
    if available < minimum_completion:
        return int(requested_completion)
    return min(int(requested_completion), available)


def _budget_aware_context(
    blocks: list[str],
    configured_max_characters: int,
    budget: Budget,
    system_prompt: str,
    user_prefix: str,
    minimum_completion: int = 128,
) -> str:
    """Compact evidence only when the exact late-stage prompt would not fit."""
    remaining = (
        int(budget.max_total_tokens)
        - int(budget.usage.total_tokens)
        - max(0, int(budget.final_reserve_tokens))
        - int(minimum_completion)
    )
    # ``estimate_message_tokens`` uses ceil(characters / 3) plus eight tokens
    # per message.  Leave one extra token for integer rounding.
    prompt_character_cap = max(0, 3 * (remaining - 17))
    fixed_characters = len(system_prompt) + len(user_prefix)
    context_cap = min(
        int(configured_max_characters),
        max(0, prompt_character_cap - fixed_characters),
    )
    context = bounded_context(blocks, context_cap)
    # ``bounded_context`` preserves blank-line separators between evidence
    # blocks, so enforce the budget against the assembled messages as the final
    # authority instead of relying only on the character algebra above.
    while context:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prefix + context},
        ]
        estimated = estimate_message_tokens(messages)
        if budget.can_call(
            minimum_completion, estimated_prompt_tokens=estimated,
        ):
            break
        over = (
            int(budget.usage.total_tokens)
            + int(budget.final_reserve_tokens)
            + int(minimum_completion)
            + estimated
            - int(budget.max_total_tokens)
        )
        context = context[:max(0, len(context) - max(16, 3 * over + 3))]
    return context


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


_EXPLICIT_NUMERIC_RANGE = re.compile(
    r"(?<![\w.])(?:"
    r"between\s+(?P<between_left>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<between_left_unit>%|percent(?:age)?s?)?\s+and\s+"
    r"(?P<between_right>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<between_right_unit>%|percent(?:age)?s?)?"
    r"|(?P<left>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<left_unit>%|percent(?:age)?s?)?\s*"
    r"(?:to|through|until|[-–—])\s*"
    r"(?P<right>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"(?P<right_unit>%|percent(?:age)?s?)?"
    r")(?![\w.])",
    re.IGNORECASE,
)
_NUMERIC_VALUE_TYPES = {
    "number", "numerical", "numeric", "count", "quantity", "fraction",
    "percentage", "percent", "decimal", "ratio",
}


def _consolidate_grounded_numeric_intervals(
    accepted_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover an explicitly stated interval split into scalar endpoints.

    Consolidation is deliberately evidence-exact.  Endpoint rows must agree on
    subject, relation, dependency claims, evidence references, source spans,
    and answer projection.  Their numeric values must then match the two ends
    of one literal range in that shared span.  No interval is inferred from
    merely co-occurring numbers.
    """
    rows = list(accepted_rows)
    consumed: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if canonical_type(row.get("value_type")) not in _NUMERIC_VALUE_TYPES:
            continue
        key = (
            normalize_text(row.get("subject", "")),
            normalize_text(row.get("relation", "")),
            tuple(sorted(str(value) for value in row.get("dependency_claim_ids", []))),
            tuple(sorted(str(value) for value in row.get("evidence_refs", []))),
            tuple(normalize_text(value) for value in row.get("source_spans", [])),
            str(row.get("answer_position", "none")),
        )
        groups[key].append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        shared_spans = rows[indices[0]].get("source_spans", [])
        for span in shared_spans:
            for match in _EXPLICIT_NUMERIC_RANGE.finditer(str(span)):
                left_text = match.group("left") or match.group("between_left")
                right_text = match.group("right") or match.group("between_right")
                percent_range = bool(
                    match.group("left_unit") or match.group("right_unit")
                    or match.group("between_left_unit") or match.group("between_right_unit")
                )
                left = _range_endpoint_decimal(left_text, percent_range)
                right = _range_endpoint_decimal(right_text, percent_range)
                if left is None or right is None or left == right:
                    continue
                left_rows = [
                    index for index in indices
                    if index not in consumed and _row_numeric_decimal(rows[index]) == left
                ]
                right_rows = [
                    index for index in indices
                    if index not in consumed and _row_numeric_decimal(rows[index]) == right
                ]
                if len(left_rows) != 1 or len(right_rows) != 1:
                    continue
                left_index, right_index = left_rows[0], right_rows[0]
                if left_index == right_index:
                    continue
                first_index = min(left_index, right_index)
                endpoint_rows = [rows[left_index], rows[right_index]]
                consolidated = deepcopy(rows[first_index])
                surface = match.group(0).strip()
                consolidated["value"] = surface
                consolidated["value_type"] = "percentage" if percent_range else "number"
                consolidated["answer_type"] = consolidated["value_type"]
                consolidated["source_triple"] = {
                    "subject": str(consolidated.get("subject", "")),
                    "relation": str(consolidated.get("relation", "")),
                    "value": surface,
                }
                consolidated["extraction_confidence"] = min(
                    float(row.get("extraction_confidence", 0.0)) for row in endpoint_rows
                )
                consolidated["extraction_mode"] = "grounded_numeric_interval_consolidation"
                consolidated["qualifiers"] = {
                    **dict(consolidated.get("qualifiers", {})),
                    "numeric_interval_consolidation": {
                        "surface": surface,
                        "endpoint_values": [
                            str(rows[left_index].get("value", "")),
                            str(rows[right_index].get("value", "")),
                        ],
                        "endpoint_claim_ids": [
                            str(rows[left_index].get("node_id", "")),
                            str(rows[right_index].get("node_id", "")),
                        ],
                        "evidence_exact": True,
                    },
                }
                consumed.update({left_index, right_index})
                replacements[first_index] = consolidated

    return [
        replacements[index] if index in replacements else row
        for index, row in enumerate(rows)
        if index not in consumed or index in replacements
    ]


def _row_numeric_decimal(row: dict[str, Any]) -> Decimal | None:
    text = str(row.get("value", "")).strip().lower().replace(",", "")
    is_percent = text.endswith("%") or bool(re.search(r"\bpercent(?:age)?s?$", text))
    text = re.sub(r"(?:%|\s*percent(?:age)?s?)$", "", text).strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value / Decimal(100) if is_percent else value


def _range_endpoint_decimal(text: str, percent_range: bool) -> Decimal | None:
    try:
        value = Decimal(str(text).replace(",", ""))
    except InvalidOperation:
        return None
    return value / Decimal(100) if percent_range else value


def _expand_grounded_enumerations(
    accepted_rows: list[dict[str, Any]],
    evidence_by_id: dict[str, EvidenceNode],
    existing_signatures: set[tuple[str, str, str]],
    remaining_cap: int,
    creation_step: int,
    subgoal_id: str,
) -> list[dict[str, Any]]:
    """Atomize explicit same-kind enumerations anchored by an accepted claim.

    This is deliberately conservative: expansion requires a parenthetical list
    containing the accepted value and at least two members with the same lexical
    head (for example ``Gmina A, Gmina B``).  It never invents a relation or marks
    a sibling as directly answering the subgoal; later JOIN validation owns any
    compositional conclusion.
    """
    if remaining_cap <= 0:
        return []
    expanded: list[dict[str, Any]] = []
    for row in list(accepted_rows):
        anchor = str(row.get("value", "")).strip()
        for evidence_id in row.get("evidence_refs", []):
            evidence = evidence_by_id.get(str(evidence_id))
            if evidence is None:
                continue
            for sibling in _enumerated_sibling_values(evidence.source_span, anchor):
                if normalize_text(sibling) == normalize_text(row.get("subject", "")):
                    # Reusing the source entity as the enumerated object creates
                    # a reflexive claim that the passage never states.
                    continue
                signature = (
                    normalize_text(row.get("subject", "")),
                    normalize_text(row.get("relation", "")),
                    normalize_text(sibling),
                )
                if not all(signature) or signature in existing_signatures:
                    continue
                existing_signatures.add(signature)
                candidate = deepcopy(row)
                candidate["node_id"] = (
                    f"claim_v2_{creation_step}_{subgoal_id}_enum_{len(expanded) + 1}"
                )
                candidate["value"] = sibling
                candidate["source_spans"] = [evidence.source_span]
                candidate["evidence_refs"] = [evidence.node_id]
                candidate["answers_subgoal"] = False
                candidate["answer_position"] = "none"
                candidate["source_triple"] = {
                    "subject": str(row.get("subject", "")),
                    "relation": str(row.get("relation", "")),
                    "value": sibling,
                }
                candidate["extraction_confidence"] = min(
                    float(row.get("extraction_confidence", 0.0)), 0.95,
                )
                candidate["extraction_mode"] = "deterministic_enumeration_expansion"
                candidate["qualifiers"] = {
                    **dict(row.get("qualifiers", {})),
                    "enumeration_anchor": anchor,
                }
                expanded.append(candidate)
                if len(expanded) >= remaining_cap:
                    return expanded
    return expanded


def _enumerated_sibling_values(text: str, anchor: str) -> list[str]:
    normalized_anchor = normalize_text(anchor)
    anchor_head = normalize_text(anchor).split(" ", 1)[0] if normalized_anchor else ""
    if not anchor_head:
        return []
    siblings: list[str] = []
    for group in re.findall(r"\(([^()]{3,500})\)", str(text)):
        parts = [
            re.sub(r"^(?:and|or)\s+", "", value.strip(), flags=re.IGNORECASE)
            for value in re.split(r"\s*[,;]\s*|\s+and\s+", group)
        ]
        parts = [value.strip(" .") for value in parts if 1 < len(value.strip()) <= 100]
        normalized_parts = [normalize_text(value) for value in parts]
        if normalized_anchor not in normalized_parts:
            continue
        same_head = [
            value for value, normalized in zip(parts, normalized_parts)
            if normalized.split(" ", 1)[0] == anchor_head
        ]
        if len(same_head) < 2:
            continue
        siblings.extend(
            value for value in same_head if normalize_text(value) != normalized_anchor
        )
    return list(dict.fromkeys(siblings))


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
