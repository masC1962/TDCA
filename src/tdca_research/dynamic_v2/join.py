from __future__ import annotations

import json
from itertools import combinations
from dataclasses import dataclass, field, replace
import re
from typing import Any

from ..budget import Budget
from ..dynamic.graph import (
    BranchStatus,
    CandidateStatus,
    ClaimNode,
    EvidenceNode,
    GraphOperation,
    OperationType,
    SubgoalNode,
)
from ..llm import BaseLLM
from ..utils import estimate_message_tokens, normalize_text, stable_hash
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2
from .query_graph import types_compatible


JOIN_SYSTEM = """Validate one proposed typed conjunctive hypergraph JOIN using only the supplied premise
claims and their grounded evidence. Return JSON only with valid, reason_codes, premise_use,
constraint_satisfaction, and derived_claim. derived_claim must contain subject, relation, value,
subject_type, value_type, derivation_confidence, type_match, dependency_consistency, and qualifiers.
Every premise and declared binding constraint must be necessary for the conclusion; copying one premise
is invalid. Do not use prior knowledge or question-specific rules. A named variable-binding projection
premise may preserve its tuple only when every other premise establishes one of its dependency bindings.
Return valid=false when a type, binding, conjunction, shared-role, or set-intersection constraint fails."""


@dataclass(frozen=True)
class JoinCandidate:
    premise_ids: tuple[str, ...]
    binding: str
    target_subgoal: str
    signature: str
    join_depth: int
    orientation: str = "value_subject"
    open_endpoints: tuple[str, ...] = ("", "")
    projection_premise_id: str = ""
    variable_bindings: dict[str, list[str]] = field(default_factory=dict)
    constraints: tuple[dict[str, Any], ...] = ()
    join_kind: str = "relational_path"
    deterministic_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicJoinDerivation:
    claim: dict[str, Any]
    reason_codes: tuple[str, ...]
    operation_reason: str
    validation_rule: str


@dataclass(frozen=True)
class JoinFeasibilityResult:
    """Pure, zero-cost verdict for failures knowable before allocation."""

    feasible: bool
    reason_codes: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()


class MultiHopJoinEngine:
    def __init__(self, llm: BaseLLM, budget: Budget, config: DynamicV2ResearchConfig) -> None:
        self.llm = llm
        self.budget = budget
        self.config = config
        self.last_diagnostics: dict[str, Any] = {}

    def discover(
        self, graph: DynamicReasoningHypergraphV2, branch_id: str, target_subgoal: str,
    ) -> list[JoinCandidate]:
        claims = _accessible_claims(graph, branch_id)
        existing = {
            semantics.join_signature for semantics in graph.claim_semantics.values()
            if semantics.join_signature
        }
        existing_endpoints = {
            (
                semantics.normalized_subject,
                semantics.normalized_value,
                graph.node(node_id, ClaimNode).target_subgoal,
            )
            for node_id, semantics in graph.claim_semantics.items()
            if semantics.join_depth > 0 and node_id in graph.nodes
        }
        rows: list[JoinCandidate] = []
        link_rows: list[dict[str, Any]] = []
        for left_index, left in enumerate(claims):
            left_semantics = graph.claim_semantics[left.node_id]
            for right in claims[left_index + 1:]:
                if _depends_on(graph, left.node_id, right.node_id) or _depends_on(
                    graph, right.node_id, left.node_id,
                ):
                    continue
                right_semantics = graph.claim_semantics[right.node_id]
                for first, second, orientation, binding, endpoints in _endpoint_unifications(
                    left, left_semantics, right, right_semantics,
                ):
                    constraint = _constraint_row(
                        first, second, orientation, binding, graph,
                    )
                    link_rows.append(constraint)
                    candidate = _build_candidate(
                        graph,
                        (first.node_id, second.node_id),
                        (constraint,),
                        target_subgoal,
                        self.config.max_join_depth,
                    )
                    if candidate is None:
                        continue
                    normalized_endpoints = tuple(
                        normalize_text(value) for value in candidate.open_endpoints
                    )
                    if len(normalized_endpoints) == 2 and (
                        normalized_endpoints[0], normalized_endpoints[1], target_subgoal,
                    ) in existing_endpoints:
                        continue
                    if candidate.signature not in existing:
                        rows.append(candidate)
        # A directly answering child claim may carry an explicit dependency
        # established by the planner and independently checked by the verifier,
        # even when surface aliases prevent endpoint equality. Materialize that
        # dependency as an auditable projection JOIN instead of silently
        # committing or discarding the claim.
        accessible = {row.node_id: row for row in claims}
        for projection in claims:
            if projection.target_subgoal != target_subgoal:
                continue
            for dependency_id in projection.dependency_claim_ids:
                dependency = accessible.get(dependency_id)
                if dependency is None or dependency.node_id == projection.node_id:
                    continue
                constraint = {
                    "left_premise": dependency.node_id,
                    "left_endpoint": "verified_dependency",
                    "right_premise": projection.node_id,
                    "right_endpoint": "declared_dependency",
                    "orientation": "declared_dependency_binding",
                    "binding": dependency.node_id,
                    "left_type": graph.claim_semantics[dependency.node_id].value_type,
                    "right_type": graph.claim_semantics[projection.node_id].subject_type,
                    "type_compatible": True,
                }
                candidate = _build_candidate(
                    graph, (dependency.node_id, projection.node_id),
                    (constraint,), target_subgoal, self.config.max_join_depth,
                )
                if candidate is not None and candidate.signature not in existing:
                    rows.append(candidate)
        rows.extend(_explicit_set_candidates(
            graph,
            claims,
            target_subgoal,
            self.config.max_join_arity,
            self.config.max_join_depth,
            max(0, self.config.max_join_frontier_candidates - len(rows)),
            existing,
        ))
        rows.extend(_numeric_comparison_candidates(
            graph, claims, target_subgoal, self.config.max_join_arity,
            self.config.max_join_depth,
            max(0, self.config.max_join_frontier_candidates - len(rows)),
            existing,
        ))
        # Expand only connected typed frontiers. This is bounded by a global
        # structural cap and never enumerates arbitrary claim combinations.
        frontier = [row for row in rows if len(row.premise_ids) == 2]
        seen_frontiers = {
            (row.premise_ids, _constraint_signature(row.constraints)) for row in frontier
        }
        while frontier and len(rows) < self.config.max_join_frontier_candidates:
            current = frontier.pop(0)
            if len(current.premise_ids) >= self.config.max_join_arity:
                continue
            current_ids = set(current.premise_ids)
            for link in link_rows:
                pair = {str(link["left_premise"]), str(link["right_premise"])}
                overlap = pair & current_ids
                outside = pair - current_ids
                if len(overlap) != 1 or len(outside) != 1:
                    continue
                new_id = next(iter(outside))
                premise_ids = tuple(sorted((*current_ids, new_id)))
                if _derivation_cycle(graph, premise_ids):
                    continue
                constraints = tuple((*current.constraints, link))
                frontier_key = (premise_ids, _constraint_signature(constraints))
                if frontier_key in seen_frontiers:
                    continue
                seen_frontiers.add(frontier_key)
                candidate = _build_candidate(
                    graph, premise_ids, constraints, target_subgoal,
                    self.config.max_join_depth,
                )
                if candidate is None or candidate.signature in existing:
                    continue
                rows.append(candidate)
                frontier.append(candidate)
                if len(rows) >= self.config.max_join_frontier_candidates:
                    break
        unique = {row.signature: row for row in rows}
        candidates = sorted(
            unique.values(),
            key=lambda row: (
                -float(row.deterministic_validation.get("goal_alignment", 0.0)),
                -int(bool(row.projection_premise_id)),
                len(row.premise_ids),
                row.join_depth,
                row.premise_ids,
            ),
        )
        return _dominance_prune(
            candidates,
            preserve_policy_order=self.config.stable_join_frontier_priority,
        )[: self.config.max_join_frontier_candidates]

    def check_feasible(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
    ) -> JoinFeasibilityResult:
        """Reject only inevitable premise failures, without calls or mutation.

        Model-contingent entailment of the *derived* claim remains in
        :meth:`propose`; this predicate merely hoists its existing premise gate
        ahead of computation allocation.
        """
        premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
        projection = next((
            row for row in premises if row.node_id == candidate.projection_premise_id
        ), None)
        unsupported = []
        for row in premises:
            projection_exception = bool(
                row is projection
                and row.score.absolute_support >= self.config.commit_support_threshold
                and row.score.raw.dependency_consistency >= self.config.commit_support_threshold
                and row.score.raw.type_match >= self.config.terminal_min_type_consistency
                and row.score.evidence_gap <= self.config.terminal_max_evidence_gap
                and row.score.raw.contradiction_risk < self.config.terminal_max_contradiction
            )
            if (
                row.status in {
                    CandidateStatus.PROPOSED,
                    CandidateStatus.INVALID,
                    CandidateStatus.ARCHIVED,
                }
                or row.score.absolute_support < self.config.join_min_premise_support
                or row.score.raw.grounding < self.config.join_min_premise_support
                or (
                    row.score.raw.entailment < self.config.join_min_premise_support
                    and not projection_exception
                )
            ):
                unsupported.append(row.node_id)
        if unsupported:
            return JoinFeasibilityResult(
                False,
                ("unsupported_or_unverified_premise",),
                tuple(unsupported),
            )
        return JoinFeasibilityResult(True)

    def propose(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
        operation_id: str,
        token_budget: int | None = None,
    ) -> GraphOperation | None:
        feasibility = self.check_feasible(graph, candidate)
        if not feasibility.feasible:
            self.last_diagnostics = {
                "accepted": False,
                "reason_codes": list(feasibility.reason_codes),
                "premise_ids": list(feasibility.premise_ids),
                "deterministic_validation": candidate.deterministic_validation,
                "preallocation_feasible": False,
            }
            return None
        deterministic = deterministic_join_derivation(
            graph, candidate, self.config,
        )
        if deterministic is not None:
            self.last_diagnostics = {
                "accepted": True,
                "reason_codes": list(deterministic.reason_codes),
            }
            return self._operation(
                graph, candidate, deterministic.claim, operation_id,
                deterministic.operation_reason,
                deterministic.validation_rule,
                {"llm_calls": 0.0, "tokens": 0.0},
            )
        premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
        blocks = []
        for claim in premises:
            evidence = [graph.node(node_id, EvidenceNode) for node_id in claim.evidence_refs]
            source_spans = [
                str(value).strip()
                for value in claim.provenance.metadata.get("source_spans", [])
                if str(value).strip()
            ]
            if not source_spans:
                source_spans = [node.source_span[:500].strip() for node in evidence]
            source_spans = list(dict.fromkeys(source_spans))
            span_budget = min(2000, max(500, self.config.evidence_char_budget // len(premises)))
            grounded = "\n".join(
                f"Grounded span {index}: {value}"
                for index, value in enumerate(source_spans, start=1)
            )[:span_budget]
            blocks.append(
                f"Premise [{claim.node_id}] ({claim.subject}, {claim.relation}, {claim.value}); "
                f"types=({graph.claim_semantics[claim.node_id].subject_type}, "
                f"{graph.claim_semantics[claim.node_id].value_type})\n"
                + f"Evidence IDs: {[node.node_id for node in evidence]}\n{grounded}"
            )
        messages = [
            {"role": "system", "content": JOIN_SYSTEM},
            {"role": "user", "content": (
                f"Root question: {graph.question}\nShared binding: {candidate.binding}\n"
                f"Join orientation: {candidate.orientation}\n"
                f"Required open endpoints: {list(candidate.open_endpoints)}\n"
                f"Join kind: {candidate.join_kind}\n"
                f"Variable bindings: {candidate.variable_bindings}\n"
                f"Typed constraints: {list(candidate.constraints)}\n"
                f"Allowed variable-binding projection premise: "
                f"{candidate.projection_premise_id or 'none'}\n"
                f"Target subgoal: {candidate.target_subgoal}\n\n" + "\n\n".join(blocks)
            )},
        ]
        max_tokens = max(128, min(
            int(token_budget or self.config.join_validation_max_tokens),
            self.config.join_validation_max_tokens,
        ))
        self.budget.require(max_tokens, estimated_prompt_tokens=estimate_message_tokens(messages))
        data, generation = self.llm.generate_json(
            messages, "dynamic_v2_conjunctive_join_validation_v2", max_tokens, self.config.temperature,
        )
        self.budget.record_generation(generation)
        derived = data.get("derived_claim", {})
        valid = bool(data.get("valid", False)) and isinstance(derived, dict)
        normalized_derived = {
            **derived,
            "subject": _scalar_text(derived.get("subject")),
            "relation": _scalar_text(derived.get("relation")),
            "value": _scalar_text(derived.get("value")),
        } if isinstance(derived, dict) else {}
        if not valid or not all(normalized_derived.get(key) for key in ("subject", "relation", "value")):
            self.last_diagnostics = {
                "accepted": False,
                "reason_codes": [str(value) for value in data.get("reason_codes", [])][:5],
            }
            return None
        derived = normalized_derived
        if len(candidate.premise_ids) >= 3:
            premise_use = data.get("premise_use", {})
            used_ids = set(premise_use) if isinstance(premise_use, dict) else {
                str(value) for value in premise_use if isinstance(premise_use, list)
            }
            constraint_rows = data.get("constraint_satisfaction", [])
            constraints_covered = (
                isinstance(constraint_rows, list)
                and len(constraint_rows) >= len(candidate.constraints)
                and all(
                    isinstance(row, dict) and bool(row.get("satisfied", False))
                    for row in constraint_rows
                )
            )
            if used_ids != set(candidate.premise_ids) or not constraints_covered:
                self.last_diagnostics = {
                    "accepted": False,
                    "reason_codes": ["incomplete_nary_premise_or_constraint_use"],
                    "used_premise_ids": sorted(used_ids),
                }
                return None
        derived_endpoints = {
            normalize_text(derived["subject"]), normalize_text(derived["value"]),
        }
        required_endpoints = {normalize_text(value) for value in candidate.open_endpoints}
        projection = next((
            row for row in premises if row.node_id == candidate.projection_premise_id
        ), None)
        projection_endpoints = {
            normalize_text(projection.subject), normalize_text(projection.value),
        } if projection is not None else set()
        projection_match = bool(projection_endpoints) and derived_endpoints == projection_endpoints
        premise_triples = {
            (normalize_text(row.subject), normalize_text(row.relation), normalize_text(row.value))
            for row in premises
        }
        derived_triple = (
            normalize_text(derived["subject"]),
            normalize_text(derived["relation"]),
            normalize_text(derived["value"]),
        )
        endpoint_match = (
            derived_endpoints == required_endpoints
            if len(required_endpoints) <= 2
            else derived_endpoints <= required_endpoints and len(derived_endpoints) == 2
        )
        if not endpoint_match and not projection_match:
            self.last_diagnostics = {"accepted": False, "reason_codes": ["join_endpoint_mismatch"]}
            return None
        if derived_triple in premise_triples and not projection_match:
            self.last_diagnostics = {"accepted": False, "reason_codes": ["degenerate_premise_copy"]}
            return None
        self.last_diagnostics = {
            "accepted": True,
            "reason_codes": [str(value) for value in data.get("reason_codes", [])][:5],
            "premise_use": data.get("premise_use", {}),
            "constraint_satisfaction": data.get("constraint_satisfaction", []),
            "deterministic_validation": candidate.deterministic_validation,
        }
        node_id = f"join_claim_{graph.step + 1}_{candidate.signature[:12]}"
        return self._operation(
            graph, candidate, derived, operation_id,
            "validated_typed_multi_hop_join", "multi_hop_join_engine_v2",
            {
                "llm_calls": 1.0,
                "tokens": float(generation.prompt_tokens + generation.completion_tokens),
            },
        )

    def _operation(
        self, graph, candidate, derived, operation_id, reason, proposed_by, estimated_cost,
    ) -> GraphOperation:
        premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
        node_id = f"join_claim_{graph.step + 1}_{candidate.signature[:12]}"
        edge_id = f"join_edge_{graph.step + 1}_{candidate.signature[:12]}"
        return GraphOperation(
            operation_id=operation_id,
            operation_type=OperationType.MERGE,
            target_id=candidate.target_subgoal,
            source_ids=list(candidate.premise_ids),
            branch_id=_join_branch_id(graph, premises, candidate.projection_premise_id),
            payload={
                "mode": "derive_join",
                "binding": candidate.binding,
                "variable_bindings": candidate.variable_bindings,
                "constraints": [dict(value) for value in candidate.constraints],
                "join_kind": candidate.join_kind,
                "deterministic_validation": candidate.deterministic_validation,
                "join_signature": candidate.signature,
                "validation": {
                    "valid": True,
                    "reason_codes": self.last_diagnostics["reason_codes"],
                },
                "claim": {
                    "node_id": node_id,
                    "edge_id": edge_id,
                    "target_subgoal": candidate.target_subgoal,
                    **derived,
                },
            },
            reason=reason,
            proposed_by=proposed_by,
            estimated_cost=estimated_cost,
        )
    def deterministic_operation(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
        derived_claim: dict[str, Any],
        operation_id: str,
    ) -> GraphOperation:
        """Offline-test constructor; production joins always use `propose`."""
        signature = candidate.signature
        return GraphOperation(
            operation_id, OperationType.MERGE, candidate.target_subgoal,
            list(candidate.premise_ids), _join_branch_id(
                graph,
                [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids],
                candidate.projection_premise_id,
            ),
            {
                "mode": "derive_join", "binding": candidate.binding,
                "variable_bindings": candidate.variable_bindings,
                "constraints": [dict(value) for value in candidate.constraints],
                "join_kind": candidate.join_kind,
                "deterministic_validation": candidate.deterministic_validation,
                "join_signature": signature,
                "validation": {"valid": True, "reason_codes": ["offline_fixture"]},
                "claim": {
                    "node_id": f"join_claim_{graph.step + 1}_{signature[:12]}",
                    "edge_id": f"join_edge_{graph.step + 1}_{signature[:12]}",
                    "target_subgoal": candidate.target_subgoal,
                    **derived_claim,
                },
            },
            "validated_typed_multi_hop_join", "multi_hop_join_engine_v2",
            {"llm_calls": 0.0, "tokens": 0.0},
        )

    def predicted_provider_calls(
        self,
        graph: DynamicReasoningHypergraphV2,
        candidate: JoinCandidate,
    ) -> int:
        """Return the exact provider demand of the current JOIN state."""
        if not self.check_feasible(graph, candidate).feasible:
            return 1
        return int(
            deterministic_join_derivation(graph, candidate, self.config) is None
        )


def deterministic_join_derivation(
    graph: DynamicReasoningHypergraphV2,
    candidate: JoinCandidate,
    config: DynamicV2ResearchConfig,
) -> DeterministicJoinDerivation | None:
    """Materialize the provider-free derivation promised at allocation time."""
    premises = [graph.node(node_id, ClaimNode) for node_id in candidate.premise_ids]
    numeric = _deterministic_numeric_comparison_claim(graph, candidate, premises)
    if numeric is not None:
        return DeterministicJoinDerivation(
            numeric,
            (
                "explicit_query_comparison",
                "independently_verified_numeric_premises",
                "deterministic_arg_projection",
                "no_additional_generation",
            ),
            "query_constrained_numeric_comparison",
            "deterministic_numeric_comparison_join_v21",
        )
    projection = next((
        row for row in premises if row.node_id == candidate.projection_premise_id
    ), None)
    if projection is not None:
        other_support = min(
            row.score.absolute_support
            for row in premises if row.node_id != projection.node_id
        )
        if (
            projection.score.absolute_support >= config.commit_support_threshold
            and other_support >= config.commit_support_threshold
            and projection.score.raw.dependency_consistency
            >= config.commit_support_threshold
            and projection.score.raw.type_match >= config.commit_support_threshold
        ):
            semantics = graph.claim_semantics[projection.node_id]
            return DeterministicJoinDerivation(
                {
                    "subject": projection.subject,
                    "relation": projection.relation,
                    "value": projection.value,
                    "subject_type": semantics.subject_type,
                    "value_type": semantics.value_type,
                    "derivation_confidence": min(
                        row.score.absolute_support for row in premises
                    ),
                    "type_match": projection.score.raw.type_match,
                    "dependency_consistency": (
                        projection.score.raw.dependency_consistency
                    ),
                    "qualifiers": {
                        "projection_premise_id": projection.node_id,
                        "validation": "independent_raw_scoring",
                        "join_kind": candidate.join_kind,
                    },
                },
                (
                    "exact_dependency_binding",
                    "independent_verifier_projection",
                    "no_additional_generation",
                ),
                "independently_verified_variable_binding_projection",
                "deterministic_projection_join_v2",
            )
    symbolic = (
        _deterministic_path_claim(graph, candidate, premises, config)
        if config.deterministic_goal_path_join else None
    )
    if symbolic is not None:
        return DeterministicJoinDerivation(
            symbolic,
            (
                "canonical_entity_unification",
                "independently_verified_path_premises",
                "goal_aligned_symbolic_composition",
                "no_additional_generation",
            ),
            "goal_aligned_canonical_path_composition",
            "deterministic_path_join_v21",
        )
    return None


def join_candidate_from_operation(operation: GraphOperation) -> JoinCandidate:
    """Reconstruct the sealed JOIN candidate from its controller operation."""
    payload = operation.payload
    return JoinCandidate(
        tuple(str(value) for value in payload["premise_ids"]),
        str(payload["binding"]), operation.target_id,
        str(payload["join_signature"]), int(payload["join_depth"]),
        str(payload.get("orientation", "value_subject")),
        tuple(str(value) for value in payload.get("open_endpoints", [])),
        str(payload.get("projection_premise_id", "")),
        {
            str(key): [str(item) for item in value]
            for key, value in payload.get("variable_bindings", {}).items()
        },
        tuple(dict(value) for value in payload.get("constraints", [])),
        str(payload.get("join_kind", "relational_path")),
        dict(payload.get("deterministic_validation", {})),
    )


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    return str(value).strip()


def _join_branch_id(
    graph: DynamicReasoningHypergraphV2,
    premises: list[ClaimNode],
    projection_premise_id: str = "",
) -> str:
    """Attach a JOIN to its live child branch, never an archived ancestor.

    Accessible proof premises may legitimately come from an ancestor branch.
    Using the first sorted premise therefore loses the derived state after a
    branch split.  At most one live lineage is accessible to a discovery call;
    fail closed if corrupted input exposes multiple active branches.
    """
    active = sorted({
        claim.branch_id for claim in premises
        if claim.branch_id in graph.branches
        and graph.branches[claim.branch_id].status == BranchStatus.ACTIVE
    })
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        raise ValueError("JOIN premises span multiple active branches")
    projection = next((
        claim for claim in premises if claim.node_id == projection_premise_id
    ), None)
    if projection is not None:
        return projection.branch_id
    return premises[0].branch_id


def _compatible(left: str, right: str) -> bool:
    return types_compatible(left, right)


def _accessible_claims(
    graph: DynamicReasoningHypergraphV2, branch_id: str,
) -> list[ClaimNode]:
    """Return branch-local claims plus the assigned dependency lineage."""
    local = {
        claim.node_id: claim for claim in graph.claims(branch_id=branch_id)
        if claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
        and claim.node_id in graph.claim_semantics
    }
    branch = graph.branches.get(branch_id)
    queue = list((branch.assignments if branch else {}).values())
    seen = set(queue)
    while queue:
        node_id = queue.pop()
        node = graph.nodes.get(node_id)
        if not isinstance(node, ClaimNode):
            continue
        if (
            node.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
            and node.node_id in graph.claim_semantics
        ):
            local[node.node_id] = node
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in seen:
                seen.add(dependency_id)
                queue.append(dependency_id)
    return sorted(local.values(), key=lambda value: value.node_id)


def _endpoint_unifications(left, left_semantics, right, right_semantics):
    """Enumerate valid length-two relational paths without relation-name rules."""
    rows = []
    if (
        _same_entity(
            left_semantics.normalized_value, left_semantics.canonical_value_id,
            right_semantics.normalized_subject, right_semantics.canonical_subject_id,
        )
        and _compatible(left_semantics.value_type, right_semantics.subject_type)
    ):
        rows.append((
            left, right, "value_subject", left_semantics.normalized_value,
            (left.subject, right.value),
        ))
    if (
        _same_entity(
            right_semantics.normalized_value, right_semantics.canonical_value_id,
            left_semantics.normalized_subject, left_semantics.canonical_subject_id,
        )
        and _compatible(right_semantics.value_type, left_semantics.subject_type)
    ):
        rows.append((
            right, left, "value_subject", right_semantics.normalized_value,
            (right.subject, left.value),
        ))
    if (
        _same_entity(
            left_semantics.normalized_subject, left_semantics.canonical_subject_id,
            right_semantics.normalized_subject, right_semantics.canonical_subject_id,
        )
        and _compatible(left_semantics.subject_type, right_semantics.subject_type)
    ):
        rows.append((
            left, right, "shared_subject", left_semantics.normalized_subject,
            (left.value, right.value),
        ))
    if (
        _same_entity(
            left_semantics.normalized_value, left_semantics.canonical_value_id,
            right_semantics.normalized_value, right_semantics.canonical_value_id,
        )
        and _compatible(left_semantics.value_type, right_semantics.value_type)
    ):
        rows.append((
            left, right, "shared_value", left_semantics.normalized_value,
            (left.subject, right.subject),
        ))
    return [row for row in rows if row[3] and normalize_text(row[4][0]) != normalize_text(row[4][1])]


def _same_entity(
    left_text: str, left_id: str, right_text: str, right_id: str,
) -> bool:
    if left_id and right_id and left_id == right_id:
        return True
    return bool(left_text and right_text and left_text == right_text)


def _deterministic_path_claim(
    graph: DynamicReasoningHypergraphV2,
    candidate: JoinCandidate,
    premises: list[ClaimNode],
    config: DynamicV2ResearchConfig,
) -> dict[str, Any] | None:
    if candidate.join_kind not in {"relational_path", "conjunctive_relational_path"}:
        return None
    if len(candidate.open_endpoints) != 2:
        return None
    # A target-local claim alone contributes only 0.20.  Requiring an actual
    # query anchor (0.45) prevents structurally valid but question-irrelevant
    # paths from consuming the proof frontier.
    if float(candidate.deterministic_validation.get("goal_alignment", 0.0)) < 0.45:
        return None
    projection = next((
        row for row in premises if row.node_id == candidate.projection_premise_id
    ), None)
    if config.join_requires_verified_projection_premise and projection is None:
        return None
    if any(
        row.score.absolute_support < config.commit_support_threshold
        or row.score.raw.grounding < config.commit_support_threshold
        or row.score.raw.entailment < config.commit_support_threshold
        or row.score.raw.type_match < config.commit_support_threshold
        for row in premises
    ):
        return None
    left, right = candidate.open_endpoints
    if config.join_requires_verified_projection_premise:
        projected_output = normalize_text(projection.value)
        if normalize_text(left) == projected_output:
            left, right = right, left
        elif normalize_text(right) != projected_output:
            return None
    constraint = next((
        row for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == candidate.target_subgoal
    ), {})
    known = {normalize_text(value) for value in constraint.get("known_entities", [])}
    if normalize_text(right) in known and normalize_text(left) not in known:
        left, right = right, left
    left_type = _endpoint_type(graph, premises, left)
    right_type = _endpoint_type(graph, premises, right)
    return {
        "subject": left,
        "relation": "composed_path:" + " -> ".join(row.relation for row in premises),
        "value": right,
        "subject_type": left_type,
        "value_type": right_type,
        "derivation_confidence": min(row.score.absolute_support for row in premises),
        "type_match": min(row.score.raw.type_match for row in premises),
        "dependency_consistency": min(
            row.score.raw.dependency_consistency for row in premises
        ),
        "qualifiers": {
            "validation": "canonical_entity_unification_and_independent_raw_scoring",
            "proof_premise_ids": list(candidate.premise_ids),
            "join_kind": candidate.join_kind,
        },
    }


def _endpoint_type(
    graph: DynamicReasoningHypergraphV2, premises: list[ClaimNode], endpoint: str,
) -> str:
    normalized = normalize_text(endpoint)
    for claim in premises:
        semantics = graph.claim_semantics[claim.node_id]
        if semantics.normalized_subject == normalized:
            return semantics.subject_type
        if semantics.normalized_value == normalized:
            return semantics.value_type
    return "entity"


def _constraint_row(
    left: ClaimNode,
    right: ClaimNode,
    orientation: str,
    binding: str,
    graph: DynamicReasoningHypergraphV2,
) -> dict[str, Any]:
    endpoints = {
        "value_subject": ("value", "subject"),
        "shared_subject": ("subject", "subject"),
        "shared_value": ("value", "value"),
    }[orientation]
    left_semantics = graph.claim_semantics[left.node_id]
    right_semantics = graph.claim_semantics[right.node_id]
    left_type = (
        left_semantics.value_type if endpoints[0] == "value" else left_semantics.subject_type
    )
    right_type = (
        right_semantics.value_type if endpoints[1] == "value" else right_semantics.subject_type
    )
    return {
        "left_premise": left.node_id,
        "left_endpoint": endpoints[0],
        "right_premise": right.node_id,
        "right_endpoint": endpoints[1],
        "orientation": orientation,
        "binding": normalize_text(binding),
        "left_type": left_type,
        "right_type": right_type,
        "type_compatible": _compatible(left_type, right_type),
    }


def _build_candidate(
    graph: DynamicReasoningHypergraphV2,
    premise_ids: tuple[str, ...],
    constraints: tuple[dict[str, Any], ...],
    target_subgoal: str,
    max_join_depth: int,
) -> JoinCandidate | None:
    premise_ids = tuple(sorted(dict.fromkeys(premise_ids)))
    if len(premise_ids) < 2 or _derivation_cycle(graph, premise_ids):
        return None
    covered = {
        str(value[key])
        for value in constraints
        for key in ("left_premise", "right_premise")
    }
    if covered != set(premise_ids) or not _constraints_connected(premise_ids, constraints):
        return None
    if any(not bool(value.get("type_compatible", False)) for value in constraints):
        return None
    semantics = [graph.claim_semantics[node_id] for node_id in premise_ids]
    depth = max(value.join_depth for value in semantics) + 1
    if depth > max_join_depth:
        return None
    used_endpoints = {
        (str(value["left_premise"]), str(value["left_endpoint"])) for value in constraints
    } | {
        (str(value["right_premise"]), str(value["right_endpoint"])) for value in constraints
    }
    endpoint_rows: list[tuple[str, str]] = []
    for node_id in premise_ids:
        claim = graph.node(node_id, ClaimNode)
        for endpoint, raw_value in (("subject", claim.subject), ("value", claim.value)):
            if (node_id, endpoint) not in used_endpoints:
                endpoint_rows.append((normalize_text(raw_value), raw_value))
    open_endpoints = tuple(dict.fromkeys(
        raw for normalized, raw in endpoint_rows if normalized
    ))
    projection = _projection_premise_many(graph, premise_ids, target_subgoal)
    if len(open_endpoints) < 2 and projection:
        claim = graph.node(projection, ClaimNode)
        open_endpoints = (claim.subject, claim.value)
    if len(open_endpoints) < 2:
        return None
    bindings: dict[str, list[str]] = {}
    for constraint in constraints:
        binding = str(constraint["binding"])
        key = f"?v_{stable_hash(binding)[:8]}"
        bindings.setdefault(key, []).extend([
            binding,
            f"{constraint['left_premise']}.{constraint['left_endpoint']}",
            f"{constraint['right_premise']}.{constraint['right_endpoint']}",
        ])
    bindings = {
        key: list(dict.fromkeys(values)) for key, values in sorted(bindings.items())
    }
    orientations = {str(value["orientation"]) for value in constraints}
    if _has_explicit_set_semantics(graph, premise_ids):
        join_kind = "set_intersection"
    elif len(premise_ids) >= 3 and orientations <= {"shared_subject", "shared_value"}:
        join_kind = "shared_role_conjunction"
    elif len(premise_ids) >= 3:
        join_kind = "conjunctive_relational_path"
    elif orientations <= {"shared_subject", "shared_value"}:
        join_kind = "shared_role"
    else:
        join_kind = "relational_path"
    signature_payload = {
        "premises": list(premise_ids),
        "constraints": _constraint_signature(constraints),
        "open_endpoints": [normalize_text(value) for value in open_endpoints],
        "projection_premise_id": projection,
        "join_kind": join_kind,
        "target": target_subgoal,
    }
    signature = stable_hash(signature_payload)
    independent_support = {
        node_id: float(graph.node(node_id, ClaimNode).score.absolute_support)
        for node_id in premise_ids
    }
    intersection_members = sorted(set.intersection(*(
        _qualifier_members(graph.claim_semantics[node_id].qualifiers)
        for node_id in premise_ids
    ))) if join_kind == "set_intersection" else []
    goal_alignment = _goal_alignment(graph, target_subgoal, premise_ids, open_endpoints)
    return JoinCandidate(
        premise_ids=premise_ids,
        binding=";".join(sorted({str(value["binding"]) for value in constraints})),
        target_subgoal=target_subgoal,
        signature=signature,
        join_depth=depth,
        orientation=(
            str(constraints[0]["orientation"])
            if len(constraints) == 1 else "conjunctive"
        ),
        open_endpoints=open_endpoints,
        projection_premise_id=projection,
        variable_bindings=bindings,
        constraints=tuple(dict(value) for value in constraints),
        join_kind=join_kind,
        deterministic_validation={
            "connected": True,
            "acyclic_derivation": True,
            "all_premises_constrained": True,
            "types_compatible": True,
            "independent_support": independent_support,
            "set_intersection_members": intersection_members,
            "goal_alignment": goal_alignment,
        },
    )


def _constraints_connected(
    premise_ids: tuple[str, ...], constraints: tuple[dict[str, Any], ...],
) -> bool:
    adjacency = {node_id: set() for node_id in premise_ids}
    for row in constraints:
        left, right = str(row["left_premise"]), str(row["right_premise"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = {premise_ids[0]}
    queue = [premise_ids[0]]
    while queue:
        for node_id in adjacency.get(queue.pop(), set()):
            if node_id not in seen:
                seen.add(node_id)
                queue.append(node_id)
    return seen == set(premise_ids)


def _constraint_signature(constraints: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(sorted(stable_hash({
        key: row.get(key) for key in (
            "left_premise", "left_endpoint", "right_premise", "right_endpoint",
            "orientation", "binding", "left_type", "right_type",
        )
    }) for row in constraints))


def _derivation_cycle(
    graph: DynamicReasoningHypergraphV2, premise_ids: tuple[str, ...],
) -> bool:
    premise_set = set(premise_ids)
    for node_id in premise_ids:
        semantics = graph.claim_semantics[node_id]
        if semantics.join_depth <= 0:
            continue
        if (_derivation_ancestors(graph, node_id) - {node_id}) & premise_set:
            return True
    return False


def _derivation_ancestors(
    graph: DynamicReasoningHypergraphV2, claim_id: str,
) -> set[str]:
    closure = {claim_id}
    queue = [claim_id]
    while queue:
        node = graph.nodes.get(queue.pop())
        if not isinstance(node, ClaimNode):
            continue
        semantics = graph.claim_semantics.get(node.node_id)
        if semantics is None or semantics.join_depth <= 0:
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in closure:
                closure.add(dependency_id)
                queue.append(dependency_id)
    return closure


def _projection_premise_many(
    graph: DynamicReasoningHypergraphV2,
    premise_ids: tuple[str, ...],
    target_subgoal: str,
) -> str:
    premise_set = set(premise_ids)
    for node_id in premise_ids:
        candidate = graph.node(node_id, ClaimNode)
        if (
            candidate.target_subgoal != target_subgoal
            or not candidate.dependency_claim_ids
            or not _verified_answer_projection(graph, candidate)
        ):
            continue
        dependency_lineage: set[str] = set()
        for dependency_id in candidate.dependency_claim_ids:
            dependency_lineage.update(_dependency_closure(graph, dependency_id))
        others = premise_set - {node_id}
        if others and all(
            other in dependency_lineage
            for other in others
        ):
            return node_id
    return ""


def _verified_answer_projection(
    graph: DynamicReasoningHypergraphV2,
    claim: ClaimNode,
    seen: set[str] | None = None,
) -> bool:
    """Require an independently verified output projection for JOIN copying."""
    seen = set(seen or ())
    if claim.node_id in seen:
        return False
    seen.add(claim.node_id)
    semantics = graph.claim_semantics[claim.node_id]
    if semantics.join_depth == 0:
        return bool(claim.provenance.metadata.get("answers_subgoal", False))
    projection_id = str(semantics.qualifiers.get("projection_premise_id", ""))
    projection = graph.nodes.get(projection_id)
    return isinstance(projection, ClaimNode) and _verified_answer_projection(
        graph, projection, seen,
    )


def _has_explicit_set_semantics(
    graph: DynamicReasoningHypergraphV2, premise_ids: tuple[str, ...],
) -> bool:
    markers = {"set_members", "set_semantics", "collection_members"}
    return all(
        bool(markers & set(graph.claim_semantics[node_id].qualifiers))
        for node_id in premise_ids
    )


def _qualifier_members(qualifiers: dict[str, str]) -> set[str]:
    values: list[Any] = []
    for key in ("set_members", "set_semantics", "collection_members"):
        if key not in qualifiers:
            continue
        raw: Any = qualifiers[key]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = [
                    value.strip().strip("'\"")
                    for value in raw.strip("[](){} ").replace(";", ",").replace("|", ",").split(",")
                ]
        values.extend(raw if isinstance(raw, list) else [raw])
    return {normalize_text(str(value)) for value in values if normalize_text(str(value))}


def _explicit_set_candidates(
    graph: DynamicReasoningHypergraphV2,
    claims: list[ClaimNode],
    target_subgoal: str,
    max_arity: int,
    max_depth: int,
    cap: int,
    existing: set[str],
) -> list[JoinCandidate]:
    """Enumerate only premise groups indexed by a real shared set member."""
    if cap <= 0:
        return []
    by_member: dict[str, list[str]] = {}
    for claim in claims:
        members = _qualifier_members(graph.claim_semantics[claim.node_id].qualifiers)
        for member in members:
            by_member.setdefault(member, []).append(claim.node_id)
    rows: list[JoinCandidate] = []
    seen: set[str] = set()
    for member, node_ids in sorted(by_member.items()):
        unique_ids = sorted(set(node_ids))
        for arity in range(2, min(max_arity, len(unique_ids)) + 1):
            for premise_group in combinations(unique_ids, arity):
                if _derivation_cycle(graph, premise_group):
                    continue
                anchor = premise_group[0]
                constraints = tuple({
                    "left_premise": anchor,
                    "left_endpoint": "set_members",
                    "right_premise": node_id,
                    "right_endpoint": "set_members",
                    "orientation": "set_intersection",
                    "binding": member,
                    "left_type": "set_member",
                    "right_type": "set_member",
                    "type_compatible": True,
                } for node_id in premise_group[1:])
                candidate = _build_candidate(
                    graph, premise_group, constraints, target_subgoal, max_depth,
                )
                if candidate is None or candidate.signature in existing or candidate.signature in seen:
                    continue
                if not candidate.deterministic_validation.get("set_intersection_members"):
                    continue
                seen.add(candidate.signature)
                rows.append(candidate)
                if len(rows) >= cap:
                    return rows
    return rows


def _comparison_mode(question: str) -> str:
    tokens = set(normalize_text(question).split())
    maximize = tokens & {
        "more", "most", "higher", "highest", "larger", "largest",
        "greater", "greatest", "longer", "longest",
    }
    minimize = tokens & {
        "less", "least", "lower", "lowest", "smaller", "smallest",
        "fewer", "fewest", "shorter", "shortest",
    }
    if bool(maximize) == bool(minimize):
        return ""
    return "argmax" if maximize else "argmin"


def _numeric_value(value: str) -> float | None:
    match = re.match(r"^\s*([+-]?\d+(?:[.,]\d+)?)\b", str(value))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _numeric_comparison_candidates(
    graph: DynamicReasoningHypergraphV2,
    claims: list[ClaimNode],
    target_subgoal: str,
    max_arity: int,
    max_depth: int,
    cap: int,
    existing: set[str],
) -> list[JoinCandidate]:
    """Create bounded argmax/argmin JOINs only for explicit comparison queries."""
    mode = _comparison_mode(graph.node(target_subgoal, SubgoalNode).question_template)
    if not mode or cap <= 0:
        return []
    groups: dict[str, list[ClaimNode]] = {}
    for claim in claims:
        if claim.target_subgoal != target_subgoal or _numeric_value(claim.value) is None:
            continue
        groups.setdefault(normalize_text(claim.relation), []).append(claim)
    rows: list[JoinCandidate] = []
    for relation, group in sorted(groups.items()):
        unique = {normalize_text(row.subject): row for row in group if normalize_text(row.subject)}
        ordered = sorted(unique.values(), key=lambda row: row.node_id)
        for arity in range(2, min(max_arity, len(ordered)) + 1):
            for premise_rows in combinations(ordered, arity):
                premise_ids = tuple(row.node_id for row in premise_rows)
                anchor = premise_ids[0]
                constraints = tuple({
                    "left_premise": anchor,
                    "left_endpoint": "value",
                    "right_premise": node_id,
                    "right_endpoint": "value",
                    "orientation": "numeric_comparison",
                    "binding": relation,
                    "left_type": "number",
                    "right_type": "number",
                    "type_compatible": True,
                } for node_id in premise_ids[1:])
                base = _build_candidate(
                    graph, premise_ids, constraints, target_subgoal, max_depth,
                )
                if base is None:
                    continue
                numeric = {
                    row.node_id: float(_numeric_value(row.value)) for row in premise_rows
                }
                best_value = (max if mode == "argmax" else min)(numeric.values())
                selected = sorted(
                    node_id for node_id, value in numeric.items() if value == best_value
                )
                if len(selected) != 1:
                    continue
                signature = stable_hash({
                    "premises": sorted(premise_ids), "target": target_subgoal,
                    "join_kind": f"numeric_{mode}", "relation": relation,
                })
                if signature in existing:
                    continue
                candidate = replace(
                    base,
                    signature=signature,
                    join_kind=f"numeric_{mode}",
                    deterministic_validation={
                        **base.deterministic_validation,
                        "numeric_values": numeric,
                        "selected_premise_id": selected[0],
                        "comparison_mode": mode,
                        "goal_alignment": max(
                            0.8, float(base.deterministic_validation.get("goal_alignment", 0.0)),
                        ),
                    },
                )
                rows.append(candidate)
                if len(rows) >= cap:
                    return rows
    return rows


def _deterministic_numeric_comparison_claim(
    graph: DynamicReasoningHypergraphV2,
    candidate: JoinCandidate,
    premises: list[ClaimNode],
) -> dict[str, Any] | None:
    if candidate.join_kind not in {"numeric_argmax", "numeric_argmin"}:
        return None
    selected_id = str(candidate.deterministic_validation.get("selected_premise_id", ""))
    selected = next((row for row in premises if row.node_id == selected_id), None)
    if selected is None:
        return None
    semantics = graph.claim_semantics[selected.node_id]
    return {
        "subject": "comparison:" + candidate.target_subgoal,
        "relation": f"{candidate.join_kind}:{normalize_text(selected.relation)}",
        "value": selected.subject,
        "subject_type": "comparison",
        "value_type": semantics.subject_type,
        "derivation_confidence": min(row.score.absolute_support for row in premises),
        "type_match": min(row.score.raw.type_match for row in premises),
        "dependency_consistency": min(
            row.score.raw.dependency_consistency for row in premises
        ),
        "qualifiers": {
            "validation": "explicit_query_numeric_comparison_and_independent_raw_scoring",
            "proof_premise_ids": list(candidate.premise_ids),
            "numeric_values": candidate.deterministic_validation.get("numeric_values", {}),
            "join_kind": candidate.join_kind,
        },
    }


def _dominance_prune(
    candidates: list[JoinCandidate], *, preserve_policy_order: bool = False,
) -> list[JoinCandidate]:
    """Remove only structurally equivalent candidates; never drop extra constraints."""
    best: dict[tuple[Any, ...], tuple[int, JoinCandidate]] = {}
    for index, row in enumerate(candidates):
        key = (
            row.target_subgoal,
            row.premise_ids,
            tuple(sorted(normalize_text(value) for value in row.open_endpoints)),
            _constraint_signature(row.constraints),
        )
        current = best.get(key)
        if current is None:
            best[key] = (index, row)
        elif (row.join_depth, row.signature) < (
            current[1].join_depth, current[1].signature,
        ):
            # Keep the structural group's original position even if a later
            # representative is canonically preferable.
            best[key] = (current[0], row)
    if preserve_policy_order:
        return [row for _, row in sorted(best.values(), key=lambda value: value[0])]
    return sorted((row for _, row in best.values()), key=lambda row: (
        -float(row.deterministic_validation.get("goal_alignment", 0.0)),
        -int(bool(row.projection_premise_id)),
        len(row.premise_ids),
        row.join_depth,
        row.signature,
    ))


def _goal_alignment(
    graph: DynamicReasoningHypergraphV2,
    target_subgoal: str,
    premise_ids: tuple[str, ...],
    open_endpoints: tuple[str, ...],
) -> float:
    constraint = next((
        row for row in graph.query_graph.get("constraints", [])
        if str(row.get("subgoal_id")) == target_subgoal
    ), {})
    known = {
        normalize_text(value) for value in constraint.get("known_entities", [])
        if normalize_text(value)
    }
    endpoint_values = {normalize_text(value) for value in open_endpoints if normalize_text(value)}
    known_anchor = 1.0 if known & endpoint_values else 0.0
    input_variables = set(str(value) for value in constraint.get("input_variables", []))
    dependency_subgoals = {
        graph.node(node_id, ClaimNode).target_subgoal for node_id in premise_ids
    }
    dependency_coverage = 1.0 if any(
        variable.removeprefix("?answer:") in dependency_subgoals
        for variable in input_variables
    ) else 0.0
    target_claim = any(
        graph.node(node_id, ClaimNode).target_subgoal == target_subgoal
        for node_id in premise_ids
    )
    return min(1.0, 0.45 * known_anchor + 0.35 * dependency_coverage + 0.20 * target_claim)


def _depends_on(
    graph: DynamicReasoningHypergraphV2, claim_id: str, possible_ancestor_id: str,
) -> bool:
    semantics = graph.claim_semantics.get(claim_id)
    if semantics is None or semantics.join_depth <= 0:
        return False
    queue = list(graph.node(claim_id, ClaimNode).dependency_claim_ids)
    seen = set(queue)
    while queue:
        node_id = queue.pop()
        if node_id == possible_ancestor_id:
            return True
        node = graph.nodes.get(node_id)
        if not isinstance(node, ClaimNode):
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in seen:
                seen.add(dependency_id)
                queue.append(dependency_id)
    return False


def _projection_premise(
    graph: DynamicReasoningHypergraphV2,
    left: ClaimNode,
    right: ClaimNode,
    target_subgoal: str,
) -> str:
    for candidate, other in ((left, right), (right, left)):
        if candidate.target_subgoal != target_subgoal or not candidate.dependency_claim_ids:
            continue
        if set(candidate.dependency_claim_ids) & _dependency_closure(graph, other.node_id):
            return candidate.node_id
    return ""


def _dependency_closure(
    graph: DynamicReasoningHypergraphV2, claim_id: str,
) -> set[str]:
    closure = {claim_id}
    queue = [claim_id]
    while queue:
        node = graph.nodes.get(queue.pop())
        if not isinstance(node, ClaimNode):
            continue
        for dependency_id in node.dependency_claim_ids:
            if dependency_id not in closure:
                closure.add(dependency_id)
                queue.append(dependency_id)
    return closure
