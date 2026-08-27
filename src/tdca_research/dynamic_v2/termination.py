from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import asdict
import math

from ..budget import Budget
from ..dynamic.graph import AnswerStatus, CandidateStatus, ClaimNode, GraphOperation
from ..utils import normalize_text
from .allocator import ComputationPacket
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2, TerminalBeliefState, TerminationKind
from .proof import audit_graph_proof, claim_closure
from .obligations import dead_end_certificate
from .transitions import certified_transition_value


class TerminalBeliefReadout:
    """Deterministic answer-slot competition over separate belief channels.

    No LLM judges the final answer here.  Premise verification remains
    independent, JOIN validity remains independently audited, and acceptance is
    a conjunction rather than a fused probability threshold.
    """

    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config

    def evaluate(
        self,
        graph: DynamicReasoningHypergraphV2,
        operations: list[GraphOperation],
        unresolved_branch_ids: list[str] | None = None,
    ) -> tuple[list[GraphOperation], list[dict]]:
        candidates = [
            self._candidate(graph, operation)
            for operation in operations
            if operation.payload.get("mode") == "answer"
        ]
        if not candidates:
            return [], []

        # Relative terminal weight is answer-set state only.  It is derived from
        # absolute support and never fed back into any premise's raw score.
        by_value: dict[str, float] = {}
        unresolved_branch_ids = sorted(set(unresolved_branch_ids or []))
        for row in candidates:
            key = normalize_text(row["profile"].candidate_answer)
            by_value[key] = max(by_value.get(key, 0.0), row["profile"].absolute_support)
        total = sum(by_value.values())
        if total <= 0:
            weights = {key: 1.0 / len(by_value) for key in by_value}
        else:
            weights = {key: value / total for key, value in by_value.items()}
        ranked = sorted(weights, key=lambda key: (-weights[key], key))
        top_key = ranked[0]
        top_margin = (
            weights[ranked[0]] - weights[ranked[1]] if len(ranked) > 1 else 1.0
        )
        competition_entropy = _normalized_entropy(list(weights.values()))
        competition_rows = [
            {
                "normalized_answer": key,
                "absolute_support": float(by_value[key]),
            }
            for key in sorted(by_value)
        ]

        accepted: list[GraphOperation] = []
        diagnostics: list[dict] = []
        for row in candidates:
            profile = row["profile"]
            value_key = normalize_text(profile.candidate_answer)
            profile.relative_weight = _unit(weights[value_key])
            profile.relative_margin = _unit(top_margin if value_key == top_key else 0.0)
            profile.competition_entropy = competition_entropy
            profile.entropy = competition_entropy
            unresolved_competition = any(
                self._credible_unresolved_branch(graph, branch_id, row["operation"].target_id)
                for branch_id in unresolved_branch_ids
                if branch_id != profile.branch_id
            )
            profile.rejection_reasons = self._rejection_reasons(
                profile, value_key == top_key, unresolved_competition,
            )
            profile.accepted = not profile.rejection_reasons
            profile.terminal_gap = self._terminal_gap(profile)
            operation = row["operation"]
            answer = operation.payload["answer"]
            answer["confidence"] = profile.absolute_support
            answer["supporting_claims"] = list(profile.supporting_claims)
            answer["supporting_evidence"] = list(profile.supporting_evidence)
            answer["terminal_belief"] = asdict(profile)
            if self.config.certified_terminal_materialization:
                # Preserve the complete answer-slot competition so the stop
                # policy can independently reconstruct the relative channels.
                # Gold answers are never part of this state.
                answer["terminal_competition"] = {
                    "candidate_values": deepcopy(competition_rows),
                    "unresolved_branch_ids": list(unresolved_branch_ids),
                }
            diagnostics.append(asdict(profile))
            if profile.accepted:
                accepted.append(operation)
        accepted.sort(key=lambda operation: (
            -float(operation.payload["answer"]["terminal_belief"]["relative_weight"]),
            -float(operation.payload["answer"]["terminal_belief"]["absolute_support"]),
            str(operation.payload["answer"]["candidate_answer"]),
            operation.operation_id,
        ))
        return accepted, diagnostics

    def _credible_unresolved_branch(
        self,
        graph: DynamicReasoningHypergraphV2,
        branch_id: str,
        target_subgoal: str,
    ) -> bool:
        """Return whether an unfinished branch contains a viable answer rival.

        A merely open search branch is not evidence against a complete proof.
        It blocks terminal readout only after it has produced an independently
        scored answer projection that passes the same non-relative safety
        channels used by the terminal gate.  Unknown branch IDs remain blocking
        so incomplete external callers fail conservatively.
        """
        branch = graph.branches.get(branch_id)
        if branch is None:
            return True
        for claim in graph.claims():
            if claim.target_subgoal != target_subgoal or claim.branch_id != branch_id:
                continue
            if claim.status not in {
                CandidateStatus.SCORED,
                CandidateStatus.RETAINED,
                CandidateStatus.REVISED,
                CandidateStatus.COMMITTED,
            }:
                continue
            semantics = graph.claim_semantics.get(claim.node_id)
            is_projection = bool(
                claim.provenance.metadata.get("answers_subgoal", False)
                or (
                    semantics is not None
                    and semantics.qualifiers.get("projection_premise_id")
                )
            )
            if not is_projection:
                continue
            if all((
                claim.score.absolute_support >= self.config.terminal_min_absolute_support,
                claim.score.evidence_gap <= self.config.terminal_max_evidence_gap,
                claim.score.raw.grounding >= self.config.join_min_premise_support,
                claim.score.raw.type_match >= self.config.terminal_min_type_consistency,
                claim.score.raw.contradiction_risk < self.config.terminal_max_contradiction,
                bool(claim.evidence_refs),
                not self.config.query_conditioned_semantic_alignment or all((
                    claim.score.raw.relation_target_alignment
                    >= self.config.terminal_min_relation_target_alignment,
                    claim.score.raw.subject_binding_coverage
                    >= self.config.terminal_min_subject_binding_coverage,
                    claim.score.raw.dependency_binding_coverage
                    >= self.config.terminal_min_dependency_binding_coverage,
                    claim.score.raw.qualifier_coverage
                    >= self.config.terminal_min_qualifier_coverage,
                    claim.score.raw.output_slot_coverage
                    >= self.config.terminal_min_output_slot_coverage,
                    claim.score.raw.full_subgoal_coverage
                    >= self.config.terminal_min_full_subgoal_coverage,
                )),
            )):
                return True
        return False

    def _candidate(
        self, graph: DynamicReasoningHypergraphV2, source: GraphOperation,
    ) -> dict:
        operation = deepcopy(source)
        answer = operation.payload["answer"]
        initial_claims = [str(value) for value in answer.get("supporting_claims", [])]
        claim_ids = claim_closure(graph, initial_claims)
        claims = [graph.node(node_id, ClaimNode) for node_id in claim_ids]
        answer_claims = [
            graph.node(node_id, ClaimNode) for node_id in initial_claims
            if node_id in graph.nodes
        ]
        evidence_ids = list(dict.fromkeys(
            evidence_id for claim in claims for evidence_id in claim.evidence_refs
            if evidence_id in graph.nodes
        ))
        raw_channels = {
            claim.node_id: {
                "absolute_support": float(claim.score.absolute_support),
                "relative_weight": float(claim.score.relative_weight),
                "entropy": float(claim.score.set_entropy),
                "evidence_gap": float(claim.score.evidence_gap),
                "grounding": float(claim.score.raw.grounding),
                "entailment": float(claim.score.raw.entailment),
                "type_match": float(claim.score.raw.type_match),
                "dependency_consistency": float(claim.score.raw.dependency_consistency),
                "retrieval_support": float(claim.score.raw.retrieval_support),
                "contradiction_risk": float(claim.score.raw.contradiction_risk),
                "raw_model_confidence": float(claim.score.raw.raw_model_confidence),
                "relation_target_alignment": float(claim.score.raw.relation_target_alignment),
                "subject_binding_coverage": float(claim.score.raw.subject_binding_coverage),
                "dependency_binding_coverage": float(claim.score.raw.dependency_binding_coverage),
                "qualifier_coverage": float(claim.score.raw.qualifier_coverage),
                "output_slot_coverage": float(claim.score.raw.output_slot_coverage),
                "full_subgoal_coverage": float(claim.score.raw.full_subgoal_coverage),
            }
            for claim in claims
        }
        absolute_support = min(
            (claim.score.absolute_support for claim in claims), default=0.0,
        )
        evidence_gap = max((claim.score.evidence_gap for claim in claims), default=1.0)
        contradiction = max((
            max(
                claim.score.raw.contradiction_risk,
                graph.belief_states.get(claim.node_id).contradiction_pressure
                if graph.belief_states.get(claim.node_id) is not None else 0.0,
            )
            for claim in claims
        ), default=1.0)
        proof_audit = audit_graph_proof(
            graph, operation.target_id, operation.branch_id, initial_claims,
        )
        chain_coverage = proof_audit.dependency_coverage
        proof_connected = proof_audit.proof_connected
        statuses_valid = all(
            claim.status not in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED}
            for claim in claims
        )
        evidence_complete = bool(claims) and all(
            claim.evidence_refs
            and all(evidence_id in graph.nodes for evidence_id in claim.evidence_refs)
            for claim in claims
        )
        sufficient_chain = (
            bool(normalize_text(str(answer.get("candidate_answer", ""))))
            and statuses_valid and evidence_complete and proof_connected
            and chain_coverage >= self.config.terminal_min_chain_coverage
        )
        alignment_names = (
            "relation_target_alignment", "subject_binding_coverage",
            "dependency_binding_coverage", "qualifier_coverage",
            "output_slot_coverage", "full_subgoal_coverage",
        )
        alignment = (
            {
                name: min(
                    (float(getattr(claim.score.raw, name)) for claim in answer_claims),
                    default=0.0,
                )
                for name in alignment_names
            }
            if self.config.query_conditioned_semantic_alignment
            else {name: 0.0 for name in alignment_names}
        )
        alignment_thresholds = {
            "relation_target_alignment": self.config.terminal_min_relation_target_alignment,
            "subject_binding_coverage": self.config.terminal_min_subject_binding_coverage,
            "dependency_binding_coverage": self.config.terminal_min_dependency_binding_coverage,
            "qualifier_coverage": self.config.terminal_min_qualifier_coverage,
            "output_slot_coverage": self.config.terminal_min_output_slot_coverage,
            "full_subgoal_coverage": self.config.terminal_min_full_subgoal_coverage,
        }
        alignment_gaps = (
            {
                name: _below_gap(alignment[name], threshold)
                for name, threshold in alignment_thresholds.items()
            }
            if self.config.query_conditioned_semantic_alignment
            else {}
        )
        profile = TerminalBeliefState(
            answer_node_id=str(answer.get("node_id", "")),
            candidate_answer=str(answer.get("candidate_answer", "")).strip(),
            branch_id=operation.branch_id,
            absolute_support=_unit(absolute_support),
            relative_weight=0.0,
            # Terminal entropy is computed over distinct answer-slot values in
            # `evaluate`; premise-level entropies remain in raw_claim_channels.
            entropy=0.0,
            competition_entropy=0.0,
            evidence_gap=_unit(evidence_gap),
            relative_margin=0.0,
            contradiction_pressure=_unit(contradiction),
            answer_type_consistency=_unit(answer.get("answer_type_consistency", 0.0)),
            chain_coverage=_unit(chain_coverage),
            terminal_gap=1.0,
            proof_depth=max((
                graph.claim_semantics[claim.node_id].join_depth for claim in claims
            ), default=0),
            supporting_claims=claim_ids,
            supporting_evidence=evidence_ids,
            raw_claim_channels=raw_channels,
            sufficient_chain=sufficient_chain,
            accepted=False,
            **alignment,
            query_alignment_gaps=alignment_gaps,
            scoring_version=(
                "hara-separate-query-alignment-terminal-v2.4.3.19"
                if self.config.query_conditioned_semantic_alignment
                else "terminal-belief-readout-v2.2"
            ),
        )
        return {"operation": operation, "profile": profile}

    def _rejection_reasons(
        self, profile: TerminalBeliefState, is_top_value: bool,
        unresolved_competition: bool,
    ) -> list[str]:
        reasons = []
        if unresolved_competition:
            reasons.append("unresolved_competing_branches")
        if not profile.sufficient_chain:
            reasons.append("insufficient_support_chain")
        if profile.absolute_support < self.config.terminal_min_absolute_support:
            reasons.append("absolute_support_below_minimum")
        if not is_top_value or profile.relative_margin < self.config.terminal_min_relative_margin:
            reasons.append("relative_margin_below_minimum")
        if profile.entropy > self.config.terminal_max_entropy:
            reasons.append("claim_set_entropy_above_maximum")
        if profile.evidence_gap > self.config.terminal_max_evidence_gap:
            reasons.append("evidence_gap_above_maximum")
        if profile.contradiction_pressure >= self.config.terminal_max_contradiction:
            reasons.append("contradiction_pressure_above_maximum")
        if profile.answer_type_consistency < self.config.terminal_min_type_consistency:
            reasons.append("answer_type_consistency_below_minimum")
        if profile.chain_coverage < self.config.terminal_min_chain_coverage:
            reasons.append("chain_coverage_below_minimum")
        if self.config.query_conditioned_semantic_alignment:
            thresholds = {
                "relation_target_alignment": self.config.terminal_min_relation_target_alignment,
                "subject_binding_coverage": self.config.terminal_min_subject_binding_coverage,
                "dependency_binding_coverage": self.config.terminal_min_dependency_binding_coverage,
                "qualifier_coverage": self.config.terminal_min_qualifier_coverage,
                "output_slot_coverage": self.config.terminal_min_output_slot_coverage,
                "full_subgoal_coverage": self.config.terminal_min_full_subgoal_coverage,
            }
            for name, threshold in thresholds.items():
                if float(getattr(profile, name)) < float(threshold):
                    reasons.append(f"{name}_below_minimum")
        return reasons

    def _terminal_gap(self, profile: TerminalBeliefState) -> float:
        cfg = self.config
        gaps = [
            _below_gap(profile.absolute_support, cfg.terminal_min_absolute_support),
            _below_gap(profile.relative_margin, cfg.terminal_min_relative_margin),
            _above_gap(profile.entropy, cfg.terminal_max_entropy),
            _above_gap(profile.evidence_gap, cfg.terminal_max_evidence_gap),
            _above_gap(profile.contradiction_pressure, cfg.terminal_max_contradiction),
            _below_gap(profile.answer_type_consistency, cfg.terminal_min_type_consistency),
            _below_gap(profile.chain_coverage, cfg.terminal_min_chain_coverage),
            0.0 if profile.sufficient_chain else 1.0,
            1.0 if "unresolved_competing_branches" in profile.rejection_reasons else 0.0,
        ]
        if self.config.query_conditioned_semantic_alignment:
            gaps.extend(profile.query_alignment_gaps.values())
        return _unit(max(gaps))


@dataclass(frozen=True)
class MetaDecision:
    outcome: TerminationKind
    reason: str
    best_predicted_evc: float
    answer_node_id: str | None = None
    selected_allocation_id: str | None = None
    dead_end_certificate: dict = field(default_factory=dict)


class MetaStopPolicy:
    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config

    def decide(
        self,
        graph: DynamicReasoningHypergraphV2,
        packets: list[ComputationPacket],
        budget: Budget,
    ) -> MetaDecision:
        answer = self._supported_answer(graph)
        if answer is not None:
            return MetaDecision(
                TerminationKind.ANSWER, "accepted_graph_grounded_answer",
                max((row.predicted_evc for row in packets), default=0.0), answer.node_id,
            )
        if self.config.certified_meta_stop:
            return self._certified_decide(graph, packets, budget)
        if not packets:
            return MetaDecision(TerminationKind.ABSTAIN, "no_executable_computation", 0.0)
        best = packets[0]
        if not self._affordable(best, graph, budget):
            return MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED,
                "positive_value_computation_exceeds_remaining_budget",
                best.predicted_evc,
            )
        if best.predicted_evc <= self.config.meta_stop_evc_threshold:
            return MetaDecision(
                TerminationKind.ABSTAIN,
                "best_expected_value_not_above_marginal_cost_threshold",
                best.predicted_evc,
            )
        return MetaDecision(TerminationKind.CONTINUE, "positive_expected_value", best.predicted_evc)

    def _certified_decide(
        self,
        graph: DynamicReasoningHypergraphV2,
        packets: list[ComputationPacket],
        budget: Budget,
    ) -> MetaDecision:
        remaining = {
            "llm_calls": max(0, budget.max_llm_calls - budget.usage.llm_calls),
            "tokens": max(0, budget.max_total_tokens - budget.usage.total_tokens),
            "retrieval_calls": max(
                0, graph.limits.max_retrieval_calls - graph.retrieval_calls,
            ),
            "graph_operations": max(
                0, graph.limits.max_graph_operations - len(graph.operation_history),
            ),
        }
        certificate = dead_end_certificate(graph, packets, remaining)
        if not packets:
            exhausted = any(
                remaining[name] <= 0
                for name in ("llm_calls", "tokens", "retrieval_calls", "graph_operations")
            ) and bool(certificate["open_obligations"])
            return MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED if exhausted else TerminationKind.ABSTAIN,
                (
                    "proof_obligations_blocked_by_budget"
                    if exhausted else "no_executable_computation_with_certificate"
                ),
                0.0,
                dead_end_certificate=certificate,
            )
        affordable = [row for row in packets if self._affordable(row, graph, budget)]
        if not affordable:
            best_gross = max(row.predicted_gross_opportunity for row in packets)
            certificate["decision_layer"] = "feasibility_before_net_evc"
            return MetaDecision(
                TerminationKind.BUDGET_EXHAUSTED,
                "positive_gross_proof_opportunity_is_unaffordable",
                max(row.predicted_evc for row in packets),
                dead_end_certificate=certificate,
            )
        if self.config.certified_transition_option_value:
            certified = []
            for row in affordable:
                transition = certified_transition_value(graph, row.operation, self.config)
                if (
                    bool(transition.get("mandatory", False))
                    and bool(transition.get("deterministic", False))
                    and int(transition.get("provider_calls", -1)) == 0
                    and int(row.requested_budget.get("llm_calls", 0)) == 0
                ):
                    certified.append((row, transition))
            if certified:
                selected, transition = max(certified, key=lambda value: (
                    float(value[1].get("predicted_transition_value", 0.0)),
                    value[0].predicted_evc,
                    value[0].allocation_id,
                ))
                certificate["decision_layer"] = "certified_transition_before_net_evc"
                certificate["selected_transition_certificate"] = transition
                return MetaDecision(
                    TerminationKind.CONTINUE,
                    "certified_state_transition",
                    selected.predicted_evc,
                    selected_allocation_id=selected.allocation_id,
                    dead_end_certificate=certificate,
                )
        best = max(affordable, key=lambda row: (
            row.predicted_evc,
            row.predicted_gross_opportunity,
            -row.predicted_normalized_cost,
            row.allocation_id,
        ))
        if best.predicted_evc <= self.config.meta_stop_evc_threshold:
            certificate["decision_layer"] = "net_evc_after_feasibility"
            return MetaDecision(
                TerminationKind.ABSTAIN,
                "affordable_proof_opportunity_below_net_value_threshold",
                best.predicted_evc,
                selected_allocation_id=best.allocation_id,
                dead_end_certificate=certificate,
            )
        return MetaDecision(
            TerminationKind.CONTINUE,
            "affordable_positive_graph_local_expected_value",
            best.predicted_evc,
            selected_allocation_id=best.allocation_id,
        )

    def _supported_answer(self, graph: DynamicReasoningHypergraphV2):
        valid = []
        for answer in graph.answers():
            if answer.status != AnswerStatus.ACCEPTED or answer.derivation_edge in graph.invalidated_hyperedges:
                continue
            claims = [graph.node(node_id, ClaimNode) for node_id in answer.supporting_claims]
            if any(claim.status in {CandidateStatus.INVALID, CandidateStatus.ARCHIVED} for claim in claims):
                continue
            if not answer.supporting_evidence:
                continue
            if graph.terminal_readout_version:
                terminal = graph.terminal_beliefs.get(answer.node_id)
                if terminal is None or not terminal.accepted or terminal.rejection_reasons:
                    continue
                # Recheck the independent channels at decision time so a stale
                # or hand-crafted accepted flag can never bypass the hard gate.
                if self._terminal_belief_rejected(terminal):
                    continue
            valid.append(answer)
        return max(valid, key=lambda value: (
            graph.terminal_beliefs[value.node_id].relative_weight
            if value.node_id in graph.terminal_beliefs else 0.0,
            value.confidence, value.node_id,
        ), default=None)

    def _terminal_belief_rejected(self, terminal: TerminalBeliefState) -> bool:
        rejected = any((
            not terminal.sufficient_chain,
            terminal.absolute_support < self.config.terminal_min_absolute_support,
            terminal.relative_margin < self.config.terminal_min_relative_margin,
            terminal.entropy > self.config.terminal_max_entropy,
            terminal.evidence_gap > self.config.terminal_max_evidence_gap,
            terminal.contradiction_pressure >= self.config.terminal_max_contradiction,
            terminal.answer_type_consistency < self.config.terminal_min_type_consistency,
            terminal.chain_coverage < self.config.terminal_min_chain_coverage,
        ))
        if rejected or not self.config.query_conditioned_semantic_alignment:
            return rejected
        return any((
            terminal.relation_target_alignment < self.config.terminal_min_relation_target_alignment,
            terminal.subject_binding_coverage < self.config.terminal_min_subject_binding_coverage,
            terminal.dependency_binding_coverage < self.config.terminal_min_dependency_binding_coverage,
            terminal.qualifier_coverage < self.config.terminal_min_qualifier_coverage,
            terminal.output_slot_coverage < self.config.terminal_min_output_slot_coverage,
            terminal.full_subgoal_coverage < self.config.terminal_min_full_subgoal_coverage,
        ))

    @staticmethod
    def _affordable(
        packet: ComputationPacket,
        graph: DynamicReasoningHypergraphV2,
        budget: Budget,
    ) -> bool:
        request = packet.requested_budget
        if request.get("max_tokens", 0) > 0 and not budget.can_call(request["max_tokens"]):
            return False
        if packet.operation.operation_type.value == "RETRIEVE" and graph.retrieval_calls >= graph.limits.max_retrieval_calls:
            return False
        return len(graph.operation_history) < graph.limits.max_graph_operations


def _normalized_entropy(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    value = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
    return _unit(value / math.log(len(weights)))


def _below_gap(value: float, minimum: float) -> float:
    if minimum <= 0 or value >= minimum:
        return 0.0
    return _unit((minimum - value) / minimum)


def _above_gap(value: float, maximum: float) -> float:
    if value <= maximum:
        return 0.0
    if maximum >= 1:
        return 1.0
    return _unit((value - maximum) / (1.0 - maximum))


def _unit(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
