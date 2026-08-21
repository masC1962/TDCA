from __future__ import annotations

from dataclasses import dataclass

from ..dynamic.graph import CandidateStatus, ClaimNode, GraphOperation, OperationType
from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2


@dataclass(frozen=True)
class RevisionTrigger:
    claim_id: str
    trigger: str
    magnitude: float
    evidence_ids: tuple[str, ...]


class BeliefRevisionDetector:
    """Detect measurable belief collapse without looking at evaluation labels."""

    def __init__(self, config: DynamicV2ResearchConfig) -> None:
        self.config = config

    def detect(self, graph: DynamicReasoningHypergraphV2) -> list[RevisionTrigger]:
        revised = {row.target_claim_id for row in graph.supersession_history}
        rows: list[RevisionTrigger] = []
        for claim in graph.claims():
            if claim.node_id in revised or claim.status != CandidateStatus.COMMITTED:
                continue
            baseline = claim.provenance.metadata.get("commit_belief_baseline", {})
            if not isinstance(baseline, dict) or not baseline:
                continue
            state = graph.belief_states.get(claim.node_id)
            if state is None:
                continue
            support_drop = float(baseline.get("absolute_support", claim.score.absolute_support)) - state.absolute_support
            entropy_rise = state.entropy - float(baseline.get("entropy", state.entropy))
            gap_rise = state.evidence_gap - float(baseline.get("evidence_gap", state.evidence_gap))
            contradiction = state.contradiction_pressure
            candidates = [
                ("contradiction_pressure", contradiction, self.config.contradiction_threshold),
                ("support_collapse", support_drop, self.config.revision_support_drop_threshold),
                ("entropy_rise", entropy_rise, self.config.revision_entropy_rise_threshold),
                ("evidence_gap_rise", gap_rise, self.config.revision_evidence_gap_rise_threshold),
            ]
            triggered = [value for value in candidates if value[1] >= value[2]]
            if triggered:
                name, magnitude, _ = max(triggered, key=lambda value: value[1] - value[2])
                evidence = tuple(dict.fromkeys(
                    evidence_id for linked_id in claim.contradiction_links
                    if linked_id in graph.nodes and isinstance(graph.nodes[linked_id], ClaimNode)
                    for evidence_id in graph.node(linked_id, ClaimNode).evidence_refs
                ))
                rows.append(RevisionTrigger(claim.node_id, name, magnitude, evidence))
        return sorted(rows, key=lambda value: (-value.magnitude, value.claim_id))

    @staticmethod
    def operation(
        graph: DynamicReasoningHypergraphV2,
        trigger: RevisionTrigger,
        branch_id: str,
        operation_id: str,
        *,
        natural: bool = True,
    ) -> GraphOperation:
        claim = graph.node(trigger.claim_id, ClaimNode)
        return GraphOperation(
            operation_id=operation_id,
            operation_type=OperationType.REVISE,
            target_id=claim.target_subgoal,
            source_ids=[trigger.claim_id, *trigger.evidence_ids],
            branch_id=branch_id,
            payload={
                "action": "invalidate_cascade",
                "claim_id": trigger.claim_id,
                "trigger": trigger.trigger,
                "trigger_source": "belief_revision_detector_v2",
                "evidence_ids": list(trigger.evidence_ids),
                "natural": natural,
                "correctness_label": "pending" if natural else "adversarial_expected",
            },
            reason=f"active_revision:{trigger.trigger}",
            proposed_by="belief_revision_detector_v2",
            estimated_cost={"llm_calls": 0.0, "tokens": 0.0},
        )
