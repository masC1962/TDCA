from __future__ import annotations

from dataclasses import replace

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import OperationType
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.query_graph import canonical_type, types_compatible
from tdca_research.dynamic_v2.termination import TerminalBeliefReadout
from tdca_research.dynamic_v2.verifier import (
    MultiSampleIndependentVerifier,
    _query_conditioned_signals,
    _relation_target_certificate,
    _structural_dependency_binding_coverage,
)
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Usage
from tdca_research.dynamic.scoring import fuse_candidate_scores
from tdca_research.dynamic.graph import VerificationSignals

from tests_research.test_dynamic_v2_core import (
    chain_graph,
    config,
    empty_graph,
    operation,
    terminal_operation,
)


class CapturingMock(DeterministicMockLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []

    def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
        self.requests.append((schema_name, messages))
        return super().generate_json(messages, schema_name, max_tokens, temperature)


def _alignment_row(candidate_id: str, *, relation: float = 0.95) -> dict:
    return {
        "candidate_id": candidate_id,
        "grounding": 1.0,
        "entailment": 0.95,
        "type_match": 1.0,
        "dependency_consistency": 0.1,
        "contradiction_risk": 0.0,
        "raw_model_confidence": 0.95,
        "relation_target_alignment": relation,
        "subject_binding_coverage": 0.2,
        "dependency_binding_coverage": 0.1,
        "qualifier_coverage": 1.0,
        "output_slot_coverage": 0.2,
        "answer_position": "none",
        "contradiction_candidate_ids": [],
    }


def test_hara_is_a_native_compatible_method_name():
    cfg = DynamicV2ResearchConfig(
        method="hara", llm_backend="mock", max_total_tokens=6000,
        final_reserve_tokens=200,
    )
    assert cfg.method == "hara"
    assert cfg.query_conditioned_semantic_alignment is False


def test_hara_fields_do_not_change_frozen_payload_schema_when_disabled():
    _, _, graph = chain_graph()
    payload = graph.to_dict()
    assert "query_alignment_version" not in payload
    for node in payload["nodes"].values():
        raw = node.get("score", {}).get("raw", {})
        assert "relation_target_alignment" not in raw
        assert "full_subgoal_coverage" not in raw
    for terminal in payload["terminal_beliefs"].values():
        assert "query_alignment_gaps" not in terminal


def test_administrative_entity_types_are_location_compatible():
    assert canonical_type("administrative territorial entity") == (
        "administrative_territorial_entity"
    )
    assert types_compatible("state", "administrative territorial entity")
    assert types_compatible("municipality", "location")


def test_structural_binding_requires_every_declared_dependency():
    _, _, graph = chain_graph()
    claim = graph.node("c2")
    subgoal = graph.node(claim.target_subgoal)
    subgoal.dependencies.append("uncovered_dependency")
    assert _structural_dependency_binding_coverage(
        claim, graph, claim.target_subgoal,
    ) == 0.0


def test_structural_lineage_does_not_raise_ungrounded_tuple_support_channel():
    _, _, graph = chain_graph()
    claim = graph.node("c2")
    raw = replace(
        claim.score.raw,
        grounding=0.0,
        dependency_consistency=0.25,
        relation_target_alignment=1.0,
        subject_binding_coverage=1.0,
        dependency_binding_coverage=1.0,
        qualifier_coverage=1.0,
        output_slot_coverage=1.0,
    )
    updated = _query_conditioned_signals(
        raw, claim, graph, claim.target_subgoal, "value",
        structural_dependency=True,
    )
    assert updated.dependency_binding_coverage == 1.0
    assert updated.dependency_consistency == 0.25


def test_query_alignment_separates_true_tuple_from_subgoal_coverage_and_repairs_binding():
    cfg = config(
        query_conditioned_semantic_alignment=True,
        structural_dependency_binding_coverage=True,
        generic_evidence_endpoint_grounding=True,
    )
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(
        101, OperationType.EXPAND, payload={"subgoals": [
            {
                "node_id": "s_bridge",
                "question_template": "Where was Darley Ramon Torres born?",
                "instantiated_question": "Where was Darley Ramon Torres born?",
                "dependencies": [], "variable_bindings": {},
                "answer_type": "location", "terminal": False,
            },
            {
                "node_id": "s_root",
                "question_template": (
                    "What administrative territorial entity contains "
                    "Darley Ramon Torres's place of birth?"
                ),
                "instantiated_question": (
                    "What administrative territorial entity contains "
                    "Darley Ramon Torres's place of birth?"
                ),
                "dependencies": ["s_bridge"], "variable_bindings": {},
                "answer_type": "administrative_territorial_entity", "terminal": True,
            },
        ]},
    ))
    graph = controller.apply(graph, operation(
        102, OperationType.RETRIEVE, target="s_bridge", payload={
            "query": "Darley Ramon Torres birthplace",
            "evidence": [{
                "node_id": "e_birth", "document_id": "p_birth", "passage_id": "p_birth",
                "title": "Darley Ramon Torres",
                "source_span": "Darley Ramon Torres was born in Pedro Leopoldo.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "Darley Ramon Torres birthplace",
                "retriever_identity": "fixture",
            }],
        },
    ))
    graph = controller.apply(graph, operation(
        103, OperationType.BRANCH, target="s_bridge", payload={
            "mode": "candidates", "candidates": [{
                "node_id": "c_birth", "subject": "Darley Ramon Torres",
                "relation": "born_in", "value": "Pedro Leopoldo",
                "subject_type": "person", "value_type": "location",
                "answer_type": "location", "evidence_refs": ["e_birth"],
                "source_spans": ["Darley Ramon Torres was born in Pedro Leopoldo."],
                "dependency_claim_ids": [], "extraction_confidence": 0.95,
                "answers_subgoal": True, "answer_position": "value",
            }]},
        ),
    )
    graph = controller.apply(graph, operation(
        104, OperationType.RETRIEVE, target="s_root", payload={
            "query": "Pedro Leopoldo administrative entity",
            "evidence": [{
                "node_id": "e_state", "document_id": "p_state", "passage_id": "p_state",
                "title": "Pedro Leopoldo",
                "source_span": "Pedro Leopoldo is in the state of Minas Gerais.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "Pedro Leopoldo administrative entity",
                "retriever_identity": "fixture",
            }],
        },
    ))
    graph = controller.apply(graph, operation(
        105, OperationType.BRANCH, target="s_root", payload={
            "mode": "candidates", "candidates": [{
                "node_id": "c_state", "subject": "Pedro Leopoldo",
                "relation": "located_in", "value": "Minas Gerais",
                "subject_type": "location", "value_type": "state",
                "answer_type": "administrative_territorial_entity",
                "evidence_refs": ["e_state"],
                "source_spans": ["Pedro Leopoldo is in the state of Minas Gerais."],
                "dependency_claim_ids": ["c_birth"], "extraction_confidence": 0.95,
                "answers_subgoal": False, "answer_position": "none",
            }]},
        ),
    )
    mock = CapturingMock(json_responses=[
        {"scores": [_alignment_row("c_state")]},
        {"scores": [_alignment_row("c_state")]},
    ])
    preverify = graph
    budget = Budget(16, 6000, 200, Usage())
    verifier = MultiSampleIndependentVerifier(mock, budget, cfg)
    proposal = verifier.propose(
        graph, "s_root", "branch_root",
        "What administrative territorial entity contains Darley Ramon Torres's place of birth?",
        "op_v2_106", 1, 800,
    )
    assert proposal is not None
    graph = controller.apply(graph, proposal)
    claim = graph.node("c_state")
    assert claim.score.raw.entailment > 0.9
    assert claim.score.raw.dependency_consistency == 1.0
    assert claim.score.raw.dependency_binding_coverage == 1.0
    assert claim.score.raw.subject_binding_coverage == 1.0
    assert claim.score.raw.output_slot_coverage == 1.0
    assert claim.score.raw.full_subgoal_coverage >= 0.95
    assert claim.score.absolute_support >= cfg.terminal_min_absolute_support
    assert claim.provenance.metadata["verified_answer_position"] == "value"
    assert budget.usage.llm_calls == 2
    assert mock.requests[0][0] == "dynamic_v2_independent_verification_v1_pass_1"
    assert "Compiled query constraint" not in mock.requests[0][1][1]["content"]
    assert mock.requests[1][0] == "hara_v24319_independent_query_alignment_pass_1"
    assert "Compiled query constraint" in mock.requests[1][1][1]["content"]
    assert "Shared evidence" not in mock.requests[1][1][1]["content"]

    certificate_cfg = replace(
        cfg, controller_query_alignment_certificates=True,
    )
    certificate_mock = CapturingMock(json_responses=[
        {"scores": [_alignment_row("c_state")]},
    ])
    certificate_budget = Budget(16, 6000, 200, Usage())
    certificate_verifier = MultiSampleIndependentVerifier(
        certificate_mock, certificate_budget, certificate_cfg,
    )
    certificate_proposal = certificate_verifier.propose(
        preverify, "s_root", "branch_root",
        "What administrative territorial entity contains Darley Ramon Torres's place of birth?",
        "op_v2_107", 1, 800,
    )
    assert certificate_proposal is not None
    certificate_graph = V2GraphController(certificate_cfg).apply(
        preverify, certificate_proposal,
    )
    certificate_claim = certificate_graph.node("c_state")
    assert certificate_claim.score.raw.relation_target_alignment == 1.0
    assert certificate_claim.score.raw.full_subgoal_coverage == 1.0
    assert certificate_budget.usage.llm_calls == 1
    audit = certificate_proposal.payload["scores"]["c_state"]["scoring_audit"]
    assert audit["query_alignment_passes_completed"] == 0
    assert audit["query_alignment_certificates_completed"] == 1
    assert audit["query_alignment_passes"][0]["mode"] == (
        "controller_query_graph_certificate"
    )


def test_controller_relation_certificate_rejects_output_type_as_predicate():
    question = "What administrative territorial entity is Pedro Leopoldo located in?"
    assert _relation_target_certificate(
        question, "located_in", "administrative_territorial_entity",
        known_entities=["Pedro Leopoldo"],
    ) == 1.0
    assert _relation_target_certificate(
        question, "administrative_territorial_entity",
        "administrative_territorial_entity", known_entities=["Pedro Leopoldo"],
    ) == 0.0


def test_conjunctive_grounding_prevents_additive_support_compensation():
    base = config()
    raw = VerificationSignals(
        grounding=0.0, entailment=1.0, type_match=1.0,
        dependency_consistency=1.0, retrieval_support=1.0,
        contradiction_risk=0.0, raw_model_confidence=1.0,
    )
    legacy, _ = fuse_candidate_scores({"c": raw}, base)
    guarded, _ = fuse_candidate_scores({"c": raw}, replace(
        base, grounding_conjunctive_absolute_support=True,
    ))
    assert legacy["c"].absolute_support > 0.0
    assert guarded["c"].absolute_support == 0.0
    assert guarded["c"].evidence_gap == legacy["c"].evidence_gap


def test_terminal_query_alignment_is_conjunctive_not_fused_into_support():
    base, controller, graph = chain_graph()
    cfg = replace(base, query_conditioned_semantic_alignment=True)
    claim = graph.node("c2")
    high_support = claim.score.absolute_support
    graph = controller.apply(graph, operation(110, OperationType.VERIFY, payload={
        "scores": {"c2": {
            **claim.score.raw.__dict__,
            "relation_target_alignment": 0.2,
            "subject_binding_coverage": 1.0,
            "dependency_binding_coverage": 1.0,
            "qualifier_coverage": 1.0,
            "output_slot_coverage": 1.0,
            "full_subgoal_coverage": 0.2,
            "absolute_support": high_support,
            "relative_weight": 1.0, "set_entropy": 0.0,
            "evidence_gap": claim.score.evidence_gap, "status": "scored",
        }},
    }))
    rejected, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(111)],
    )
    assert rejected == []
    assert diagnostics[0]["absolute_support"] == high_support
    assert "relation_target_alignment_below_minimum" in diagnostics[0]["rejection_reasons"]
    assert "full_subgoal_coverage_below_minimum" in diagnostics[0]["rejection_reasons"]

    claim = graph.node("c2")
    graph = controller.apply(graph, operation(112, OperationType.VERIFY, payload={
        "scores": {"c2": {
            **claim.score.raw.__dict__,
            **{name: 1.0 for name in (
                "relation_target_alignment", "subject_binding_coverage",
                "dependency_binding_coverage", "qualifier_coverage",
                "output_slot_coverage", "full_subgoal_coverage",
            )},
            "absolute_support": high_support,
            "relative_weight": 1.0, "set_entropy": 0.0,
            "evidence_gap": claim.score.evidence_gap, "status": "scored",
        }},
    }))
    accepted, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(113)],
    )
    assert len(accepted) == 1
    assert diagnostics[0]["accepted"]
    assert diagnostics[0]["query_alignment_gaps"] == {
        "relation_target_alignment": 0.0,
        "subject_binding_coverage": 0.0,
        "dependency_binding_coverage": 0.0,
        "qualifier_coverage": 0.0,
        "output_slot_coverage": 0.0,
        "full_subgoal_coverage": 0.0,
    }
