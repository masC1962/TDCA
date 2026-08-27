from __future__ import annotations

from dataclasses import replace

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import OperationType
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.allocator import AdaptiveComputationAllocator
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.query_graph import canonical_type, types_compatible
from tdca_research.dynamic_v2.recovery import claim_projects_target
from tdca_research.dynamic_v2.termination import TerminalBeliefReadout
from tdca_research.dynamic_v2.verifier import (
    MultiSampleIndependentVerifier,
    _controller_query_alignment_certificates,
    _candidate_relation_concepts,
    _endpoint_in_anchors,
    _endpoint_mentioned_in_description,
    _inverse_bound_output_role_certificate,
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

    parallel_graph = controller.apply(preverify, operation(
        108, OperationType.BRANCH, target="s_root", payload={
            "mode": "candidates", "candidates": [{
                "node_id": "c_parallel", "subject": "Pedro Leopoldo",
                "relation": "administrative_territorial_entity",
                "value": "Minas Gerais", "subject_type": "location",
                "value_type": "state",
                "answer_type": "administrative_territorial_entity",
                "evidence_refs": ["e_state"],
                "source_spans": ["Pedro Leopoldo is in the state of Minas Gerais."],
                "dependency_claim_ids": ["c_birth"],
                "extraction_confidence": 0.95,
                "answers_subgoal": False, "answer_position": "none",
            }]},
    ))
    parallel_claims = parallel_graph.claims("s_root", "branch_root")
    profiles = {
        claim.node_id: VerificationSignals(
            grounding=1.0, entailment=1.0, type_match=1.0,
            dependency_consistency=1.0, retrieval_support=1.0,
            contradiction_risk=0.0, raw_model_confidence=1.0,
        )
        for claim in parallel_claims
    }
    parallel_certificates = _controller_query_alignment_certificates(
        parallel_claims, parallel_graph, "s_root", profiles,
    )
    assert parallel_certificates["c_state"]["subject_binding_coverage"] == 1.0
    assert parallel_certificates["c_state"]["output_slot_coverage"] == 1.0
    assert parallel_certificates["c_state"]["certificate"][
        "excluded_parallel_tuple_edges"
    ] == 1
    competition_mock = CapturingMock(json_responses=[{
        "scores": [_alignment_row("c_state"), _alignment_row("c_parallel")],
    }])
    competition_proposal = MultiSampleIndependentVerifier(
        competition_mock, Budget(16, 6000, 200, Usage()), certificate_cfg,
    ).propose(
        parallel_graph, "s_root", "branch_root",
        "What administrative territorial entity contains Darley Ramon Torres's place of birth?",
        "op_v2_108_verify", 1, 800,
    )
    assert competition_proposal is not None
    competition_graph = V2GraphController(certificate_cfg).apply(
        parallel_graph, competition_proposal,
    )
    assert competition_graph.node("c_state").score.relative_weight == 1.0
    assert competition_graph.node("c_state").score.set_entropy == 0.0
    assert competition_graph.node("c_parallel").provenance.metadata[
        "verified_answer_position"
    ] == "none"

    reflexive_graph = controller.apply(preverify, operation(
        109, OperationType.BRANCH, target="s_root", payload={
            "mode": "candidates", "candidates": [{
                "node_id": "c_reflexive", "subject": "Pedro Leopoldo",
                "relation": "located_in", "value": "Pedro Leopoldo",
                "subject_type": "location", "value_type": "state",
                "answer_type": "administrative_territorial_entity",
                "evidence_refs": ["e_state"],
                "source_spans": ["Pedro Leopoldo is in the state of Minas Gerais."],
                "dependency_claim_ids": ["c_birth"],
                "extraction_confidence": 0.95,
                "answers_subgoal": False, "answer_position": "none",
            }]},
    ))
    reflexive_claims = reflexive_graph.claims("s_root", "branch_root")
    reflexive_profiles = {
        claim.node_id: VerificationSignals(
            grounding=1.0, entailment=1.0, type_match=1.0,
            dependency_consistency=1.0, retrieval_support=1.0,
            contradiction_risk=0.0, raw_model_confidence=1.0,
        )
        for claim in reflexive_claims
    }
    reflexive_certificate = _controller_query_alignment_certificates(
        reflexive_claims, reflexive_graph, "s_root", reflexive_profiles,
    )["c_reflexive"]
    assert reflexive_certificate["relation_target_alignment"] == 1.0
    assert reflexive_certificate["subject_binding_coverage"] == 1.0
    assert reflexive_certificate["output_slot_coverage"] == 1.0
    assert reflexive_certificate["answer_position"] == "value"


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
    assert _endpoint_in_anchors("Elizabeth Berg", {"elizabeth bergs"})
    assert _endpoint_in_anchors("Mississippi River Delta", {"mississippi river"})
    assert not _endpoint_in_anchors(
        "Mississippi River Delta", {"mississippi river"}, strict_identity=True,
    )
    assert "origin" in _candidate_relation_concepts("has_border_troops")
    assert _inverse_bound_output_role_certificate(
        "has_border_troops", "GDR border guards",
        "Border troops of East Germany are from what country?",
    )


def test_query_mentioned_endpoint_binding_handles_titles_and_connectors():
    assert _endpoint_mentioned_in_description(
        "La fida ninfa", "Who composed La fida ninfa?",
    )
    assert _endpoint_mentioned_in_description(
        "Indian Rebellion of 1857",
        "Which entity was dissolved after the Indian Rebellion in 1857?",
    )
    assert _endpoint_mentioned_in_description(
        "palitaw", "What country does palitaw come from?",
    )
    assert not _endpoint_mentioned_in_description(
        "British East India Company",
        "Which entity was dissolved after the Indian Rebellion in 1857?",
    )
    assert not _endpoint_mentioned_in_description(
        "country", "What country does palitaw come from?",
    )


def test_query_mentioned_title_becomes_controller_base_binding():
    cfg = config(
        query_conditioned_semantic_alignment=True,
        controller_query_alignment_certificates=True,
    )
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(
        160, OperationType.EXPAND, payload={"subgoals": [{
            "node_id": "s_root",
            "question_template": "Who composed La fida ninfa?",
            "instantiated_question": "Who composed La fida ninfa?",
            "dependencies": [], "variable_bindings": {},
            "answer_type": "person", "terminal": True,
        }]},
    ))
    graph = controller.apply(graph, operation(
        161, OperationType.RETRIEVE, target="s_root", payload={
            "query": "Who composed La fida ninfa?",
            "evidence": [{
                "node_id": "e_title", "document_id": "p_title",
                "passage_id": "p_title", "title": "La fida ninfa",
                "source_span": "La fida ninfa was composed by Antonio Vivaldi.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "Who composed La fida ninfa?",
                "retriever_identity": "hybrid",
            }],
        },
    ))
    graph = controller.apply(graph, operation(
        162, OperationType.BRANCH, target="s_root", payload={
            "mode": "candidates", "candidates": [{
                "node_id": "c_title", "subject": "La fida ninfa",
                "relation": "composed_by", "value": "Antonio Vivaldi",
                "subject_type": "creative_work", "value_type": "person",
                "answer_type": "person", "evidence_refs": ["e_title"],
                "source_spans": ["La fida ninfa was composed by Antonio Vivaldi."],
                "dependency_claim_ids": [], "extraction_confidence": 1.0,
            }]},
    ))
    claim = graph.node("c_title")
    profiles = {"c_title": VerificationSignals(
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
    )}
    legacy = _controller_query_alignment_certificates(
        [claim], graph, "s_root", profiles, strict_endpoint_identity=True,
    )["c_title"]
    certified = _controller_query_alignment_certificates(
        [claim], graph, "s_root", profiles, strict_endpoint_identity=True,
        query_mentioned_endpoint_binding=True,
    )["c_title"]
    assert legacy["answer_position"] == "none"
    assert certified["answer_position"] == "value"
    assert certified["certificate"]["subject_base_bound"] is True
    assert certified["certificate"]["value_base_bound"] is False


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


def test_controller_certificate_is_not_overwritten_by_weaker_legacy_binding():
    _, _, graph = chain_graph()
    claim = graph.node("c2")
    raw = replace(
        claim.score.raw,
        relation_target_alignment=1.0,
        subject_binding_coverage=1.0,
        dependency_binding_coverage=1.0,
        qualifier_coverage=1.0,
        output_slot_coverage=1.0,
    )
    updated = _query_conditioned_signals(
        raw, claim, graph, claim.target_subgoal, "none",
        structural_dependency=True, controller_certificate=True,
    )
    assert updated.subject_binding_coverage == 1.0
    assert updated.output_slot_coverage == 1.0
    assert updated.full_subgoal_coverage == 1.0


def test_controller_exact_output_type_can_certify_type_consistency():
    _, _, graph = chain_graph()
    claim = graph.node("c2")
    raw = replace(
        claim.score.raw,
        type_match=0.75,
        relation_target_alignment=1.0,
        subject_binding_coverage=1.0,
        dependency_binding_coverage=1.0,
        qualifier_coverage=1.0,
        output_slot_coverage=1.0,
    )
    updated = _query_conditioned_signals(
        raw, claim, graph, claim.target_subgoal, "value",
        structural_dependency=True,
        controller_certificate=True,
        controller_typed_output=True,
    )
    assert updated.type_match == 1.0
    assert "controller_typed_output_consistency" in updated.reasons


def test_independent_reachability_can_expose_non_base_typed_output():
    cfg = config(
        query_conditioned_semantic_alignment=True,
        controller_query_alignment_certificates=True,
    )
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(
        150, OperationType.EXPAND, payload={"subgoals": [{
            "node_id": "s_root",
            "question_template": "What state is Beyoncé Knowles from?",
            "instantiated_question": "What state is Beyoncé Knowles from?",
            "dependencies": [], "variable_bindings": {},
            "answer_type": "state", "terminal": True,
        }]},
    ))
    graph = controller.apply(graph, operation(
        151, OperationType.RETRIEVE, target="s_root", payload={
            "query": "What state is Beyoncé Knowles from?",
            "evidence": [{
                "node_id": "e_reach", "document_id": "p_reach",
                "passage_id": "p_reach", "title": "Beyoncé Knowles",
                "source_span": "Beyoncé Knowles was born in Houston, Texas.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "What state is Beyoncé Knowles from?",
                "retriever_identity": "hybrid",
            }],
        },
    ))
    graph = controller.apply(graph, operation(
        152, OperationType.BRANCH, target="s_root", payload={
            "mode": "candidates", "candidates": [
                {
                    "node_id": "c_target", "subject": "Beyoncé Knowles",
                    "relation": "is_from", "value": "Texas",
                    "subject_type": "person", "value_type": "state",
                    "answer_type": "state", "evidence_refs": ["e_reach"],
                    "source_spans": ["Beyoncé Knowles was born in Houston, Texas"],
                    "dependency_claim_ids": [], "extraction_confidence": 1.0,
                },
                {
                    "node_id": "c_birth", "subject": "Beyoncé Knowles",
                    "relation": "born_in", "value": "Houston",
                    "subject_type": "person", "value_type": "city",
                    "answer_type": "city", "evidence_refs": ["e_reach"],
                    "source_spans": ["Beyoncé Knowles was born in Houston"],
                    "dependency_claim_ids": [], "extraction_confidence": 1.0,
                },
                {
                    "node_id": "c_located", "subject": "Houston",
                    "relation": "located_in", "value": "Texas",
                    "subject_type": "city", "value_type": "state",
                    "answer_type": "state", "evidence_refs": ["e_reach"],
                    "source_spans": ["Houston, Texas"],
                    "dependency_claim_ids": [], "extraction_confidence": 1.0,
                },
            ],
        },
    ))
    claims = graph.claims("s_root", "branch_root")
    profiles = {
        claim.node_id: VerificationSignals(
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        ) for claim in claims
    }
    legacy = _controller_query_alignment_certificates(
        claims, graph, "s_root", profiles, strict_endpoint_identity=True,
    )
    certified = _controller_query_alignment_certificates(
        claims, graph, "s_root", profiles,
        strict_endpoint_identity=True,
        independent_reachability_projection=True,
    )
    assert legacy["c_target"]["answer_position"] == "none"
    assert certified["c_target"]["answer_position"] == "value"
    assert certified["c_target"]["certificate"]["subject_base_bound"] is True
    assert certified["c_target"]["certificate"]["value_base_bound"] is False


def test_prompt_inclusive_verifier_packets_do_not_fit_only_on_output_tokens():
    base, _, graph = chain_graph()
    cfg = replace(
        base,
        absolute_resource_cost=True,
        exact_fidelity_resource_accounting=True,
        prompt_inclusive_verifier_resource_accounting=True,
        controller_query_alignment_certificates=True,
    )
    allocator = AdaptiveComputationAllocator(cfg)
    verify = operation(
        140, OperationType.VERIFY,
        sources=[f"candidate_{index}" for index in range(6)],
    )
    no_room = Budget(16, 6000, 0, Usage(prompt_tokens=4000))
    assert allocator.allocate(graph, [verify], no_room) == []

    packets = allocator.allocate(
        graph, [verify], Budget(16, 6000, 0, Usage(prompt_tokens=1500)),
    )
    assert packets
    assert all(
        row.requested_budget["token_upper_bound"]
        > row.requested_budget["max_tokens"]
        * row.requested_budget["verification_samples"]
        for row in packets
    )


def test_controller_full_coverage_is_the_projection_certificate():
    _, _, graph = chain_graph()
    claim = graph.node("c2")
    claim.provenance.metadata["answers_subgoal"] = False
    claim.score.raw.full_subgoal_coverage = 1.0
    graph.query_alignment_version = "hara-controller-query-certificate-v2.4.3.20"
    assert claim_projects_target(graph, claim)
    claim.score.raw.full_subgoal_coverage = 0.0
    assert not claim_projects_target(graph, claim)


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
