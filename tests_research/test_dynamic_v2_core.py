from pathlib import Path

import pytest

from tdca_research.budget import Budget
from tdca_research.dynamic.graph import (
    BranchState,
    BranchStatus,
    CandidateStatus,
    GraphInvariantError,
    GraphLimits,
    GraphOperation,
    OperationType,
    VerificationSignals,
)
from tdca_research.dynamic.scoring import fuse_candidate_scores
from tdca_research.dynamic_v2.allocator import (
    AdaptiveComputationAllocator,
    ComputationPacket,
    EVCSignals,
)
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.editor import EventTriggeredGraphEditorV2
from tdca_research.dynamic_v2.engine import DynamicHypergraphV2Reasoner
from tdca_research.dynamic_v2.extraction import TypedClaimExtractor
from tdca_research.dynamic_v2.graph import DynamicReasoningHypergraphV2, TerminationKind
from tdca_research.dynamic_v2.join import MultiHopJoinEngine
from tdca_research.dynamic_v2.revision import BeliefRevisionDetector
from tdca_research.dynamic_v2.termination import MetaStopPolicy
from tdca_research.dynamic_v2.verifier import MultiSampleIndependentVerifier
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Usage
from tdca_research.retrieval import BM25Retriever


def config(**overrides):
    return DynamicV2ResearchConfig(
        llm_backend="mock",
        max_total_tokens=6000,
        final_reserve_tokens=200,
        max_candidates_per_subgoal=12,
        max_graph_nodes=96,
        **overrides,
    )


def operation(number, kind, target="s_root", payload=None, sources=None):
    return GraphOperation(
        f"op_v2_{number}", kind, target, sources or [], "branch_root",
        payload or {}, "test", "offline_test",
        {"llm_calls": 0.0, "tokens": 0.0},
    )


def empty_graph(cfg=None):
    cfg = cfg or config()
    graph = DynamicReasoningHypergraphV2(
        "Which country contains the city where Alpha was founded?",
        GraphLimits(
            cfg.max_candidates_per_subgoal, cfg.max_active_branches,
            cfg.max_graph_nodes, cfg.max_hyperedges, cfg.max_graph_revisions,
            cfg.max_revision_per_candidate, cfg.max_graph_depth,
            cfg.max_graph_operations, cfg.max_retrieval_calls,
        ),
    )
    graph.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    graph.seal_controller_state()
    return graph


def chain_graph():
    cfg = config()
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(1, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_root",
        "question_template": "Which country contains the city where Alpha was founded?",
        "instantiated_question": "Which country contains the city where Alpha was founded?",
        "dependencies": [], "variable_bindings": {}, "answer_type": "country", "terminal": True,
    }]}))
    graph = controller.apply(graph, operation(2, OperationType.RETRIEVE, payload={
        "query": "Alpha founded city country",
        "evidence": [
            {
                "node_id": "e1", "document_id": "p1", "passage_id": "p1", "title": "Alpha",
                "source_span": "Alpha was founded in Beta City.", "retrieval_rank": 1,
                "retrieval_score": 2.0, "retrieval_query": "Alpha founded city country",
                "retriever_identity": "hybrid",
            },
            {
                "node_id": "e2", "document_id": "p2", "passage_id": "p2", "title": "Beta City",
                "source_span": "Beta City is located in Gamma Country.", "retrieval_rank": 2,
                "retrieval_score": 1.5, "retrieval_query": "Alpha founded city country",
                "retriever_identity": "hybrid",
            },
        ],
    }))
    graph = controller.apply(graph, operation(3, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [
            {
                "node_id": "c1", "subject": "Alpha", "relation": "founded in", "value": "Beta City",
                "subject_type": "organization", "value_type": "location", "answer_type": "location",
                "evidence_refs": ["e1"], "source_spans": ["Alpha was founded in Beta City"],
                "dependency_claim_ids": [], "extraction_confidence": 0.9,
            },
            {
                "node_id": "c2", "subject": "Beta City", "relation": "located in", "value": "Gamma Country",
                "subject_type": "location", "value_type": "country", "answer_type": "country",
                "evidence_refs": ["e2"], "source_spans": ["Beta City is located in Gamma Country"],
                "dependency_claim_ids": [], "extraction_confidence": 0.9,
            },
        ],
    }))
    profiles, _ = fuse_candidate_scores({
        "c1": VerificationSignals(1.0, 0.9, 1.0, 1.0, 1.0, 0.0, 0.9),
        "c2": VerificationSignals(1.0, 0.9, 1.0, 1.0, 0.5, 0.0, 0.9),
    }, cfg)
    graph = controller.apply(graph, operation(4, OperationType.VERIFY, payload={
        "scores": {
            node_id: {
                **profile.raw.__dict__,
                "absolute_support": profile.absolute_support,
                "relative_weight": profile.relative_weight,
                "set_entropy": profile.set_entropy,
                "evidence_gap": profile.evidence_gap,
                "status": "scored",
            }
            for node_id, profile in profiles.items()
        },
    }))
    return cfg, controller, graph


def joined_graph():
    cfg, controller, graph = chain_graph()
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidates = engine.discover(graph, "branch_root", "s_root")
    assert candidates
    join = engine.deterministic_operation(graph, candidates[0], {
        "subject": "Alpha", "relation": "founded in country", "value": "Gamma Country",
        "subject_type": "organization", "value_type": "country",
        "derivation_confidence": 0.85, "type_match": 1.0,
        "dependency_consistency": 1.0, "qualifiers": {},
    }, "op_v2_5")
    return cfg, controller, controller.apply(graph, join)


def test_v2_graph_roundtrip_and_controller_seal_detects_external_mutation():
    _, _, graph = joined_graph()
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.canonical_json() == graph.canonical_json()
    restored.nodes["c1"].value = "tampered"
    with pytest.raises(GraphInvariantError, match="outside the V2 controller"):
        restored.validate()


def test_typed_join_is_discovered_and_materializes_multi_premise_hyperedge():
    cfg, _, graph = joined_graph()
    joined = [value for value in graph.claim_semantics.values() if value.join_depth > 0]
    assert len(joined) == 1
    claim_id = joined[0].node_id
    edges = [edge for edge in graph.hyperedges.values() if edge.target_node == claim_id]
    assert len(edges) == 1
    assert edges[0].source_node_set == ["c1", "c2"]
    assert graph.belief_states[claim_id].downstream_answer_impact > 0
    assert graph.diffusion_history[-1].typed_messages
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    assert not engine.discover(graph, "branch_root", "s_root")


def test_join_discovers_shared_object_path_without_relation_specific_rules():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "c3", "subject": "Delta Region", "relation": "located in",
            "value": "Gamma Country", "subject_type": "location",
            "value_type": "country", "answer_type": "location",
            "evidence_refs": ["e2"], "source_spans": ["Gamma Country"],
            "dependency_claim_ids": [], "extraction_confidence": 0.8,
        }],
    }))
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidates = engine.discover(graph, "branch_root", "s_root")
    shared = [
        row for row in candidates
        if row.orientation == "shared_value" and set(row.premise_ids) == {"c2", "c3"}
    ]
    assert shared
    assert set(shared[0].open_endpoints) == {"Beta City", "Delta Region"}


def test_join_sees_assigned_parent_claim_from_child_branch_lineage():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.BRANCH, payload={
        "mode": "assignments", "candidate_ids": ["c1", "c2"],
    }, sources=["c1", "c2"]))
    child = next(
        row for row in graph.active_branches() if row.assignments.get("s_root") == "c1"
    )
    graph = controller.apply(graph, GraphOperation(
        "op_v2_6", OperationType.BRANCH, "s_root", ["e2"], child.branch_id,
        {"mode": "candidates", "candidates": [{
            "node_id": "c3", "subject": "Beta City", "relation": "located in",
            "value": "Delta Region", "subject_type": "location",
            "value_type": "location", "answer_type": "location",
            "evidence_refs": ["e2"], "source_spans": ["Beta City"],
            "dependency_claim_ids": ["c1"], "extraction_confidence": 0.8,
        }]},
        "test_child_claim", "offline_test", {"llm_calls": 0.0, "tokens": 0.0},
    ))
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidates = engine.discover(graph, child.branch_id, "s_root")
    assert any(set(row.premise_ids) == {"c1", "c3"} for row in candidates)


def test_join_allows_auditable_variable_binding_projection():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "c3", "subject": "Beta City", "relation": "chartered in",
            "value": "1600", "subject_type": "location", "value_type": "date",
            "answer_type": "date", "evidence_refs": ["e2"],
            "source_spans": ["Beta City was chartered in 1600"],
            "dependency_claim_ids": ["c1"], "extraction_confidence": 0.9,
        }],
    }))
    graph = controller.apply(graph, operation(6, OperationType.VERIFY, payload={
        "scores": {"c3": {
            "grounding": 1.0, "entailment": 0.9, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "absolute_support": 0.9, "relative_weight": 1.0,
            "set_entropy": 0.0, "evidence_gap": 0.1, "status": "scored",
        }},
    }))
    budget = Budget(16, 6000, 200, Usage())
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(json_responses=[{
            "valid": True,
            "reason_codes": ["dependency_binding_established"],
            "derived_claim": {
                "subject": "Beta City", "relation": "chartered in", "value": "1600",
                "subject_type": "location", "value_type": "date",
                "derivation_confidence": 0.9, "type_match": 1.0,
                "dependency_consistency": 1.0, "qualifiers": {},
            },
        }]),
        budget, cfg,
    )
    candidate = next(
        row for row in engine.discover(graph, "branch_root", "s_root")
        if set(row.premise_ids) == {"c1", "c3"}
    )
    assert candidate.projection_premise_id == "c3"
    assert engine.propose(graph, candidate, "op_v2_projection", 500) is not None
    assert budget.usage.llm_calls == 0


def test_active_revision_invalidates_join_descendants_without_deleting_history():
    cfg, controller, graph = joined_graph()
    graph = controller.apply(graph, operation(6, OperationType.COMMIT, payload={"candidate_id": "c1"}))
    profile = graph.node("c1").score
    graph = controller.apply(graph, operation(7, OperationType.VERIFY, payload={
        "scores": {
            "c1": {
                **profile.raw.__dict__, "contradiction_risk": 1.0,
                "absolute_support": 0.25, "relative_weight": 1.0,
                "set_entropy": 0.9, "evidence_gap": 0.8,
                "status": "committed", "contradiction_links": ["c2"],
            }
        },
    }))
    detector = BeliefRevisionDetector(cfg)
    triggers = detector.detect(graph)
    assert triggers and triggers[0].claim_id == "c1"
    graph = controller.apply(
        graph, detector.operation(graph, triggers[0], "branch_root", "op_v2_8", natural=False),
    )
    joined_id = next(value.node_id for value in graph.claim_semantics.values() if value.join_depth > 0)
    assert graph.node("c1").status == CandidateStatus.INVALID
    assert graph.node(joined_id).status == CandidateStatus.INVALID
    invalidated = {"c1", joined_id}
    assert not any(
        row["source"] in invalidated or row["target"] in invalidated
        for row in graph.diffusion_history[-1].typed_messages
    )
    assert graph.belief_states["c1"].computation_heat == 0.0
    assert graph.belief_states[joined_id].computation_heat == 0.0
    assert graph.supersession_history[-1].natural is False
    assert joined_id in graph.nodes


def test_allocator_records_complete_evc_and_produces_non_uniform_budget_packets():
    cfg, controller, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 6000, 200, Usage())
    operations = [
        operation(10, OperationType.RETRIEVE, sources=["e1"]),
        operation(11, OperationType.VERIFY, sources=["c1"]),
        operation(12, OperationType.COMMIT, payload={"candidate_id": "c1"}, sources=["c1"]),
    ]
    packets = allocator.allocate(graph, operations, budget)
    assert len(packets) == 3
    assert all(packet.raw.__dict__ and packet.normalized.__dict__ for packet in packets)
    assert len({packet.requested_budget["max_tokens"] for packet in packets}) > 1
    selected = packets[0]
    attached = allocator.attach(selected.operation, selected)
    graph = controller.apply(graph, attached)
    record = graph.allocation_history[-1]
    assert record.predicted_evc == selected.predicted_evc
    assert record.actual_cost
    assert record.completed


def test_assignment_branch_consumes_adaptive_branch_width_packet():
    cfg, controller, graph = chain_graph()
    branch_operation = operation(
        5,
        OperationType.BRANCH,
        payload={"mode": "assignments", "candidate_ids": ["c1", "c2"]},
        sources=["c1", "c2"],
    )
    allocated = AdaptiveComputationAllocator(cfg).allocate(
        graph, [branch_operation], Budget(16, 6000, 200, Usage()),
    )[0]
    assert allocated.requested_budget["max_tokens"] == 0
    assert allocated.requested_budget["branch_width"] == 2
    packet = ComputationPacket(
        "allocation_branch_test",
        branch_operation,
        ("s_root", "c1", "c2"),
        0.5,
        EVCSignals(),
        EVCSignals(),
        {
            "max_tokens": 0,
            "retrieval_top_k": 1,
            "candidate_cap": 1,
            "verification_samples": 1,
            "branch_width": 2,
            "revision_allowance": 0,
        },
        {"llm_calls": 16, "tokens": 6000, "retrieval_calls": 8, "graph_operations": 60},
    )
    reasoner = DynamicHypergraphV2Reasoner(
        DeterministicMockLLM(), BM25Retriever([]), cfg,
    )
    updated, progressed = reasoner._execute(
        {}, graph, controller, None, None, None, None, packet,
        [], [], [], set(), set(), set(), Budget(16, 6000, 200, Usage()),
    )
    assert progressed
    assert updated.branches["branch_root"].status == BranchStatus.ARCHIVED
    children = [
        row for row in updated.active_branches()
        if row.parent_branch_id == "branch_root"
    ]
    assert len(children) == 2
    assert {row.assignments["s_root"] for row in children} == {"c1", "c2"}


def test_allocator_preserves_schema_safe_budget_while_reducing_output_cardinality():
    cfg, _, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    packet = allocator.allocate(
        graph,
        [operation(13, OperationType.BRANCH, sources=["e1"])],
        Budget(16, 6000, 200, Usage()),
    )[0]
    request = packet.requested_budget
    assert request["candidate_cap"] < cfg.max_extracted_claims_per_round
    assert request["max_tokens"] >= 260 + 90 * request["candidate_cap"]
    assert request["max_tokens"] <= cfg.typed_extraction_max_tokens


def test_event_triggered_editor_proposes_only_controller_applied_structural_edit():
    cfg, controller, graph = chain_graph()
    editor = EventTriggeredGraphEditorV2(
        DeterministicMockLLM(json_responses=[{"operations": [{
            "operation": "EXPAND",
            "subgoal": {
                "question_template": "What missing relation connects Alpha to the requested country?",
                "answer_type": "entity",
                "dependencies": [],
                "variable_bindings": {},
            },
        }]}]),
        Budget(16, 6000, 200, Usage()),
        cfg,
    )
    proposal = editor.propose(
        graph, "missing_terminal_path", graph.branches["branch_root"],
        "op_v2_editor", "s_root", 500,
    )
    assert proposal is not None
    assert graph.operation_history[-1].operation_id == "op_v2_4"
    updated = controller.apply(graph, proposal)
    assert updated.operation_history[-1].operation_id == "op_v2_editor"
    assert updated.operation_history[-1].reason == "event_triggered:missing_terminal_path"


def test_failed_selected_allocation_keeps_predicted_evc_and_measured_cost():
    cfg, controller, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    packet = allocator.allocate(
        graph,
        [operation(14, OperationType.BRANCH, sources=["e1"])],
        Budget(16, 6000, 200, Usage()),
    )[0]
    updated = controller.reconcile_allocation(
        graph, packet,
        {"llm_calls": 1.0, "tokens": 440.0, "retrieval_calls": 0.0},
        completed=False,
        failure_reason="StructuredOutputError",
    )
    row = updated.allocation_history[-1]
    assert row.predicted_evc == packet.predicted_evc
    assert row.actual_cost["tokens"] == 440.0
    assert not row.completed
    assert row.failure_reason == "StructuredOutputError"
    next_packet = allocator.allocate(
        updated,
        [operation(15, OperationType.BRANCH, sources=["e1"])],
        Budget(16, 6000, 200, Usage()),
    )[0]
    assert next_packet.allocation_id != packet.allocation_id


def test_dynamic_v2_cross_dataset_configs_are_frozen_and_resolvable():
    for name, dataset, split in (
        ("hotpot_smoke20", "hotpotqa", "smoke"),
        ("hotpot_heldout200", "hotpotqa", "validation"),
        ("2wiki_smoke20", "2wikimultihopqa", "smoke"),
        ("2wiki_heldout200", "2wikimultihopqa", "validation"),
    ):
        cfg = DynamicV2ResearchConfig.from_yaml(
            Path(f"configs/dynamic_hypergraph_v2_qwen_{name}.yaml")
        )
        cfg.validate()
        assert cfg.dataset == dataset and cfg.split == split
        assert cfg.split_seed == 20260820
        assert Path(cfg.dataset_path).exists()
        assert Path(cfg.split_manifest_path).exists()
        assert cfg.prompt_version == "dynamic-hypergraph-v2-frozen-crossdataset"


def test_subject_answer_projection_canonicalizes_claim_value_for_slot_binding():
    cfg, controller, graph = chain_graph()
    extractor = TypedClaimExtractor(
        DeterministicMockLLM(json_responses=[{"claims": [{
            "subject": "Alpha",
            "relation": "was founded in",
            "value": "Beta City",
            "subject_type": "organization",
            "value_type": "location",
            "evidence_ids": ["e1"],
            "source_spans": ["Alpha was founded in Beta City"],
            "qualifiers": {},
            "extraction_confidence": 0.9,
            "answers_subgoal": True,
            "answer_position": "subject",
        }]}]),
        Budget(16, 6000, 200, Usage()),
        cfg,
    )
    proposal = extractor.propose(
        graph, "s_root", "branch_root", "What was founded in Beta City?",
        [], "op_v2_projection", 500, 2,
    )
    assert proposal is not None
    updated = controller.apply(graph, proposal)
    claim = updated.node("claim_v2_5_s_root_1")
    assert claim.value == "Alpha"
    assert claim.subject == "Beta City"
    assert claim.relation == "inverse_of:was founded in"
    assert claim.provenance.metadata["source_triple"]["value"] == "Beta City"


def test_verifier_can_recover_a_missed_subject_projection_through_controller():
    _, controller, graph = chain_graph()
    claim = graph.node("c2")
    graph = controller.apply(graph, operation(16, OperationType.VERIFY, payload={
        "scores": {"c2": {
            **claim.score.raw.__dict__,
            "absolute_support": claim.score.absolute_support,
            "relative_weight": 1.0,
            "set_entropy": 0.0,
            "evidence_gap": claim.score.evidence_gap,
            "status": "scored",
            "answer_position": "subject",
        }},
    }))
    recovered = graph.node("c2")
    assert recovered.value == "Beta City"
    assert recovered.subject == "Gamma Country"
    assert recovered.provenance.metadata["answers_subgoal"] is True
    assert graph.claim_semantics["c2"].normalized_value == "beta city"


def test_join_rejects_non_scalar_derived_fields_before_controller_transaction():
    cfg, _, graph = chain_graph()
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(json_responses=[{
            "valid": True,
            "reason_codes": [],
            "derived_claim": {
                "subject": "Alpha", "relation": "founded in country", "value": [],
                "subject_type": "organization", "value_type": "country",
            },
        }]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    candidate = engine.discover(graph, "branch_root", "s_root")[0]
    assert engine.propose(graph, candidate, "op_v2_bad_join", 500) is None


def test_multivalued_relations_are_not_automatic_contradictions():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(17, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [
            {
                "node_id": "c3", "subject": "Beta City", "relation": "borders", "value": "Delta",
                "subject_type": "location", "value_type": "location", "answer_type": "location",
                "evidence_refs": ["e1"], "source_spans": ["Beta City"],
                "dependency_claim_ids": [], "extraction_confidence": 0.8,
                "answers_subgoal": True, "answer_position": "value",
            },
            {
                "node_id": "c4", "subject": "Beta City", "relation": "borders", "value": "Epsilon",
                "subject_type": "location", "value_type": "location", "answer_type": "location",
                "evidence_refs": ["e1"], "source_spans": ["Beta City"],
                "dependency_claim_ids": [], "extraction_confidence": 0.8,
                "answers_subgoal": True, "answer_position": "value",
            },
        ],
    }))
    verifier = MultiSampleIndependentVerifier(
        DeterministicMockLLM(json_responses=[{"scores": [
            {
                "candidate_id": node_id, "grounding": 1.0, "entailment": 0.9,
                "type_match": 1.0, "dependency_consistency": 1.0,
                "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
                "answer_position": "value", "contradiction_candidate_ids": [], "reasons": [],
            }
            for node_id in ("c3", "c4")
        ]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = verifier.propose(graph, "s_root", "branch_root", "What borders Beta City?", "op_v2_18", 1, 700)
    assert proposal is not None
    graph = controller.apply(graph, proposal)
    assert not graph.node("c3").contradiction_links
    assert not graph.node("c4").contradiction_links


def test_meta_stop_separates_abstain_and_budget_exhausted():
    cfg, _, graph = chain_graph()
    policy = MetaStopPolicy(cfg)
    normal_budget = Budget(16, 6000, 200, Usage())
    assert policy.decide(graph, [], normal_budget).outcome == TerminationKind.ABSTAIN
    allocator = AdaptiveComputationAllocator(cfg)
    packets = allocator.allocate(
        graph, [operation(20, OperationType.VERIFY, sources=["c1"])], normal_budget,
    )
    exhausted = Budget(0, 0, 0, Usage())
    assert policy.decide(graph, packets, exhausted).outcome == TerminationKind.BUDGET_EXHAUSTED
