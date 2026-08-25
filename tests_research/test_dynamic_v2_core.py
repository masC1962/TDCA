import json
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
from tdca_research.dynamic.candidates import _explicit_parenthetical_alias
from tdca_research.dynamic_v2.allocator import (
    AdaptiveComputationAllocator,
    ComputationPacket,
    EVCSignals,
    feedback_prior,
    operation_coarse_region_key,
    operation_family,
    operation_region_key,
)
from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.controller import V2GraphController
from tdca_research.dynamic_v2.editor import EventTriggeredGraphEditorV2
from tdca_research.dynamic_v2.engine import (
    DynamicHypergraphV2Reasoner,
    _charged_join_attempt_count,
    _claim_answers_subgoal,
    _execution_packet,
    _extraction_state_fingerprint,
    _join_attempt_key,
    _join_can_answer_subgoal,
    _missing_binding_query,
    _novel_retrieval_hits_for_region,
    _query_token_overlap,
    _retrieval_retry_gate,
    _nary_relevant,
    _suppress_terminal_expansion_when_commit_ready,
)
from tdca_research.dynamic_v2.extraction import (
    TypedClaimExtractor,
    _budget_aware_context,
    _canonicalize_typed_value,
    _fit_completion_to_remaining_budget,
    _enumerated_sibling_values,
)
from tdca_research.dynamic_v2.graph import (
    DynamicReasoningHypergraphV2,
    RetrievalAttemptRecord,
    TerminationKind,
)
from tdca_research.dynamic_v2.join import MultiHopJoinEngine
from tdca_research.dynamic_v2.memory import RelationLightCorpusMemory
from tdca_research.dynamic_v2.query_graph import compile_query_graph, types_compatible
from tdca_research.dynamic_v2.revision import BeliefRevisionDetector
from tdca_research.dynamic_v2.termination import MetaStopPolicy, TerminalBeliefReadout
from tdca_research.dynamic_v2.proof import audit_graph_proof
from tdca_research.dynamic_v2.verifier import (
    MultiSampleIndependentVerifier,
    _projection_type_compatible,
    _type_corrected_projection,
)
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Passage, Usage
from tdca_research.retrieval import BM25Retriever
from tdca_research.utils import normalize_text


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


def test_initial_alias_collapse_rejects_incompatible_output_types():
    proposed = GraphOperation(
        "op_plan", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [
            {
                "node_id": "subgoal_1", "question_template": "What country is palitaw from?",
                "instantiated_question": "What country is palitaw from?", "dependencies": [],
                "variable_bindings": {}, "answer_type": "country", "terminal": False,
            },
            {
                "node_id": "subgoal_root", "question_template": "What country is palitaw from?",
                "instantiated_question": "What country is palitaw from?",
                "dependencies": ["subgoal_1"], "variable_bindings": {},
                "answer_type": "person", "terminal": True,
            },
        ]},
        "test", "offline_test",
    )
    normalized = DynamicHypergraphV2Reasoner._normalize_initial_plan(proposed)
    assert [row["node_id"] for row in normalized.payload["subgoals"]] == [
        "subgoal_1", "subgoal_root",
    ]


def test_terminal_dependency_closure_repairs_only_dependency_free_root():
    proposed = GraphOperation(
        "op_plan", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [
            {
                "node_id": "subgoal_1", "question_template": "Where did Alpha die?",
                "instantiated_question": "Where did Alpha die?", "dependencies": [],
                "variable_bindings": {}, "answer_type": "city", "terminal": False,
            },
            {
                "node_id": "subgoal_root", "question_template": "What else changed?",
                "instantiated_question": "What else changed?", "dependencies": [],
                "variable_bindings": {}, "answer_type": "thing", "terminal": True,
            },
        ]}, "test", "offline_test",
    )
    unchanged = DynamicHypergraphV2Reasoner._normalize_initial_plan(proposed)
    repaired = DynamicHypergraphV2Reasoner._normalize_initial_plan(proposed, True)
    assert next(
        row for row in unchanged.payload["subgoals"] if row["node_id"] == "subgoal_root"
    )["dependencies"] == []
    root = next(
        row for row in repaired.payload["subgoals"] if row["node_id"] == "subgoal_root"
    )
    assert root["dependencies"] == ["subgoal_1"]
    assert root["dependency_repair"] == "terminal_sink_closure_v1"


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


def terminal_operation(number, claim_id="c2", answer="Gamma Country"):
    return operation(number, OperationType.COMMIT, target="s_root", sources=[claim_id], payload={
        "mode": "answer",
        "answer": {
            "node_id": f"answer_{number}",
            "candidate_answer": answer,
            "answer_type": "country",
            "supporting_claims": [claim_id],
            "supporting_evidence": ["e2"],
            "derivation_edge": f"hyperedge_answer_{number}",
            "confidence": 1.0,
            "answer_type_consistency": 1.0,
            "contradiction_risk": 0.0,
            "inference_type": "test_terminal",
            "status": "accepted",
        },
    })


def test_v2_graph_roundtrip_and_controller_seal_detects_external_mutation():
    _, _, graph = joined_graph()
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.canonical_json() == graph.canonical_json()
    restored.nodes["c1"].value = "tampered"
    with pytest.raises(GraphInvariantError, match="outside the V2 controller"):
        restored.validate()


def test_terminal_readout_preserves_channels_and_controller_stores_passing_profile():
    cfg, controller, graph = chain_graph()
    proposals, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(30)],
    )
    assert len(proposals) == 1
    assert diagnostics[0]["accepted"]
    assert diagnostics[0]["absolute_support"] >= cfg.terminal_min_absolute_support
    assert diagnostics[0]["raw_claim_channels"]["c2"] == {
        "absolute_support": graph.node("c2").score.absolute_support,
        "relative_weight": graph.node("c2").score.relative_weight,
        "entropy": graph.node("c2").score.set_entropy,
        "evidence_gap": graph.node("c2").score.evidence_gap,
        "grounding": graph.node("c2").score.raw.grounding,
        "entailment": graph.node("c2").score.raw.entailment,
        "type_match": graph.node("c2").score.raw.type_match,
        "dependency_consistency": graph.node("c2").score.raw.dependency_consistency,
        "retrieval_support": graph.node("c2").score.raw.retrieval_support,
        "contradiction_risk": graph.node("c2").score.raw.contradiction_risk,
        "raw_model_confidence": graph.node("c2").score.raw.raw_model_confidence,
    }
    updated = controller.apply(graph, proposals[0])
    assert updated.terminal_beliefs["answer_30"].accepted
    restored = DynamicReasoningHypergraphV2.from_dict(updated.to_dict())
    restored.validate()
    assert restored.terminal_beliefs == updated.terminal_beliefs
    legacy_payload = updated.to_dict()
    legacy_payload.pop("terminal_beliefs")
    legacy_payload.pop("terminal_readout_version")
    legacy_payload.pop("controller_state_hash")
    legacy = DynamicReasoningHypergraphV2.from_dict(legacy_payload)
    legacy.seal_controller_state()
    legacy.validate()
    assert legacy.terminal_readout_version == ""


def test_terminal_readout_rejects_high_support_with_large_evidence_gap():
    cfg, controller, graph = chain_graph()
    claim = graph.node("c2")
    graph = controller.apply(graph, operation(31, OperationType.VERIFY, payload={
        "scores": {"c2": {
            **claim.score.raw.__dict__,
            "absolute_support": 0.95,
            "relative_weight": 1.0,
            "set_entropy": 0.0,
            "evidence_gap": 0.90,
            "status": "scored",
        }},
    }))
    proposals, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(32)],
    )
    assert proposals == []
    assert diagnostics[0]["absolute_support"] == 0.95
    assert diagnostics[0]["evidence_gap"] == 0.90
    assert "evidence_gap_above_maximum" in diagnostics[0]["rejection_reasons"]


def test_terminal_readout_waits_for_unresolved_competing_branch():
    cfg, _, graph = chain_graph()
    proposals, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(34)], unresolved_branch_ids=["branch_other"],
    )
    assert proposals == []
    assert diagnostics[0]["terminal_gap"] == 1.0
    assert "unresolved_competing_branches" in diagnostics[0]["rejection_reasons"]


def test_terminal_readout_ignores_empty_known_search_branch():
    cfg, _, graph = chain_graph()
    graph.branches["branch_empty"] = BranchState(
        "branch_empty", "branch_root", {}, [], 0.25, BranchStatus.ACTIVE, graph.step,
    )
    graph.seal_controller_state()
    proposals, diagnostics = TerminalBeliefReadout(cfg).evaluate(
        graph, [terminal_operation(35)], unresolved_branch_ids=["branch_empty"],
    )
    assert len(proposals) == 1
    assert diagnostics[0]["accepted"]
    assert "unresolved_competing_branches" not in diagnostics[0]["rejection_reasons"]


def test_terminal_gap_changes_adaptive_evc_ranking_for_matched_operations():
    cfg, _, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    closed = operation(35, OperationType.VERIFY, sources=["c2"], payload={
        "_terminal_context": {
            "terminal_gap": 0.0, "rejection_reasons": [],
        },
    })
    open_gap = operation(36, OperationType.VERIFY, sources=["c2"], payload={
        "_terminal_context": {
            "terminal_gap": 1.0,
            "rejection_reasons": ["absolute_support_below_minimum"],
        },
    })
    packets = allocator.allocate(
        graph, [closed, open_gap], Budget(16, 6000, 200, Usage()),
    )
    best_by_operation = {
        operation_id: max(
            (row for row in packets if row.operation.operation_id == operation_id),
            key=lambda row: row.predicted_evc,
        )
        for operation_id in (closed.operation_id, open_gap.operation_id)
    }
    assert best_by_operation[open_gap.operation_id].raw.terminal_gap > 0
    assert (
        best_by_operation[open_gap.operation_id].predicted_evc
        > best_by_operation[closed.operation_id].predicted_evc
    )


def test_measured_terminal_gap_reduction_enters_utility_and_feedback():
    cfg, controller, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    candidate = operation(37, OperationType.VERIFY, sources=["c2"], payload={
        "_terminal_context": {
            "terminal_gap": 1.0,
            "absolute_support": 0.60,
            "relative_weight": 1.0,
            "entropy": 0.0,
            "evidence_gap": 0.25,
            "chain_coverage": 1.0,
            "rejection_reasons": ["absolute_support_below_minimum"],
        },
    })
    packet = allocator.allocate(
        graph, [candidate], Budget(16, 6000, 200, Usage()),
    )[0]
    updated = controller.reconcile_allocation(
        graph, packet,
        {"llm_calls": 0.0, "tokens": 0.0, "retrieval_calls": 0.0},
        True,
        outcome_metadata={"terminal_state_after": {
            "terminal_gap": 0.0,
            "absolute_support": 0.90,
            "relative_weight": 1.0,
            "entropy": 0.0,
            "evidence_gap": 0.10,
            "chain_coverage": 1.0,
        }},
    )
    outcome = updated.operation_outcome_history[-1]
    assert outcome.state_delta["terminal_gap"] == -1.0
    assert outcome.actual_utility_components_raw["terminal_gap_reduction"] == 1.0
    assert outcome.actual_utility_components_normalized["terminal_gap_reduction"] == 1.0
    assert outcome.actual_utility > 0.0
    later = allocator.allocate(
        updated, [candidate], Budget(16, 6000, 200, Usage()),
    )[0]
    assert later.feedback_prior["observations"] == 1.0
    assert later.feedback_prior["posterior_value"] > 0.5


def test_v22_controller_rejects_answer_without_terminal_readout():
    _, controller, graph = chain_graph()
    with pytest.raises(GraphInvariantError, match="requires terminal belief"):
        controller.apply(graph, terminal_operation(33))


def test_relation_light_memory_activates_three_layer_state_through_controller():
    cfg = config()
    passages = [
        Passage("p1", "Alpha Institute", "Alpha Institute was founded in Beta City."),
        Passage("p2", "Beta City", "Beta City is located in Gamma Country."),
    ]
    memory = RelationLightCorpusMemory.build(passages)
    hits = BM25Retriever(passages).search("Alpha Institute Beta City", 2)
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(1, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_root", "question_template": graph.question,
        "instantiated_question": graph.question, "dependencies": [],
        "variable_bindings": {}, "answer_type": "country", "terminal": True,
    }]}))
    evidence_ids = [f"e{index}" for index in range(1, len(hits) + 1)]
    activation = memory.activate(
        hits, evidence_ids, graph.question, "s_root", "branch_root",
        "Alpha Institute Beta City",
    ).to_payload()
    activation["corpus_memory_fingerprint"] = memory.fingerprint
    graph = controller.apply(graph, operation(2, OperationType.RETRIEVE, payload={
        "query": "Alpha Institute Beta City",
        "memory_activation": activation,
        "evidence": [{
            "node_id": evidence_id, "document_id": hit.passage.passage_id,
            "passage_id": hit.passage.passage_id, "title": hit.passage.title,
            "source_span": hit.passage.text, "retrieval_rank": hit.rank,
            "retrieval_score": hit.raw_score, "retrieval_query": hit.query,
            "retriever_identity": hit.retriever,
        } for evidence_id, hit in zip(evidence_ids, hits)],
    }))
    assert graph.corpus_memory_fingerprint == memory.fingerprint
    assert len(graph.activated_passages) == 2
    assert graph.activated_entities
    assert graph.cross_layer_edges
    assert graph.query_graph["variables"][0]["expected_type"] == "country"
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.canonical_json() == graph.canonical_json()
    assert any(
        row["edge_type"] == "memory_query_activation"
        for row in graph.diffusion_history[-1].typed_messages
    )


def test_query_graph_and_hierarchical_types_are_training_free_and_generic():
    cfg, _, graph = chain_graph()
    compiled = compile_query_graph(graph.question, graph.subgoals())
    assert compiled.variables
    assert compiled.constraints
    assert types_compatible("city", "location")
    assert types_compatible("company", "organization")
    assert not types_compatible("date", "location")


def test_adaptive_allocator_exposes_operation_fidelity_alternatives():
    cfg, _, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    packets = allocator.allocate(graph, [operation(
        99, OperationType.RETRIEVE, payload={"query": "missing relation"},
    )], Budget(16, 6000, 200, Usage()))
    assert {row.fidelity_level for row in packets} == {"low", "medium", "high"}
    assert len({row.requested_budget["retrieval_top_k"] for row in packets}) >= 2
    assert all(row.trace()["fidelity_fraction"] > 0 for row in packets)


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


def test_goal_conditioned_join_filter_keeps_only_output_projecting_path():
    cfg, _, graph = chain_graph()
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidate = next(
        row for row in engine.discover(graph, "branch_root", "s_root")
        if set(row.premise_ids) == {"c1", "c2"}
    )
    assert _join_can_answer_subgoal(
        graph, candidate, graph.node("s_root"), graph.branches["branch_root"],
    )


def test_failed_join_key_changes_only_after_controller_belief_update():
    cfg, controller, graph = chain_graph()
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidate = engine.discover(graph, "branch_root", "s_root")[0]
    before = _join_attempt_key(graph, candidate)
    claim = graph.node("c2")
    graph = controller.apply(graph, operation(90, OperationType.VERIFY, payload={
        "scores": {"c2": {
            **claim.score.raw.__dict__, "absolute_support": 0.61,
            "relative_weight": claim.score.relative_weight,
            "set_entropy": claim.score.set_entropy,
            "evidence_gap": 0.39, "status": "scored",
        }},
    }))
    refreshed = next(
        row for row in engine.discover(graph, "branch_root", "s_root")
        if row.signature == candidate.signature
    )
    assert _join_attempt_key(graph, refreshed) != before


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
    candidate = next(row for row in candidates if set(row.premise_ids) == {"c1", "c3"})
    join = engine.deterministic_operation(graph, candidate, {
        "subject": "Beta City", "relation": "located in", "value": "Delta Region",
        "subject_type": "location", "value_type": "location",
        "derivation_confidence": 0.8, "type_match": 1.0,
        "dependency_consistency": 1.0, "qualifiers": {},
    }, "op_v2_child_join")
    assert join.branch_id == child.branch_id


def test_join_allows_auditable_variable_binding_projection_despite_surface_alias_mismatch():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            # Deliberately does not string-match c1.value. The verified,
            # controller-recorded dependency is the auditable binding.
            "node_id": "c3", "subject": "The greater Beta municipal area", "relation": "chartered in",
            "value": "1600", "subject_type": "location", "value_type": "date",
            "answer_type": "date", "evidence_refs": ["e2"],
            "source_spans": ["Beta City was chartered in 1600"],
            "dependency_claim_ids": ["c1"], "extraction_confidence": 0.9,
            "answers_subgoal": True, "answer_position": "value",
        }],
    }))
    graph = controller.apply(graph, operation(6, OperationType.VERIFY, payload={
        "scores": {
            "c1": {
                "grounding": 1.0, "entailment": 0.9, "type_match": 1.0,
                "dependency_consistency": 1.0, "retrieval_support": 1.0,
                "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
                "absolute_support": 0.9, "relative_weight": 1.0,
                "set_entropy": 0.0, "evidence_gap": 0.1, "status": "scored",
            },
            "c2": {
                "grounding": 1.0, "entailment": 0.9, "type_match": 1.0,
                "dependency_consistency": 1.0, "retrieval_support": 1.0,
                "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
                "absolute_support": 0.9, "relative_weight": 1.0,
                "set_entropy": 0.0, "evidence_gap": 0.1, "status": "scored",
            },
            "c3": {
            # Entailment alone is just below the generic JOIN admission threshold,
            # while the independent projection channels and fused support pass.
            "grounding": 1.0, "entailment": 0.52, "type_match": 1.0,
            "dependency_consistency": 0.75, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.52,
            "absolute_support": 0.75, "relative_weight": 1.0,
            "set_entropy": 0.0, "evidence_gap": 0.25, "status": "scored",
            "answer_position": "value",
            },
        },
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
    assert candidate.orientation == "declared_dependency_binding"
    sibling_augmented = next(
        row for row in engine.discover(graph, "branch_root", "s_root")
        if set(row.premise_ids) == {"c1", "c2", "c3"}
    )
    assert sibling_augmented.projection_premise_id == ""
    assert not _nary_relevant(graph, sibling_augmented, {"c1"})
    assert engine.propose(graph, candidate, "op_v2_projection", 500) is not None
    assert budget.usage.llm_calls == 0


def test_non_answer_bridge_fact_cannot_become_slot_projection():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "c3", "subject": "Beta City", "relation": "located in",
            "value": "Gamma Country", "subject_type": "location",
            "value_type": "country", "answer_type": "country",
            "evidence_refs": ["e2"], "source_spans": ["Beta City"],
            "dependency_claim_ids": ["c1"], "extraction_confidence": 0.95,
            "answers_subgoal": False, "answer_position": "none",
        }],
    }))
    graph = controller.apply(graph, operation(6, OperationType.VERIFY, payload={
        "scores": {"c3": {
            "grounding": 1.0, "entailment": 0.99, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.99,
            "absolute_support": 0.99, "relative_weight": 1.0,
            "set_entropy": 0.0, "evidence_gap": 0.0, "status": "scored",
            "answer_position": "none",
        }},
    }))
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidates = [
        row for row in engine.discover(graph, "branch_root", "s_root")
        if set(row.premise_ids) == {"c1", "c3"}
    ]
    assert candidates
    assert all(not row.projection_premise_id for row in candidates)
    assert not _claim_answers_subgoal(graph, graph.node("c3"))


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
    assert len(packets) == 7
    assert {packet.operation.operation_id for packet in packets} == {
        operation.operation_id for operation in operations
    }
    assert all(packet.raw.__dict__ and packet.normalized.__dict__ for packet in packets)
    assert len({packet.requested_budget["max_tokens"] for packet in packets}) > 1
    selected = next(
        packet for packet in packets
        if packet.operation.operation_type == OperationType.COMMIT
    )
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


def test_extraction_completion_shrinks_to_exact_remaining_prompt_budget():
    budget = Budget(
        max_llm_calls=16,
        max_total_tokens=1000,
        final_reserve_tokens=100,
        usage=Usage(prompt_tokens=300, completion_tokens=50),
    )
    fitted = _fit_completion_to_remaining_budget(
        budget, requested_completion=900, estimated_prompt_tokens=250,
    )
    assert fitted == 300
    assert budget.can_call(fitted, estimated_prompt_tokens=250)
    assert not budget.can_call(fitted + 1, estimated_prompt_tokens=250)


def test_extraction_context_compacts_before_schema_safe_budget_exhaustion():
    budget = Budget(
        max_llm_calls=16,
        max_total_tokens=1000,
        final_reserve_tokens=100,
        usage=Usage(prompt_tokens=300, completion_tokens=50),
    )
    system_prompt = "s" * 120
    user_prefix = "u" * 90
    context = _budget_aware_context(
        ["e" * 700, "f" * 700, "g" * 700], 2000,
        budget, system_prompt, user_prefix,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prefix + context},
    ]
    from tdca_research.utils import estimate_message_tokens
    assert 0 < len(context) < 2000
    assert budget.can_call(
        128, estimated_prompt_tokens=estimate_message_tokens(messages),
    )


def test_missing_binding_query_retrieves_original_objective_as_independent_turn():
    _, _, graph = chain_graph()
    subgoal = graph.node("s_root")
    subgoal.question_template = "Which country contains Beta City?"
    subgoal.instantiated_question = subgoal.question_template
    query = _missing_binding_query(
        graph, subgoal, graph.branches["branch_root"],
        subgoal.instantiated_question, [], [], [],
    )
    assert query == graph.question


def test_zero_call_join_rejection_does_not_consume_semantic_join_budget():
    _, controller, graph = chain_graph()
    merge = operation(
        72, OperationType.MERGE, payload={
            "premise_ids": ["c1", "c2"], "join_signature": "sig",
            "join_kind": "relational_path", "variable_bindings": {},
            "constraints": [], "deterministic_validation": {},
        }, sources=["c1", "c2"],
    )
    packet = AdaptiveComputationAllocator(config()).allocate(
        graph, [merge], Budget(16, 6000, 200, Usage()),
    )[0]
    graph = controller.reconcile_allocation(
        graph, packet,
        {"llm_calls": 0.0, "tokens": 0.0, "retrieval_calls": 0.0},
        completed=False, failure_reason="unsupported_or_unverified_premise",
    )
    assert len(graph.join_attempt_history) == 1
    assert _charged_join_attempt_count(graph) == 0


def test_commit_ready_branch_blocks_shared_terminal_schema_expansion():
    commit = operation(
        73, OperationType.COMMIT, payload={"candidate_id": "c2"}, sources=["c2"],
    )
    expand = operation(74, OperationType.EXPAND, payload={"event": "high_uncertainty_no_join"})
    verify = operation(75, OperationType.VERIFY, sources=["c1"])
    filtered = _suppress_terminal_expansion_when_commit_ready([expand, verify, commit])
    assert [row.operation_type for row in filtered] == [
        OperationType.VERIFY, OperationType.COMMIT,
    ]


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


def test_event_triggered_editor_rejects_attachment_cycle_before_controller_apply():
    cfg = config()
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(80, OperationType.EXPAND, payload={"subgoals": [
        {
            "node_id": "s_target", "question_template": "Who played Victrola?",
            "instantiated_question": "Who played Victrola?", "dependencies": [],
            "variable_bindings": {}, "answer_type": "person", "terminal": False,
        },
        {
            "node_id": "s_downstream", "question_template": "Who was the character named after?",
            "instantiated_question": "Who was the character named after?",
            "dependencies": ["s_target"], "variable_bindings": {},
            "answer_type": "person", "terminal": True,
        },
    ]}))
    before = graph.state_hash()
    editor = EventTriggeredGraphEditorV2(
        DeterministicMockLLM(json_responses=[{"operations": [{
            "operation": "EXPAND",
            "subgoal": {
                "question_template": "Which missing relation identifies the performer?",
                "answer_type": "person",
                "dependencies": ["s_downstream"],
                "variable_bindings": {},
            },
        }]}]),
        Budget(16, 6000, 200, Usage()),
        cfg,
    )
    proposal = editor.propose(
        graph, "high_uncertainty_no_join", graph.branches["branch_root"],
        "op_v2_editor_cycle", "s_target", 500,
    )
    assert proposal is None
    assert editor.last_diagnostics["reason"] == "unsafe_execution_cycle"
    assert graph.state_hash() == before
    graph.validate()


def test_event_editor_atomically_demotes_underdecomposed_root_and_migrates_state():
    cfg = config()
    template = empty_graph(cfg)
    graph = DynamicReasoningHypergraphV2(
        "Border troops of the literature country of Rainer Ernst are from what country?",
        template.limits,
    )
    graph.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    graph.seal_controller_state()
    controller = V2GraphController(cfg)
    graph = controller.apply(graph, operation(90, OperationType.EXPAND, payload={"subgoals": [
        {
            "node_id": "s_country", "question_template": "What country was Rainer Ernst from?",
            "instantiated_question": "What country was Rainer Ernst from?", "dependencies": [],
            "variable_bindings": {}, "answer_type": "country", "terminal": False,
        },
        {
            "node_id": "s_root", "question_template": "What is the literature country of $country?",
            "instantiated_question": "What is the literature country of $country?",
            "dependencies": ["s_country"], "variable_bindings": {"$country": "s_country"},
            "answer_type": "country", "terminal": True,
        },
    ]}))
    graph = controller.apply(graph, operation(91, OperationType.RETRIEVE, target="s_root", payload={
        "query": "East German literature", "evidence": [{
            "node_id": "e_literature", "document_id": "p_literature",
            "passage_id": "p_literature", "title": "Literature of East Germany",
            "source_span": "East German literature was produced in East Germany.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "East German literature", "retriever_identity": "fixture",
        }],
    }))
    graph = controller.apply(graph, operation(92, OperationType.BRANCH, target="s_root", payload={
        "mode": "candidates", "candidates": [{
            "node_id": "c_literature", "subject": "East German literature",
            "relation": "country", "value": "East Germany", "subject_type": "textual",
            "value_type": "country", "answer_type": "country", "evidence_refs": ["e_literature"],
            "source_spans": ["East German literature was produced in East Germany."],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
            "answers_subgoal": True, "answer_position": "value",
        }]},
    ))
    editor = EventTriggeredGraphEditorV2(
        DeterministicMockLLM(json_responses=[{"operations": [{
            "operation": "REPAIR_ROOT",
            "root_question_template": "Border troops of $literature_country are from what country?",
        }]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = editor.propose(
        graph, "high_uncertainty_no_join", graph.branches["branch_root"],
        "op_v2_repair_root", "s_root", 500,
    )
    assert proposal is not None
    bridge_id = proposal.payload["attach_node"]
    updated = controller.apply(graph, proposal)
    root = updated.node("s_root")
    bridge = updated.node(bridge_id)
    assert root.question_template == "Border troops of $literature_country are from what country?"
    assert root.dependencies == [bridge_id]
    assert root.variable_bindings == {"$literature_country": bridge_id}
    assert bridge.question_template == "What is the literature country of $country?"
    assert updated.node("c_literature").target_subgoal == bridge_id
    assert updated.node("e_literature").target_subgoal == bridge_id
    updated.validate()


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
            "quote": "Alpha was founded in Beta City",
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


def test_goal_conditioned_projection_keeps_only_complete_output_candidates():
    cfg = config()
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(70, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_answer", "question_template": "When did Republicans take control?",
        "instantiated_question": "When did Republicans take control?", "dependencies": [],
        "variable_bindings": {}, "answer_type": "date", "terminal": True,
    }]}))
    graph = controller.apply(graph, operation(71, OperationType.RETRIEVE, target="s_answer", payload={
        "query": "Republicans take control", "evidence": [{
            "node_id": "e_date", "document_id": "p_date", "passage_id": "p_date",
            "title": "Congress", "source_span": "Republicans took control in January 2015.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Republicans take control", "retriever_identity": "hybrid",
        }],
    }))
    extractor = TypedClaimExtractor(
        DeterministicMockLLM(json_responses=[{"claims": [
            {
                "subject": "Republicans", "relation": "controlled", "value": "Congress",
                "subject_type": "party", "value_type": "organization",
                "evidence_ids": ["e_date"], "quote": "Republicans took control",
                "extraction_confidence": 0.95, "answer_position": "none",
            },
            {
                "subject": "Republicans", "relation": "took control in",
                "value": "January 2015", "subject_type": "party", "value_type": "date",
                "evidence_ids": ["e_date"],
                "quote": "Republicans took control in January 2015",
                "extraction_confidence": 0.95, "answer_position": "value",
            },
        ]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = extractor.propose(
        graph, "s_answer", "branch_root", "When did Republicans take control?",
        [], "op_v2_goal_projection", 500, 4, "direct_answer",
    )
    assert proposal is not None
    assert proposal.proposed_by == "goal_conditioned_typed_projector_v22"
    assert [row["value"] for row in proposal.payload["candidates"]] == ["January 2015"]
    updated = controller.apply(graph, proposal)
    claim = updated.node("claim_v2_3_s_answer_2")
    assert claim.provenance.metadata["extraction_focus_mode"] == "direct_answer"


def test_typed_value_canonicalization_keeps_atomic_infobox_and_scalar_endpoints():
    location, location_audit = _canonicalize_typed_value(
        "Thaba Putsoa - location Maloti Mountains, Lesotho", "location",
    )
    distance, distance_audit = _canonicalize_typed_value(
        "45 miles northwest of Nashville", "distance",
    )
    assert location == "Thaba Putsoa"
    assert location_audit["original_value"].startswith("Thaba Putsoa")
    assert distance == "45"
    assert distance_audit["kind"] == "typed_scalar"
    assert _canonicalize_typed_value("8.11 million", "number")[0] == "8.11 million"
    assert _canonicalize_typed_value("323-272 BC", "date")[0] == "323-272 BC"
    assert _projection_type_compatible("phrase", "acronym_expansion")
    assert _projection_type_compatible("destroyer_class", "list[destroyer_class]")
    assert not _projection_type_compatible("country", "meaning")


def test_dependency_identity_exception_requires_literal_parenthetical_alias():
    assert _explicit_parenthetical_alias(
        "the literature of the German Democratic Republic (East Germany) was studied",
        "East Germany",
    )
    assert not _explicit_parenthetical_alias(
        "East German literature was produced in East Germany.", "East Germany",
    )
    assert not _explicit_parenthetical_alias(
        "The population was 16 million (East Germany estimate).", "East Germany",
    )
    assert _type_corrected_projection("value", "work", "band", "person") == "none"


def test_query_binding_projects_unbound_endpoint_despite_model_none_vote():
    cfg = config()
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(81, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_dev", "question_template": "Who develops Mozilla Sunbird?",
        "instantiated_question": "Who develops Mozilla Sunbird?", "dependencies": [],
        "variable_bindings": {}, "answer_type": "person_or_organization", "terminal": True,
    }]}))
    graph = controller.apply(graph, operation(82, OperationType.RETRIEVE, target="s_dev", payload={
        "query": "Mozilla Sunbird developer", "evidence": [{
            "node_id": "e_dev", "document_id": "p_dev", "passage_id": "p_dev",
            "title": "Mozilla Sunbird", "source_span": "Mozilla Sunbird was developed by Mozilla Foundation.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Mozilla Sunbird developer", "retriever_identity": "fixture",
        }],
    }))
    graph = controller.apply(graph, operation(83, OperationType.BRANCH, target="s_dev", payload={
        "mode": "candidates", "candidates": [{
            "node_id": "c_dev", "subject": "Mozilla Sunbird", "relation": "developed by",
            "value": "Mozilla Foundation", "subject_type": "software",
            "value_type": "organization", "answer_type": "organization",
            "evidence_refs": ["e_dev"],
            "source_spans": ["Mozilla Sunbird was developed by Mozilla Foundation."],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
            "answers_subgoal": False, "answer_position": "none",
        }]},
    ))
    verifier = MultiSampleIndependentVerifier(
        DeterministicMockLLM(json_responses=[{"scores": [{
            "candidate_id": "c_dev", "grounding": 1.0, "entailment": 0.9,
            "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "answer_position": "none", "contradiction_candidate_ids": [],
        }]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = verifier.propose(
        graph, "s_dev", "branch_root", "Who develops Mozilla Sunbird?",
        "op_v2_verify_binding", 1, 700,
    )
    assert proposal is not None
    graph = controller.apply(graph, proposal)
    assert graph.node("c_dev").provenance.metadata["verified_answer_position"] == "value"
    assert graph.node("c_dev").provenance.metadata["answers_subgoal"] is True


def test_query_binding_rejects_conflicting_model_endpoint_without_changing_support():
    cfg = config()
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(84, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_location", "question_template": "Where is Voshmgir District located?",
        "instantiated_question": "Where is Voshmgir District located?", "dependencies": [],
        "variable_bindings": {}, "answer_type": "location", "terminal": True,
    }]}))
    graph = controller.apply(graph, operation(85, OperationType.RETRIEVE, target="s_location", payload={
        "query": "Voshmgir District location", "evidence": [{
            "node_id": "e_location", "document_id": "p_location", "passage_id": "p_location",
            "title": "Voshmgir District",
            "source_span": "Voshmgir District is in Aqqala County, Golestan Province, Iran.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Voshmgir District location", "retriever_identity": "fixture",
        }],
    }))
    graph = controller.apply(graph, operation(86, OperationType.BRANCH, target="s_location", payload={
        "mode": "candidates", "candidates": [{
            "node_id": "c_location", "subject": "Voshmgir District", "relation": "located in",
            "value": "Aqqala County", "subject_type": "district", "value_type": "county",
            "answer_type": "location", "evidence_refs": ["e_location"],
            "source_spans": ["Voshmgir District is in Aqqala County, Golestan Province, Iran."],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
            "answers_subgoal": True, "answer_position": "value",
        }]},
    ))
    verifier = MultiSampleIndependentVerifier(
        DeterministicMockLLM(json_responses=[{"scores": [{
            "candidate_id": "c_location", "grounding": 1.0, "entailment": 0.9,
            "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "answer_position": "subject", "contradiction_candidate_ids": [],
        }]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = verifier.propose(
        graph, "s_location", "branch_root", "Where is Voshmgir District located?",
        "op_v2_verify_projection_conflict", 1, 700,
    )
    assert proposal is not None
    before_support = graph.node("c_location").score.absolute_support
    graph = controller.apply(graph, proposal)
    claim = graph.node("c_location")
    assert claim.provenance.metadata["verified_answer_position"] == "none"
    assert claim.provenance.metadata["answers_subgoal"] is False
    assert claim.score.absolute_support >= before_support

    # A later pass scores only the new proposed claim.  The old claim is a
    # comparison row and must retain the independent projection decision rather
    # than falling back to its extraction-time label.
    graph = controller.apply(graph, operation(87, OperationType.BRANCH, target="s_location", payload={
        "mode": "candidates", "candidates": [{
            "node_id": "c_country", "subject": "Voshmgir District", "relation": "located in",
            "value": "Iran", "subject_type": "district", "value_type": "country",
            "answer_type": "location", "evidence_refs": ["e_location"],
            "source_spans": ["Voshmgir District is in Aqqala County, Golestan Province, Iran."],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
            "answers_subgoal": True, "answer_position": "value",
        }]},
    ))
    verifier = MultiSampleIndependentVerifier(
        DeterministicMockLLM(json_responses=[{"scores": [{
            "candidate_id": "c_country", "grounding": 1.0, "entailment": 0.9,
            "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "answer_position": "value", "contradiction_candidate_ids": [],
        }]}]),
        Budget(16, 6000, 200, Usage()), cfg,
    )
    proposal = verifier.propose(
        graph, "s_location", "branch_root", "Where is Voshmgir District located?",
        "op_v2_verify_projection_persistence", 1, 700,
    )
    assert proposal is not None
    graph = controller.apply(graph, proposal)
    assert graph.node("c_location").provenance.metadata["verified_answer_position"] == "none"
    assert graph.node("c_location").provenance.metadata["answers_subgoal"] is False
    assert graph.node("c_country").provenance.metadata["answers_subgoal"] is True


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
    cfg.deterministic_goal_path_join = False
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


def test_nary_conjunctive_join_materializes_three_premise_hyperedge_and_audit():
    cfg, controller, graph = chain_graph()
    graph = controller.apply(graph, operation(5, OperationType.RETRIEVE, payload={
        "query": "Gamma Country continent",
        "evidence": [{
            "node_id": "e3", "document_id": "p3", "passage_id": "p3",
            "title": "Gamma Country", "source_span": "Gamma Country is in Delta Continent.",
            "retrieval_rank": 1, "retrieval_score": 1.4,
            "retrieval_query": "Gamma Country continent", "retriever_identity": "hybrid",
        }],
    }))
    graph = controller.apply(graph, operation(6, OperationType.BRANCH, payload={
        "mode": "candidates",
        "candidates": [{
            "node_id": "c3", "subject": "Gamma Country", "relation": "located in",
            "value": "Delta Continent", "subject_type": "country",
            "value_type": "location", "answer_type": "location",
            "evidence_refs": ["e3"], "source_spans": ["Gamma Country is in Delta Continent"],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
        }],
    }))
    scores = {
        node_id: {
            "grounding": 1.0, "entailment": 0.9, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "absolute_support": 0.9, "relative_weight": 1.0,
            "set_entropy": 0.0, "evidence_gap": 0.1, "status": "scored",
        }
        for node_id in ("c1", "c2", "c3")
    }
    graph = controller.apply(graph, operation(7, OperationType.VERIFY, payload={
        "scores": scores,
    }))
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidate = next(
        row for row in engine.discover(graph, "branch_root", "s_root")
        if set(row.premise_ids) == {"c1", "c2", "c3"}
    )
    assert candidate.join_kind == "conjunctive_relational_path"
    assert len(candidate.constraints) == 2
    assert candidate.variable_bindings
    assert all(
        graph.node(node_id).score.absolute_support >= cfg.join_min_premise_support
        and graph.node(node_id).status != CandidateStatus.PROPOSED
        for node_id in candidate.premise_ids
    ), [
        (node_id, graph.node(node_id).status, graph.node(node_id).score.absolute_support)
        for node_id in candidate.premise_ids
    ]
    join = engine.deterministic_operation(graph, candidate, {
        "subject": "Alpha", "relation": "founded in continent", "value": "Delta Continent",
        "subject_type": "organization", "value_type": "location",
        "derivation_confidence": 0.85, "type_match": 1.0,
        "dependency_consistency": 1.0, "qualifiers": {},
    }, "op_v2_nary")
    graph = controller.apply(graph, join)
    attempt = graph.join_attempt_history[-1]
    assert attempt.accepted and len(attempt.premise_ids) == 3
    assert attempt.variable_bindings and attempt.constraints
    edge = next(row for row in graph.hyperedges.values() if row.target_node == attempt.conclusion_node_id)
    assert set(edge.source_node_set) == {"c1", "c2", "c3"}
    assert not any(
        attempt.conclusion_node_id in row.premise_ids and "c1" in row.premise_ids
        for row in engine.discover(graph, "branch_root", "s_root")
    )
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.join_attempt_history[-1].premise_versions == attempt.premise_versions


def test_outcome_feedback_changes_later_evc_and_budget_without_cross_question_state():
    cfg, controller, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 6000, 200, Usage())
    first = allocator.allocate(
        graph, [operation(30, OperationType.BRANCH, sources=["e1"])], budget,
    )[0]
    graph = controller.reconcile_allocation(
        graph, first, {"llm_calls": 1.0, "tokens": 500.0, "retrieval_calls": 0.0},
        completed=False, failure_reason="no_candidates",
    )
    second = allocator.allocate(
        graph, [operation(31, OperationType.BRANCH, sources=["e1"])], budget,
    )[0]
    graph = controller.reconcile_allocation(
        graph, second, {"llm_calls": 1.0, "tokens": 500.0, "retrieval_calls": 0.0},
        completed=False, failure_reason="no_candidates",
    )
    third = allocator.allocate(
        graph, [operation(32, OperationType.BRANCH, sources=["e1"])], budget,
    )[0]
    assert third.feedback_prior["observations"] == 2.0
    assert third.feedback_prior["cooldown_active"] == 1.0
    assert third.predicted_evc < first.predicted_evc
    assert third.requested_budget["max_tokens"] <= first.requested_budget["max_tokens"]
    assert len(graph.operation_outcome_history) == 2
    assert all(row.feedback_applied for row in graph.allocation_history)
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.operation_feedback == graph.operation_feedback
    fresh_graph = chain_graph()[2]
    fresh = AdaptiveComputationAllocator(cfg).allocate(
        fresh_graph, [operation(33, OperationType.BRANCH, sources=["e1"])], budget,
    )[0]
    assert fresh.feedback_prior["observations"] == 0.0
    other_region = operation(34, OperationType.BRANCH, target="unseen_subgoal", sources=["e1"])
    assert feedback_prior(
        graph, operation_family(other_region), operation_region_key(other_region),
    )["observations"] == 0.0


def test_v23_retrieval_attempt_ledger_records_empty_calls_and_roundtrips():
    cfg, _, graph = chain_graph()
    controller = V2GraphController(cfg.merged(retrieval_attempt_aware_scheduling=True))
    before = len(graph.retrieval_attempt_history)
    graph = controller.apply(graph, operation(
        330, OperationType.RETRIEVE,
        payload={
            "query": "a deterministic zero-yield query",
            "evidence": [],
            "allocated_top_k": 7,
            "hit_count": 0,
        },
    ))
    assert len(graph.retrieval_attempt_history) == before + 1
    attempt = graph.retrieval_attempt_history[-1]
    assert attempt.allocated_top_k == 7
    assert attempt.hit_count == attempt.new_evidence_count == 0
    assert not attempt.passage_ids
    restored = DynamicReasoningHypergraphV2.from_dict(graph.to_dict())
    assert restored.retrieval_attempt_history[-1] == attempt


def test_v23_missing_binding_query_never_repeats_an_attempt_without_evidence():
    _, _, graph = chain_graph()
    subgoal = graph.node("s_root")
    subgoal.question_template = "Which country contains Beta City?"
    subgoal.instantiated_question = subgoal.question_template
    query = _missing_binding_query(
        graph, subgoal, graph.branches["branch_root"],
        subgoal.instantiated_question, [], [], [],
        attempted_queries={normalize_text(graph.question)},
    )
    assert query
    assert normalize_text(query) != normalize_text(graph.question)


def test_v23_changed_query_does_not_duplicate_a_passage_in_the_same_region():
    _, _, graph = chain_graph()
    retriever = BM25Retriever([
        Passage("p1", "Alpha", "Alpha was founded in Beta City."),
        Passage("p3", "Gamma", "Gamma is a new passage."),
    ])
    hits = retriever.search("Alpha Gamma", 2)
    novel = _novel_retrieval_hits_for_region(
        graph, hits, "s_root", "branch_root",
    )
    assert [row.passage.passage_id for row in novel] == ["p3"]


def test_v23_enumeration_expansion_requires_same_kind_explicit_members():
    text = (
        "The park lies in Kielce County (Gmina Bieliny, Gmina Daleszyce, "
        "Gmina Górno, Gmina Łagów)."
    )
    assert _enumerated_sibling_values(text, "Gmina Bieliny") == [
        "Gmina Daleszyce", "Gmina Górno", "Gmina Łagów",
    ]
    assert _enumerated_sibling_values(
        "Alice met Bob (CEO, writer, philanthropist).", "Bob",
    ) == []


def test_v23_multi_resource_signals_follow_budget_scarcity_without_fidelity_minmax():
    _, _, graph = chain_graph()
    cfg = config(
        multi_resource_evc=True,
        retrieval_attempt_aware_scheduling=True,
    )
    allocator = AdaptiveComputationAllocator(cfg)
    branch = operation(340, OperationType.BRANCH, sources=["e1"])
    fresh = allocator._signals(graph, branch, Budget(16, 6000, 0, Usage()))
    scarce = allocator._signals(
        graph, branch,
        Budget(16, 6000, 0, Usage(llm_calls=15, prompt_tokens=5700)),
    )
    assert scarce.expected_call_cost > fresh.expected_call_cost
    assert scarce.expected_token_cost > fresh.expected_token_cost

    retrieve = operation(
        341, OperationType.RETRIEVE,
        payload={"query": "new relation query"},
    )
    packets = allocator.allocate(graph, [retrieve], Budget(16, 6000, 0, Usage()))
    assert len(packets) == 3
    assert len({row.normalized.expected_retrieval_cost for row in packets}) == 1
    assert all(
        0.0 <= value <= 1.0
        for row in packets for value in row.normalized.__dict__.values()
    )


def test_v23_hierarchical_feedback_transfers_only_within_same_question_region():
    _, _, graph = chain_graph()
    cfg = config(
        multi_resource_evc=True,
        hierarchical_within_question_feedback=True,
    )
    controller = V2GraphController(cfg)
    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 6000, 0, Usage())
    first_operation = operation(350, OperationType.BRANCH, sources=["e1"])
    first = allocator.allocate(graph, [first_operation], budget)[0]
    graph = controller.reconcile_allocation(
        graph, first,
        {"llm_calls": 1.0, "tokens": 500.0, "retrieval_calls": 0.0},
        completed=False, failure_reason="no_candidates",
    )
    variant = operation(351, OperationType.BRANCH, sources=["e1", "e2"])
    assert operation_region_key(variant) != operation_region_key(first_operation)
    assert operation_coarse_region_key(variant) == operation_coarse_region_key(first_operation)
    later = allocator.allocate(graph, [variant], budget)[0]
    assert later.feedback_prior["exact_observations"] == 0.0
    assert later.feedback_prior["coarse_observations"] == 0.5
    assert later.feedback_prior["posterior_value"] < 0.5
    new_evidence_context = operation(
        35, OperationType.BRANCH, sources=["e1", "e2"],
    )
    assert feedback_prior(
        graph,
        operation_family(new_evidence_context),
        operation_region_key(new_evidence_context),
    )["observations"] == 0.0


def test_uniform_and_fixed_order_allocators_do_not_use_adaptive_ranking():
    _, _, graph = chain_graph()
    budget = Budget(16, 6000, 200, Usage())
    branch_rows = [
        operation(40, OperationType.BRANCH, sources=["e1"]),
        operation(41, OperationType.BRANCH, sources=["e2"]),
    ]
    uniform = AdaptiveComputationAllocator(config(allocator_mode="uniform")).allocate(
        graph, branch_rows, budget,
    )
    assert [row.operation.operation_id for row in uniform] == ["op_v2_40", "op_v2_41"]
    assert len({row.predicted_evc for row in uniform}) == 1
    assert len({tuple(sorted(row.requested_budget.items())) for row in uniform}) == 1
    fixed = AdaptiveComputationAllocator(config(allocator_mode="fixed_order")).allocate(
        graph,
        [
            operation(42, OperationType.RETRIEVE),
            operation(43, OperationType.COMMIT, payload={"candidate_id": "c1"}, sources=["c1"]),
        ],
        budget,
    )
    assert fixed[0].operation.operation_type == OperationType.COMMIT
    assert all(row.allocator_mode == "fixed_order" for row in fixed)


def test_selected_operation_id_is_unique_even_when_graph_step_does_not_change():
    cfg, _, graph = chain_graph()
    allocator = AdaptiveComputationAllocator(cfg)
    budget = Budget(16, 6000, 200, Usage())
    ready = operation(60, OperationType.MERGE, sources=["c1", "c2"], payload={
        "mode": "validate_join",
    })
    first = _execution_packet(allocator.allocate(graph, [ready], budget)[0])
    second = _execution_packet(allocator.allocate(graph, [ready], budget)[0])
    assert first.operation.operation_id != second.operation.operation_id
    assert first.operation.operation_id.endswith(first.allocation_id)


def test_neutral_feedback_prior_does_not_dilute_initial_hot_budget():
    cfg = config()
    allocator = AdaptiveComputationAllocator(cfg)
    remaining = {"llm_calls": 16, "tokens": 6000, "retrieval_calls": 8, "graph_operations": 48}
    packet = allocator._budget_packet(
        operation(61, OperationType.BRANCH),
        0.75,
        remaining,
        {"observations": 0.0, "posterior_value": 0.5, "cooldown_active": 0.0},
    )
    assert packet["candidate_cap"] == cfg.max_extracted_claims_per_round
    assert packet["verification_samples"] == cfg.max_independent_verifications


def test_explicit_set_intersection_requires_a_real_shared_member():
    cfg = config(max_join_arity=4)
    controller = V2GraphController(cfg)
    graph = empty_graph(cfg)
    graph = controller.apply(graph, operation(70, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s_root", "question_template": "Which member is shared by all rosters?",
        "instantiated_question": "Which member is shared by all rosters?",
        "dependencies": [], "variable_bindings": {}, "answer_type": "entity", "terminal": True,
    }]}))
    graph = controller.apply(graph, operation(71, OperationType.RETRIEVE, payload={
        "query": "rosters", "evidence": [{
            "node_id": f"se{index}", "document_id": f"sd{index}", "passage_id": f"sp{index}",
            "title": f"Roster {index}", "source_span": f"Roster {index} has listed members.",
            "retrieval_rank": index, "retrieval_score": 1.0, "retrieval_query": "rosters",
            "retriever_identity": "fixture",
        } for index in range(1, 4)],
    }))
    member_sets = (["Ada", "Bea"], ["Bea", "Cy"], ["Bea", "Dee"])
    graph = controller.apply(graph, operation(72, OperationType.BRANCH, payload={
        "mode": "candidates", "candidates": [{
            "node_id": f"sc{index}", "subject": f"Club {index}", "relation": "has roster",
            "value": f"Roster {index}", "subject_type": "organization", "value_type": "collection",
            "answer_type": "entity", "qualifiers": {"set_members": members},
            "evidence_refs": [f"se{index}"], "source_spans": [f"Roster {index} has listed members."],
            "dependency_claim_ids": [], "extraction_confidence": 0.9,
        } for index, members in enumerate(member_sets, start=1)],
    }))
    graph = controller.apply(graph, operation(73, OperationType.VERIFY, payload={
        "scores": {f"sc{index}": {
            "grounding": 1.0, "entailment": 0.9, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.9,
            "absolute_support": 0.9, "relative_weight": 1.0,
            "set_entropy": 0.0, "evidence_gap": 0.1, "status": "scored",
        } for index in range(1, 4)},
    }))
    candidates = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    ).discover(graph, "branch_root", "s_root")
    intersection = next(row for row in candidates if set(row.premise_ids) == {"sc1", "sc2", "sc3"})
    assert intersection.join_kind == "set_intersection"
    assert intersection.deterministic_validation["set_intersection_members"] == ["bea"]
    assert len(intersection.constraints) == 2


def test_explicit_numeric_comparison_materializes_deterministic_argmax_join():
    cfg = config(max_join_arity=4)
    controller = V2GraphController(cfg)
    graph = DynamicReasoningHypergraphV2(
        "Which university won more national championships?",
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
    graph = controller.apply(graph, operation(90, OperationType.EXPAND, target="s_compare", payload={
        "subgoals": [{
            "node_id": "s_compare",
            "question_template": "Which university won more national championships?",
            "instantiated_question": "Which university won more national championships?",
            "dependencies": [], "variable_bindings": {},
            "answer_type": "university", "terminal": True,
        }],
    }))
    graph = controller.apply(graph, operation(91, OperationType.RETRIEVE, target="s_compare", payload={
        "query": "university national championships", "evidence": [
            {
                "node_id": "e_clemson", "document_id": "p_clemson", "passage_id": "p_clemson",
                "title": "Clemson", "source_span": "Clemson University won 5 national championships.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "university national championships", "retriever_identity": "fixture",
            },
            {
                "node_id": "e_carolina", "document_id": "p_carolina", "passage_id": "p_carolina",
                "title": "South Carolina", "source_span": "University of South Carolina won 10 national championships.",
                "retrieval_rank": 2, "retrieval_score": 0.9,
                "retrieval_query": "university national championships", "retriever_identity": "fixture",
            },
        ],
    }))
    graph = controller.apply(graph, operation(92, OperationType.BRANCH, target="s_compare", payload={
        "mode": "candidates", "candidates": [
            {
                "node_id": "c_clemson", "subject": "Clemson University",
                "relation": "national championships", "value": "5",
                "subject_type": "university", "value_type": "number", "answer_type": "number",
                "evidence_refs": ["e_clemson"], "source_spans": ["won 5 national championships"],
                "dependency_claim_ids": [], "extraction_confidence": 0.9,
            },
            {
                "node_id": "c_carolina", "subject": "University of South Carolina",
                "relation": "national championships", "value": "10",
                "subject_type": "university", "value_type": "number", "answer_type": "number",
                "evidence_refs": ["e_carolina"], "source_spans": ["won 10 national championships"],
                "dependency_claim_ids": [], "extraction_confidence": 0.9,
            },
        ],
    }))
    graph = controller.apply(graph, operation(93, OperationType.VERIFY, target="s_compare", payload={
        "scores": {node_id: {
            "grounding": 1.0, "entailment": 0.95, "type_match": 1.0,
            "dependency_consistency": 1.0, "retrieval_support": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.95,
            "absolute_support": 0.95, "relative_weight": 0.5,
            "set_entropy": 1.0, "evidence_gap": 0.05, "status": "scored",
        } for node_id in ("c_clemson", "c_carolina")},
    }))
    engine = MultiHopJoinEngine(
        DeterministicMockLLM(), Budget(16, 6000, 200, Usage()), cfg,
    )
    candidate = next(
        row for row in engine.discover(graph, "branch_root", "s_compare")
        if row.join_kind == "numeric_argmax"
    )
    proposal = engine.propose(graph, candidate, "op_v2_numeric_compare", 500)
    assert proposal is not None
    assert proposal.estimated_cost["llm_calls"] == 0.0
    graph = controller.apply(graph, proposal)
    conclusion = next(
        graph.node(node_id) for node_id, semantics in graph.claim_semantics.items()
        if semantics.join_signature == candidate.signature
    )
    assert conclusion.value == "University of South Carolina"
    assert len(conclusion.dependency_claim_ids) == 2


def test_v24_join_feasibility_is_pure_and_precedes_provider_budget():
    cfg, _, graph = chain_graph()
    budget = Budget(16, 6000, 0, Usage())
    engine = MultiHopJoinEngine(DeterministicMockLLM(), budget, cfg)
    candidate = engine.discover(graph, "branch_root", "s_root")[0]
    assert engine.check_feasible(graph, candidate).feasible
    failing_id = candidate.premise_ids[0]
    graph.node(failing_id).score.absolute_support = 0.0
    graph.seal_controller_state()
    verdict = engine.check_feasible(graph, candidate)
    assert not verdict.feasible
    assert verdict.reason_codes == ("unsupported_or_unverified_premise",)
    assert failing_id in verdict.premise_ids
    before = (budget.usage.llm_calls, budget.usage.total_tokens)
    assert engine.propose(graph, candidate, "op_infeasible") is None
    assert (budget.usage.llm_calls, budget.usage.total_tokens) == before
    assert engine.last_diagnostics["preallocation_feasible"] is False


def test_v24_extraction_fingerprint_changes_only_with_material_graph_state():
    _, controller, graph = chain_graph()
    evidence = graph.evidence("s_root", "branch_root")
    first = _extraction_state_fingerprint(
        graph, "s_root", "branch_root", evidence, [],
    )
    assert first == _extraction_state_fingerprint(
        graph, "s_root", "branch_root", list(reversed(evidence)), [],
    )
    graph = controller.apply(graph, operation(
        390, OperationType.RETRIEVE, payload={
            "query": "independent new evidence",
            "evidence": [{
                "node_id": "e3", "document_id": "p3", "passage_id": "p3",
                "title": "Gamma", "source_span": "Gamma is independently documented.",
                "retrieval_rank": 1, "retrieval_score": 1.0,
                "retrieval_query": "independent new evidence",
                "retriever_identity": "hybrid",
            }],
        },
    ))
    second = _extraction_state_fingerprint(
        graph, "s_root", "branch_root",
        graph.evidence("s_root", "branch_root"), [],
    )
    assert first != second


def test_v24_region_retrieval_gate_requires_material_query_novelty():
    _, _, graph = chain_graph()
    subgoal = graph.node("s_root")
    branch = graph.branches["branch_root"]
    attempt = RetrievalAttemptRecord(
        "a0", "o0", 1, "s_root", "branch_root",
        "Alpha founded city country", "alpha founded city country",
        10, 0, 0, [],
    )
    rejected = _retrieval_retry_gate(
        graph, subgoal, branch, "Alpha founded city country details", [], [],
        [attempt], set(), 0.80,
    )
    assert not rejected["allowed"]
    assert rejected["reason_code"] == "query_not_materially_novel"
    allowed = _retrieval_retry_gate(
        graph, subgoal, branch, "unrelated bridge relation evidence", [], [],
        [attempt], set(), 0.80,
    )
    assert allowed["allowed"]
    assert allowed["reason_code"] == "zero_yield_novel_query_recovery"
    assert _query_token_overlap("alpha beta", "alpha beta gamma") == 1.0


def test_v24_graph_proof_audit_separates_structure_from_plan_completion():
    _, _, graph = chain_graph()
    audit = audit_graph_proof(graph, "s_root", "branch_root", ["c2"])
    assert audit.graph_proof_completion
    assert audit.dependency_coverage == 1.0
    assert audit.evidence_leaf_coverage == 1.0
    assert audit.proof_connected
    assert audit.evidence_ids == ("e2",)

    _, _, joined = joined_graph()
    joined_claim = max(
        joined.claims("s_root", "branch_root"),
        key=lambda row: joined.claim_semantics[row.node_id].join_depth,
    )
    joined_audit = audit_graph_proof(
        joined, "s_root", "branch_root", [joined_claim.node_id],
    )
    assert joined_audit.graph_proof_completion
    assert joined_audit.proof_depth >= 1
    edge = next(
        row for row in joined.hyperedges.values()
        if row.target_node == joined_claim.node_id
    )
    joined.invalidated_hyperedges.append(edge.edge_id)
    broken = audit_graph_proof(
        joined, "s_root", "branch_root", [joined_claim.node_id],
    )
    assert not broken.graph_proof_completion
    assert not broken.proof_connected
    assert "joined_claim_lacks_valid_hyperedge" in broken.reason_codes


def test_v24_configs_freeze_campaign_caps_without_changing_v23_defaults():
    v24 = DynamicV2ResearchConfig.from_yaml(
        Path("configs/dynamic_hypergraph_v24_qwen_smoke20.yaml")
    )
    v24.validate()
    assert v24.join_preallocation_feasibility_filter
    assert v24.region_level_retrieval_stopping
    assert v24.bounded_extraction_recovery
    assert v24.campaign_provider_call_cap == 2000
    assert v24.campaign_provider_token_cap == 2_000_000
    v23 = DynamicV2ResearchConfig.from_yaml(
        Path("configs/dynamic_hypergraph_v23_qwen_smoke20.yaml")
    )
    assert not v23.join_preallocation_feasibility_filter
    assert not v23.region_level_retrieval_stopping
    assert not v23.bounded_extraction_recovery
    prereg = json.loads(Path(
        "configs/dynamic_v24_preregistration.json"
    ).read_text(encoding="utf-8"))
    assert prereg["campaign"]["provider_attempt_cap"] == 2000
    assert prereg["campaign"]["provider_reported_token_cap"] == 2_000_000
    assert prereg["safe_stop"]["heldout200_authorized"] is False
