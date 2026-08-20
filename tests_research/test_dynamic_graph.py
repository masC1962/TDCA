import pytest

from tdca_research.budget import Budget
from tdca_research.dynamic.config import DynamicResearchConfig
from tdca_research.dynamic.candidates import SoftCandidateVerifier
from tdca_research.dynamic.controller import GraphController
from tdca_research.dynamic.editor import EventTriggeredGraphEditor
from tdca_research.dynamic.graph import (
    BranchState,
    BranchStatus,
    CandidateStatus,
    DynamicReasoningHypergraph,
    ExecutionDependencyGraph,
    GraphInvariantError,
    GraphLimits,
    GraphOperation,
    OperationType,
)
from tdca_research.dynamic.policy import decide_candidate_set
from tdca_research.dynamic.scheduler import OperationScheduler, OperationSignals
from tdca_research.dynamic.scoring import fuse_candidate_scores
from tdca_research.dynamic.terminal import GraphGroundedTerminalReasoner, _template_signature
from tdca_research.dynamic.graph import VerificationSignals
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Usage


def config(**overrides):
    return DynamicResearchConfig(
        llm_backend="mock", max_total_tokens=5000, final_reserve_tokens=200,
        **overrides,
    )


def graph(cfg=None):
    cfg = cfg or config()
    value = DynamicReasoningHypergraph(
        "Where was the leader born?",
        GraphLimits(
            cfg.max_candidates_per_subgoal, cfg.max_active_branches,
            cfg.max_graph_nodes, cfg.max_hyperedges, cfg.max_graph_revisions,
            cfg.max_revision_per_candidate, cfg.max_graph_depth,
            cfg.max_graph_operations, cfg.max_retrieval_calls,
        ),
    )
    value.branches["branch_root"] = BranchState(
        "branch_root", None, {}, [], 1.0, BranchStatus.ACTIVE, 0,
    )
    return value


def op(number, kind, target="s1", payload=None, branch="branch_root"):
    return GraphOperation(
        f"op_{number}", kind, target, [], branch, payload or {}, "test", "test",
    )


def expanded():
    controller = GraphController()
    value = graph()
    value = controller.apply(value, op(1, OperationType.EXPAND, payload={"subgoals": [{
        "node_id": "s1", "question_template": "Who leads Alpha?",
        "instantiated_question": "Who leads Alpha?", "dependencies": [],
        "variable_bindings": {}, "answer_type": "entity", "terminal": True,
    }]}))
    return controller, value


def with_candidates():
    controller, value = expanded()
    value = controller.apply(value, op(2, OperationType.RETRIEVE, payload={
        "query": "Who leads Alpha?", "evidence": [{
            "node_id": "e1", "document_id": "p1", "passage_id": "p1",
            "title": "Alpha", "source_span": "Alpha is led by Ada.",
            "retrieval_rank": 1, "retrieval_score": 2.3,
            "retrieval_query": "Who leads Alpha?", "retriever_identity": "bm25",
        }],
    }))
    value = controller.apply(value, op(3, OperationType.BRANCH, payload={
        "mode": "candidates", "candidates": [
            {"node_id": "c1", "value": "Ada", "subject": "Alpha", "relation": "led by", "evidence_refs": ["e1"]},
            {"node_id": "c2", "value": "Grace", "subject": "Alpha", "relation": "led by", "evidence_refs": ["e1"]},
        ],
    }))
    scores, _ = fuse_candidate_scores({
        "c1": VerificationSignals(.9, .9, 1, 1, 1, .1, .8),
        "c2": VerificationSignals(.8, .8, 1, 1, 1, .1, .8),
    }, config())
    value = controller.apply(value, op(4, OperationType.VERIFY, payload={
        "scores": {
            candidate_id: {
                **profile.raw.__dict__,
                "absolute_support": profile.absolute_support,
                "relative_weight": profile.relative_weight,
                "set_entropy": profile.set_entropy,
                "evidence_gap": profile.evidence_gap,
                "status": "scored",
            }
            for candidate_id, profile in scores.items()
        },
    }))
    return controller, value


def test_graph_roundtrip_is_exact_and_operation_is_auditable():
    _, value = with_candidates()
    restored = DynamicReasoningHypergraph.from_dict(value.to_dict())
    assert restored.canonical_json() == value.canonical_json()
    assert value.operation_history[-1].graph_before_hash
    assert value.operation_history[-1].graph_after_hash
    assert value.operation_history[-1].graph_before_hash != value.operation_history[-1].graph_after_hash


def test_execution_graph_is_acyclic_even_when_structural_contradictions_are_allowed():
    execution = ExecutionDependencyGraph()
    execution.add_node("a", [])
    execution.add_node("b", ["a"])
    with pytest.raises(GraphInvariantError, match="acyclic"):
        execution.replace_dependencies("a", ["b"])


def test_evidence_and_claim_provenance_cannot_reference_missing_nodes():
    controller, value = expanded()
    with pytest.raises(GraphInvariantError, match="missing"):
        controller.apply(value, op(2, OperationType.BRANCH, payload={
            "mode": "candidates", "candidates": [
                {"node_id": "c1", "value": "Ada", "evidence_refs": ["missing"]},
            ],
        }))


def test_uncertainty_preserves_raw_absolute_relative_entropy_and_gap():
    cfg = config()
    profiles, summary = fuse_candidate_scores({
        "a": VerificationSignals(.9, .8, .7, .6, .5, .1, .9),
        "b": VerificationSignals(.8, .7, .6, .5, .4, .2, .8),
    }, cfg)
    assert profiles["a"].raw.entailment == .8
    assert profiles["a"].absolute_support > profiles["b"].absolute_support
    assert profiles["a"].relative_weight + profiles["b"].relative_weight == pytest.approx(1)
    assert 0 < summary.entropy <= 1
    assert profiles["a"].evidence_gap != profiles["a"].absolute_support


def test_lazy_branching_creates_lightweight_assignment_branches():
    controller, value = with_candidates()
    value = controller.apply(value, op(5, OperationType.BRANCH, payload={
        "mode": "assignments", "candidate_ids": ["c1", "c2"],
        "branch_ids": ["branch_a", "branch_b"],
    }))
    assert value.branches["branch_root"].status == BranchStatus.ARCHIVED
    assert value.branches["branch_a"].assignments == {"s1": "c1"}
    assert value.branches["branch_b"].assignments == {"s1": "c2"}
    assert value.nodes["c1"].status == CandidateStatus.SCORED


def test_pruned_candidate_is_archived_and_removed_from_active_assignment():
    controller, value = with_candidates()
    value = controller.apply(value, op(5, OperationType.COMMIT, payload={"candidate_id": "c1"}))
    value = controller.apply(value, op(6, OperationType.PRUNE, payload={"candidate_ids": ["c1"]}))
    assert value.nodes["c1"].status == CandidateStatus.ARCHIVED
    assert "s1" not in value.branches["branch_root"].assignments
    value.validate()


def test_commit_can_be_reopened_with_revision_history():
    controller, value = with_candidates()
    value = controller.apply(value, op(5, OperationType.COMMIT, payload={"candidate_id": "c1"}))
    value = controller.apply(value, op(6, OperationType.REVISE, payload={"action": "reopen", "claim_id": "c1"}))
    assert value.nodes["c1"].status == CandidateStatus.REOPENED
    assert value.revision_history
    assert "s1" not in value.branches["branch_root"].assignments


def test_candidate_policy_branches_only_under_uncertainty():
    _, value = with_candidates()
    candidates = value.claims("s1", "branch_root")
    summary = type("Summary", (), {
        "top_margin": candidates[0].score.relative_weight - candidates[1].score.relative_weight,
        "entropy": candidates[0].score.set_entropy,
    })()
    decision = decide_candidate_set(candidates, summary, config(branch_margin_threshold=1.0))
    assert decision.action == "branch"


def test_operation_scheduler_normalizes_components_and_marks_single_choice_trivial():
    cfg = config()
    scheduler = OperationScheduler(cfg)
    first = op(1, OperationType.RETRIEVE)
    second = op(2, OperationType.VERIFY)
    ranked = scheduler.rank([first, second], {
        "op_1": OperationSignals(uncertainty_reduction=.9, dependency_unlock=.8, expected_cost=.8),
        "op_2": OperationSignals(uncertainty_reduction=.5, dependency_unlock=.2, expected_cost=.1),
    })
    assert len(ranked) == 2
    assert all(0 <= value <= 1 for row in ranked for value in row.normalized_signals.__dict__.values())
    only = scheduler.rank([first], {"op_1": OperationSignals(uncertainty_reduction=.9)})[0]
    assert set(only.normalized_signals.__dict__.values()) == {0.0}


def test_event_expansion_is_atomically_inserted_before_target():
    controller = GraphController()
    value = graph()
    value = controller.apply(value, op(1, OperationType.EXPAND, target="subgoal_root", payload={
        "subgoals": [
            {"node_id": "s1", "question_template": "First?", "dependencies": [], "variable_bindings": {}},
            {"node_id": "subgoal_root", "question_template": "Root?", "dependencies": ["s1"], "variable_bindings": {}, "terminal": True},
        ],
    }))
    value = controller.apply(value, op(2, OperationType.EXPAND, target="s2", payload={
        "subgoals": [{
            "node_id": "s2", "question_template": "Missing relation?",
            "dependencies": ["s1"], "variable_bindings": {},
        }],
        "attach_target": "subgoal_root", "attach_node": "s2",
    }))
    assert value.execution_graph.dependencies["s2"] == ["s1"]
    assert value.execution_graph.dependencies["subgoal_root"] == ["s2"]
    assert value.node("subgoal_root").dependencies == ["s2"]


def test_event_attachment_preserves_existing_target_variable_bindings():
    controller = GraphController()
    value = graph()
    value = controller.apply(value, op(1, OperationType.EXPAND, target="root", payload={
        "subgoals": [
            {"node_id": "s1", "question_template": "Operator?", "dependencies": [], "variable_bindings": {}},
            {"node_id": "target", "question_template": "Classes operated by $operator?",
             "dependencies": ["s1"], "variable_bindings": {"$operator": "s1"}},
            {"node_id": "root", "question_template": "Root?", "dependencies": ["target"],
             "variable_bindings": {}, "terminal": True},
        ],
    }))
    value = controller.apply(value, op(2, OperationType.EXPAND, target="inserted", payload={
        "subgoals": [{
            "node_id": "inserted", "question_template": "Extra prerequisite?",
            "dependencies": ["s1"], "variable_bindings": {},
        }],
        "attach_target": "target", "attach_node": "inserted",
    }))
    assert value.node("target").dependencies == ["inserted", "s1"]
    assert value.node("target").variable_bindings == {"$operator": "s1"}
    value.validate()


def test_graph_editor_accepts_equivalent_flat_schema_and_resolves_value_binding():
    controller, value = with_candidates()
    value = controller.apply(value, op(5, OperationType.COMMIT, payload={"candidate_id": "c1"}))
    value = controller.apply(value, op(6, OperationType.EXPAND, target="subgoal_root", payload={
        "subgoals": [{
            "node_id": "subgoal_root", "question_template": "Where was the leader born?",
            "dependencies": ["s1"], "variable_bindings": {}, "terminal": True,
        }],
    }))
    llm = DeterministicMockLLM(json_responses=[{"operations": [{
        "type": "EXPAND", "question_template": "Where was $leader born?",
        "answer_type": "location", "dependencies": ["s1"],
        "variable_bindings": {"$leader": "Ada"},
    }]}])
    cfg = config()
    editor = EventTriggeredGraphEditor(
        llm, Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage()), cfg,
    )
    proposals = editor.propose(
        value, "missing_terminal_path", value.branches["branch_root"], "op_editor", "subgoal_root",
    )
    assert len(proposals) == 1
    assert proposals[0].payload["subgoals"][0]["variable_bindings"] == {"$leader": "s1"}
    updated = controller.apply(value, proposals[0])
    assert updated.node("subgoal_root").dependencies == ["subgoal_dynamic_7"]


def test_editor_attached_final_relation_can_be_promoted_to_grounded_answer():
    cfg = config()
    controller = GraphController()
    value = graph(cfg)
    value = controller.apply(value, op(1, OperationType.EXPAND, target="subgoal_root", payload={
        "subgoals": [{
            "node_id": "subgoal_root", "question_template": "Who leads the country?",
            "dependencies": [], "variable_bindings": {}, "answer_type": "person", "terminal": True,
        }],
    }))
    edit = GraphOperation(
        "op_2", OperationType.EXPAND, "s_final", [], "branch_root",
        {"subgoals": [{
            "node_id": "s_final", "question_template": "Who leads Freedonia?",
            "dependencies": [], "variable_bindings": {}, "answer_type": "person",
        }], "attach_target": "subgoal_root", "attach_node": "s_final"},
        "event_triggered:root_answer_type_gap", "llm_graph_editor_v1",
    )
    value = controller.apply(value, edit)
    value = controller.apply(value, op(3, OperationType.RETRIEVE, target="s_final", payload={
        "query": "Who leads Freedonia?", "evidence": [{
            "node_id": "e_final", "document_id": "p", "passage_id": "p",
            "title": "Freedonia", "source_span": "Freedonia is led by Ada.",
            "retrieval_rank": 1, "retrieval_score": 1.0,
            "retrieval_query": "Who leads Freedonia?", "retriever_identity": "bm25",
        }],
    }))
    value = controller.apply(value, op(4, OperationType.BRANCH, target="s_final", payload={
        "mode": "candidates", "candidates": [{
            "node_id": "c_final", "value": "Ada", "answer_type": "person",
            "subject": "Freedonia", "relation": "led by", "evidence_refs": ["e_final"],
        }],
    }))
    profile, _ = fuse_candidate_scores({
        "c_final": VerificationSignals(1, 1, 1, 1, 1, 0, 1),
    }, cfg)
    score = profile["c_final"]
    value = controller.apply(value, op(5, OperationType.VERIFY, target="s_final", payload={
        "scores": {"c_final": {
            **score.raw.__dict__, "absolute_support": score.absolute_support,
            "relative_weight": score.relative_weight, "set_entropy": score.set_entropy,
            "evidence_gap": score.evidence_gap, "status": "scored",
        }},
    }))
    value = controller.apply(value, op(6, OperationType.COMMIT, target="s_final", payload={
        "candidate_id": "c_final",
    }))
    terminal = GraphGroundedTerminalReasoner(
        DeterministicMockLLM(),
        Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage()),
        cfg,
    )
    operations, unresolved = terminal.direct_operations(
        value, value.active_branches(), "op_7_answer",
    )
    assert not unresolved
    assert operations[0].payload["answer"]["candidate_answer"] == "Ada"
    assert operations[0].payload["answer"]["inference_type"] == "event_triggered_dependency_completion"


def test_root_alias_signature_ignores_variable_names_and_what_which_surface_form():
    assert _template_signature("Which sibling of $actor?") == _template_signature(
        "What sibling of $bridge_2?",
    )


def test_soft_verifier_missing_candidate_row_uses_audited_deterministic_fallback():
    _, value = with_candidates()
    for candidate in value.claims("s1", "branch_root"):
        candidate.status = CandidateStatus.PROPOSED
    cfg = config(soft_verifier_model_weight=0.25)
    verifier = SoftCandidateVerifier(
        DeterministicMockLLM(json_responses=[{"scores": [{
            "candidate_id": "c1", "grounding": 1, "entailment": 1,
            "type_match": 1, "dependency_consistency": 1,
            "contradiction_risk": 0, "raw_model_confidence": 1,
        }]}]),
        Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage()),
        cfg,
    )
    operation, _ = verifier.propose(value, "s1", "branch_root", "Who leads Alpha?", "op_soft")
    assert operation is not None
    missing = operation.payload["scores"]["c2"]
    assert missing["absolute_support"] > 0
    assert missing["scoring_audit"]["mode"] == "deterministic_missing_row_fallback"
    assert operation.payload["scores"]["c1"]["scoring_audit"]["model_weight"] == 0.25
