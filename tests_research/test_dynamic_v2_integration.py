from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.engine import DynamicHypergraphV2Reasoner, _template_similarity
from tdca_research.dynamic.graph import GraphOperation, OperationType
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Passage, RunStatus
from tdca_research.retrieval import BM25Retriever


def test_v2_normalizer_really_collapses_duplicate_final_subgoal_into_root():
    operation = GraphOperation(
        "op_plan", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [
            {
                "node_id": "subgoal_1", "question_template": "Who leads Alpha?",
                "instantiated_question": "Who leads Alpha?", "dependencies": [],
                "variable_bindings": {}, "answer_type": "person", "terminal": False,
            },
            {
                "node_id": "subgoal_2", "question_template": "Where was $leader born?",
                "instantiated_question": "Where was $leader born?",
                "dependencies": ["subgoal_1"],
                "variable_bindings": {"$leader": "subgoal_1"},
                "answer_type": "location", "terminal": False,
            },
            {
                "node_id": "subgoal_root", "question_template": "Where was $bridge born?",
                "instantiated_question": "Where was $bridge born?",
                "dependencies": ["subgoal_2"], "variable_bindings": {},
                "answer_type": "location", "terminal": True,
            },
        ]}, "test_plan", "offline_test",
    )
    normalized = DynamicHypergraphV2Reasoner._normalize_initial_plan(operation)
    rows = normalized.payload["subgoals"]
    assert [row["node_id"] for row in rows] == ["subgoal_1", "subgoal_root"]
    assert rows[-1]["dependencies"] == ["subgoal_1"]
    assert rows[-1]["variable_bindings"] == {"$leader": "subgoal_1"}


def test_v2_template_overlap_recognizes_inflected_full_question_restatement():
    assert _template_similarity(
        "What was the population reduction in $place due to the Black Death?",
        "As a result of the Black Death, how much was the population reduced in the place?",
    ) >= 0.55


def test_v2_normalizer_does_not_collapse_outer_relation_with_lexical_overlap():
    operation = GraphOperation(
        "op_plan", OperationType.EXPAND, "subgoal_root", [], "branch_root",
        {"subgoals": [
            {
                "node_id": "subgoal_1",
                "question_template": "What is the country of citizenship of Rainer Ernst?",
                "instantiated_question": "What is the country of citizenship of Rainer Ernst?",
                "dependencies": [], "variable_bindings": {},
                "answer_type": "country", "terminal": False,
            },
            {
                "node_id": "subgoal_2",
                "question_template": "What is the country of literature of $country_1?",
                "instantiated_question": "What is the country of literature of $country_1?",
                "dependencies": ["subgoal_1"],
                "variable_bindings": {"$country_1": "subgoal_1"},
                "answer_type": "country", "terminal": False,
            },
            {
                "node_id": "subgoal_root",
                "question_template": "Border troops of $country_2 are from what country?",
                "instantiated_question": "Border troops of $country_2 are from what country?",
                "dependencies": ["subgoal_2"],
                "variable_bindings": {"$country_2": "subgoal_2"},
                "answer_type": "country", "terminal": True,
            },
        ]}, "test_plan", "offline_test",
    )
    normalized = DynamicHypergraphV2Reasoner._normalize_initial_plan(operation)
    assert [row["node_id"] for row in normalized.payload["subgoals"]] == [
        "subgoal_1", "subgoal_2", "subgoal_root",
    ]


def test_v2_end_to_end_answer_has_diffusion_allocation_and_typed_claims():
    passages = [
        Passage("p1", "Alpha", "Alpha is led by Ada Lovelace."),
        Passage("p2", "Ada", "Ada Lovelace was born in River City."),
    ]
    responses = [
        {"subgoals": [], "root_dependencies": [], "root_answer_type": "location"},
        {"claims": [{
            "subject": "Ada Lovelace", "relation": "born in", "value": "River City",
            "subject_type": "person", "value_type": "location",
            "evidence_ids": ["evidence_v2_2_1_p2"],
            "source_spans": ["Ada Lovelace was born in River City"],
            "qualifiers": {}, "extraction_confidence": 0.95, "answers_subgoal": True,
        }]},
        {"scores": [{
            "candidate_id": "claim_v2_3_subgoal_root_1", "grounding": 1.0,
            "entailment": 0.95, "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.95, "reasons": [],
        }]},
        {"scores": [{
            "candidate_id": "claim_v2_3_subgoal_root_1", "grounding": 1.0,
            "entailment": 0.90, "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.90, "reasons": [],
        }]},
    ]
    config = DynamicV2ResearchConfig(
        llm_backend="mock", max_total_tokens=8000, final_reserve_tokens=200,
        top_k=2, max_candidates_per_subgoal=8, max_graph_nodes=96,
        meta_stop_evc_threshold=0.01,
        horizon_aware_evc=True, delayed_credit_assignment=True,
        multi_resource_evc=True, choice_conditioned_evc=True,
    )
    prediction, retrieval, reasoning = DynamicHypergraphV2Reasoner(
        DeterministicMockLLM(json_responses=responses), BM25Retriever(passages), config,
    ).solve({
        "qid": "q-v2", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status == RunStatus.ANSWER, (
        prediction.to_dict(),
        [
            {
                key: row.get(key) for key in (
                    "event", "operation", "operation_id", "outcome", "reason",
                    "error_type", "target_id", "diagnostics",
                )
            }
            for row in reasoning
        ],
    )
    assert prediction.answer == "River City"
    assert retrieval
    final = reasoning[-1]["graph_snapshot"]
    assert final["graph_schema_version"] == "dynamic-hypergraph-v2"
    assert any(
        row["value_type"] == "location"
        for row in final["claim_semantics"].values()
    )
    assert final["diffusion_history"]
    assert final["allocation_history"]
    assert all(row["actual_cost"] for row in final["allocation_history"])
    assert final["credit_assignment_history"]
    assert all(row["credit_finalized"] for row in final["allocation_history"])
    assert all(
        row["predicted_immediate_utility"] is not None
        and row["predicted_delayed_proof_return"] is not None
        and row["predicted_normalized_cost"] is not None
        for row in final["allocation_history"]
    )
    assert all(
        {"terminal_gap", "terminal_proximity"}.issubset(row["evc_components_raw"])
        for row in final["allocation_history"]
    )
    assert all(
        "terminal_gap_reduction" in row["actual_utility_components_raw"]
        for row in final["allocation_history"]
    )
    reconciled = [row for row in reasoning if row.get("event") == "allocation_reconciled"]
    assert len(reconciled) == len(final["allocation_history"])
    assert all(row["allocation"]["predicted_evc"] is not None for row in reconciled)
    assert all(row["actual_cost"] for row in reconciled)
    assert final["termination_history"][-1]["outcome"] == "ANSWER"
    assert final["terminal_beliefs"]
    assert any(row.get("event") == "terminal_belief_readout" for row in reasoning)
    answer = next(value for value in final["nodes"].values() if value["kind"] == "answer")
    assert answer["supporting_claims"] and answer["supporting_evidence"]


def test_v2_budget_exhaustion_is_not_reported_as_abstention_or_infrastructure():
    config = DynamicV2ResearchConfig(
        llm_backend="mock", max_llm_calls=1, max_total_tokens=500,
        final_reserve_tokens=100, top_k=1,
    )
    prediction, _, reasoning = DynamicHypergraphV2Reasoner(
        DeterministicMockLLM(json_responses=[
            {"subgoals": [], "root_dependencies": [], "root_answer_type": "location"},
        ]),
        BM25Retriever([Passage("p1", "Alpha", "No useful answer here.")]),
        config,
    ).solve({
        "qid": "q-budget", "question": "Where?",
        "passages": [Passage("p1", "Alpha", "No useful answer here.").public_dict()],
    })
    assert prediction.status == RunStatus.BUDGET_EXHAUSTED
    assert reasoning[-1]["outcome"] == "BUDGET_EXHAUSTED"
