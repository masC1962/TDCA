from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig
from tdca_research.dynamic_v2.engine import DynamicHypergraphV2Reasoner
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import Passage, RunStatus
from tdca_research.retrieval import BM25Retriever


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
    assert final["claim_semantics"]["claim_v2_3_subgoal_root_1"]["value_type"] == "location"
    assert final["diffusion_history"]
    assert final["allocation_history"]
    assert all(row["actual_cost"] for row in final["allocation_history"])
    reconciled = [row for row in reasoning if row.get("event") == "allocation_reconciled"]
    assert len(reconciled) == len(final["allocation_history"])
    assert all(row["allocation"]["predicted_evc"] is not None for row in reconciled)
    assert all(row["actual_cost"] for row in reconciled)
    assert final["termination_history"][-1]["outcome"] == "ANSWER"
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
