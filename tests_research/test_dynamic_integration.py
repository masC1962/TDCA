import json

from tdca_research.budget import Budget
from tdca_research.dynamic.config import DynamicResearchConfig
from tdca_research.dynamic.engine import DynamicHypergraphReasoner
from tdca_research.dynamic.planner import DynamicPlanner, _root_answer_type
from tdca_research.llm import (
    DeterministicMockLLM,
    Generation,
    ProviderRefusalError,
    StructuredOutputError,
)
from tdca_research.models import Passage, RunStatus, Usage
from tdca_research.retrieval import BM25Retriever, BaseRetriever
from tdca_research.runtime import run
from tdca_research.utils import sha256_file


def _passages():
    return [
        Passage("p1", "Alpha", "Alpha is led by Ada Lovelace."),
        Passage("p2", "Ada", "Ada Lovelace was born in River City."),
    ]


def _responses():
    return [
        {"subgoals": [], "root_dependencies": [], "root_answer_type": "location"},
        {"candidates": [{
            "answer": "River City", "subject": "Ada Lovelace", "relation": "born in",
            "answer_type": "location",
            "evidence_ids": [
                "evidence_2_1_p1", "evidence_2_1_p2",
                "evidence_2_2_p1", "evidence_2_2_p2",
            ],
            "source_spans": ["Ada Lovelace was born in River City"],
            "extraction_confidence": 0.95,
        }]},
        {"scores": [{
            "candidate_id": "claim_3_subgoal_root_1", "grounding": 1.0,
            "entailment": 0.95, "type_match": 1.0, "dependency_consistency": 1.0,
            "contradiction_risk": 0.0, "raw_model_confidence": 0.95, "reasons": [],
        }]},
    ]


def _config(**overrides):
    return DynamicResearchConfig(
        max_total_tokens=5000, final_reserve_tokens=200, top_k=2,
        **overrides,
    ).apply_ablation()


def test_dynamic_pipeline_returns_only_graph_grounded_answer():
    passages = _passages()
    prediction, retrieval, reasoning = DynamicHypergraphReasoner(
        DeterministicMockLLM(json_responses=_responses()), BM25Retriever(passages), _config(),
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status == RunStatus.ANSWER
    assert prediction.answer == "River City"
    assert prediction.usage.llm_calls == 3
    assert retrieval and reasoning[-1]["operation"] == "COMMIT"
    graph = reasoning[-1]["graph_snapshot"]
    answer = next(value for value in graph["nodes"].values() if value["kind"] == "answer")
    assert answer["supporting_claims"] and answer["supporting_evidence"]
    assert answer["derivation_edge"] in graph["hyperedges"]


def test_a1_removes_verifier_call_and_multi_candidate_preservation():
    passages = _passages()
    config = _config(dynamic_ablation="A1")
    prediction, _, reasoning = DynamicHypergraphReasoner(
        DeterministicMockLLM(json_responses=_responses()[:2]), BM25Retriever(passages), config,
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status == RunStatus.ANSWER
    assert prediction.usage.llm_calls == 2
    verifier_step = next(value for value in reasoning if value["operation"] == "VERIFY")
    assert verifier_step["llm_call_count"] == 0
    final_graph = reasoning[-1]["graph_snapshot"]
    edge = next(iter(final_graph["hyperedges"].values()))
    assert len(edge["source_node_set"]) == 1


def test_repeated_non_mutating_policy_decision_is_fused_off():
    class StalledReasoner(DynamicHypergraphReasoner):
        def _execute(self, example, graph, operation, *args, **kwargs):
            return graph, False

        def _terminalize(self, graph, controller, terminal, reasoning_trace, budget):
            return graph, False

    passages = _passages()
    config = DynamicResearchConfig(
        max_total_tokens=5000, final_reserve_tokens=200, top_k=2,
        max_policy_iterations=5, enable_adaptive_planning=False,
    )
    prediction, _, reasoning = StalledReasoner(
        DeterministicMockLLM(json_responses=_responses()[:1]), BM25Retriever(passages), config,
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status == RunStatus.ABSTAIN
    assert prediction.usage.llm_calls == 1
    assert [value["operation"] for value in reasoning] == ["EXPAND"]


def test_empty_retrieval_is_not_repeated_until_budget_exhaustion():
    class EmptyRetriever(BaseRetriever):
        name = "empty"

        def search(self, query, top_k):
            return []

    prediction, _, reasoning = DynamicHypergraphReasoner(
        DeterministicMockLLM(json_responses=[
            {"subgoals": [], "root_dependencies": [], "root_answer_type": "location"},
            {"operations": []},
            {"operations": []},
        ]),
        EmptyRetriever(),
        _config(max_retrieval_calls=8),
    ).solve({"qid": "q", "question": "Where?", "passages": []})
    assert prediction.status == RunStatus.ABSTAIN
    assert prediction.usage.retrieval_calls == 1
    assert [value.get("operation") for value in reasoning if value.get("operation")] == [
        "EXPAND", "RETRIEVE",
    ]


def test_graph_capacity_is_budget_abstention_not_infrastructure_failure():
    passages = _passages()
    config = _config(max_graph_nodes=1)
    prediction, _, _ = DynamicHypergraphReasoner(
        DeterministicMockLLM(json_responses=_responses()[:1]), BM25Retriever(passages), config,
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status == RunStatus.ABSTAIN
    assert prediction.stop_reason == "dynamic_graph_budget_exhausted"


def test_provider_refusal_is_audited_and_verifier_falls_back_without_infra_failure():
    class RefusingVerifierLLM(DeterministicMockLLM):
        def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
            if schema_name == "dynamic_soft_verification_v1":
                raise ProviderRefusalError("data_inspection_failed", provider_attempts=1)
            return super().generate_json(messages, schema_name, max_tokens, temperature)

    passages = _passages()
    prediction, _, reasoning = DynamicHypergraphReasoner(
        RefusingVerifierLLM(json_responses=_responses()[:2]),
        BM25Retriever(passages),
        _config(),
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status != RunStatus.INFRASTRUCTURE_FAILURE
    # Two successful uncached generations plus the single refused verifier.
    assert prediction.usage.provider_attempts == 3
    refusal = next(value for value in reasoning if value.get("event") == "provider_refusal")
    assert refusal["stage"] == "verify"
    verifier_step = next(
        value for value in reasoning
        if value.get("operation_id", "").endswith("_model_failure_fallback")
    )
    assert verifier_step["operation"] == "VERIFY"
    assert verifier_step["llm_call_count"] == 0


def test_malformed_verifier_json_is_model_failure_with_deterministic_fallback():
    class MalformedVerifierLLM(DeterministicMockLLM):
        def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
            if schema_name == "dynamic_soft_verification_v1":
                raise StructuredOutputError(
                    "truncated JSON",
                    Generation("{", prompt_tokens=20, completion_tokens=1, finish_reason="length"),
                )
            return super().generate_json(messages, schema_name, max_tokens, temperature)

    passages = _passages()
    prediction, _, reasoning = DynamicHypergraphReasoner(
        MalformedVerifierLLM(json_responses=_responses()[:2]),
        BM25Retriever(passages),
        _config(),
    ).solve({
        "qid": "q", "question": "Where was the leader of Alpha born?",
        "passages": [value.public_dict() for value in passages],
    })
    assert prediction.status != RunStatus.INFRASTRUCTURE_FAILURE
    assert prediction.usage.llm_calls == 3
    failure = next(value for value in reasoning if value.get("event") == "structured_output_failure")
    assert failure["stage"] == "verify"
    assert failure["finish_reason"] == "length"
    assert any(
        value.get("operation_id", "").endswith("_model_failure_fallback")
        for value in reasoning
    )


def test_root_answer_type_uses_question_form_as_deterministic_guardrail():
    assert _root_answer_type("Who leads Alpha?", "country") == "person"
    assert _root_answer_type("When did it happen?", "entity") == "date"
    assert _root_answer_type("How many groups?", "entity") == "number"


def test_initial_planner_preserves_valid_root_rewrite_and_binding():
    response = {
        "subgoals": [{
            "local_id": "leader", "question_template": "Who leads Alpha?",
            "answer_type": "person", "dependencies": [], "variable_bindings": {},
        }],
        "root_dependencies": ["leader"],
        "root_question_template": "Where was $leader born?",
        "root_variable_bindings": {"$leader": "leader"},
        "root_answer_type": "location",
    }
    cfg = _config()
    budget = Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage())
    operation = DynamicPlanner(
        DeterministicMockLLM(json_responses=[response]), budget, cfg,
    ).initial_expand("Where was the leader of Alpha born?")
    root = operation.payload["subgoals"][-1]
    assert root["question_template"] == "Where was $leader born?"
    assert root["variable_bindings"] == {"$leader": "subgoal_1"}


def test_initial_planner_collapses_alpha_equivalent_terminal_subgoal_and_root():
    response = {
        "subgoals": [
            {
                "local_id": "actor", "question_template": "Who played the Terminator?",
                "answer_type": "person", "dependencies": [], "variable_bindings": {},
            },
            {
                "local_id": "law", "question_template": "What law was passed by $actor?",
                "answer_type": "law", "dependencies": ["actor"],
                "variable_bindings": {"$actor": "actor"},
            },
        ],
        "root_dependencies": ["law"],
        "root_question_template": "What law was passed by $bridge?",
        "root_variable_bindings": {"$bridge": "law"},
        "root_answer_type": "law",
    }
    cfg = _config()
    budget = Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage())
    operation = DynamicPlanner(
        DeterministicMockLLM(json_responses=[response]), budget, cfg,
    ).initial_expand("What was the name of the law passed by the actor from Terminator?")
    rows = operation.payload["subgoals"]
    assert [row["node_id"] for row in rows] == ["subgoal_1", "subgoal_root"]
    assert rows[-1]["dependencies"] == ["subgoal_1"]
    assert rows[-1]["variable_bindings"] == {"$actor": "subgoal_1"}


def test_initial_planner_drops_runtime_unbound_subgoals():
    response = {
        "subgoals": [{
            "local_id": "bad", "question_template": "Where was $missing born?",
            "answer_type": "location", "dependencies": [], "variable_bindings": {},
        }],
        "root_dependencies": ["bad"], "root_answer_type": "location",
    }
    cfg = _config()
    budget = Budget(cfg.max_llm_calls, cfg.max_total_tokens, cfg.final_reserve_tokens, Usage())
    operation = DynamicPlanner(
        DeterministicMockLLM(json_responses=[response]), budget, cfg,
    ).initial_expand("Where was the leader born?")
    assert [row["node_id"] for row in operation.payload["subgoals"]] == ["subgoal_root"]
    assert operation.payload["subgoals"][0]["dependencies"] == []


def test_runtime_writes_dynamic_graph_and_mechanism_artifacts(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({
        "id": "q", "question": "Where was the leader of Alpha born?", "answer": "River City",
        "paragraphs": [
            {"id": "p1", "title": "Alpha", "paragraph_text": "Alpha is led by Ada Lovelace.", "is_supporting": True},
            {"id": "p2", "title": "Ada", "paragraph_text": "Ada Lovelace was born in River City.", "is_supporting": True},
            {"id": "p3", "title": "Noise", "paragraph_text": "Unrelated archival note.", "is_supporting": False},
        ],
    }) + "\n", encoding="utf-8")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "seed": 20260820, "dataset_sha256": sha256_file(dataset),
        "splits": {"smoke": ["q"]},
    }), encoding="utf-8")
    config = _config(
        dataset_path=str(dataset), split="smoke", split_seed=20260820,
        split_manifest_path=str(manifest), output_root=str(tmp_path / "outputs"),
    )
    run_dir = run(config, mock=DeterministicMockLLM(json_responses=_responses()))
    for name in (
        "dynamic_graphs.jsonl", "dynamic_per_example_metrics.jsonl",
        "dynamic_metrics.json", "dynamic_metrics_by_hop.json",
    ):
        assert (run_dir / name).exists()
    metrics = json.loads((run_dir / "dynamic_metrics.json").read_text(encoding="utf-8"))
    assert metrics["count"] == 1
    assert metrics["mean_grounded_terminal_rate"] == 1.0


def test_dynamic_split_is_disjoint_and_uses_only_previously_unassigned_ids():
    dynamic = json.loads(open(
        "configs/splits/musique_dynamic_seed20260820.json", encoding="utf-8",
    ).read())
    legacy = json.loads(open(
        "configs/splits/musique_dev_seed520.json", encoding="utf-8",
    ).read())
    assert dynamic["seed"] == 20260820
    assert {key: len(value) for key, value in dynamic["splits"].items()} == {
        "smoke": 20, "development": 50, "heldout": 200,
    }
    new_ids = [qid for values in dynamic["splits"].values() for qid in values]
    old_ids = {qid for values in legacy["splits"].values() for qid in values}
    assert len(new_ids) == len(set(new_ids))
    assert not set(new_ids) & old_ids
