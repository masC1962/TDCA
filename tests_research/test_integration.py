from tdca_research.config import ResearchConfig
from tdca_research.llm import DeterministicMockLLM, Generation, StructuredOutputError
from tdca_research.models import Passage, QAExample, RunStatus
from tdca_research.reasoning import StructuredReasoner
from tdca_research.retrieval import BM25Retriever


def responses():
    return [
        {"plan_type": "chain", "slots": [
            {"slot_id": "s1", "subquestion_template": "Who leads Alpha?", "answer_type": "person", "dependencies": [], "variable_bindings": [], "output_variable": "$leader", "terminal": False, "confidence": 0.8},
            {"slot_id": "s2", "subquestion_template": "Where was $leader born?", "answer_type": "location", "dependencies": ["s1"], "variable_bindings": [{"variable": "$leader", "source_slot": "s1"}], "output_variable": "$answer", "terminal": True, "confidence": 0.8},
        ]},
        {"answer": "Ada Lovelace", "subject": "Alpha", "relation": "led by", "answer_type": "person", "source_document_ids": ["p1"], "source_spans": ["Alpha is led by Ada Lovelace"], "confidence": 0.9},
        {"evidence_relevance": 0.9, "relation_entailment": 0.9, "answer_type_match": 0.9, "dependency_consistency": 1.0, "contradiction_detected": False, "confidence": 0.85, "reasons": []},
        {"answer": "River City", "subject": "Ada Lovelace", "relation": "born in", "answer_type": "location", "source_document_ids": ["p2"], "source_spans": ["Ada Lovelace was born in River City"], "confidence": 0.9},
        {"evidence_relevance": 0.95, "relation_entailment": 0.95, "answer_type_match": 0.95, "dependency_consistency": 0.95, "contradiction_detected": False, "confidence": 0.9, "reasons": []},
        {"answer": "River City", "confidence": 0.88, "supported": True, "reasons": []},
    ]


def example():
    return QAExample("q", "Where was the leader of Alpha born?", [
        Passage("p1", "Alpha", "Alpha is led by Ada Lovelace."),
        Passage("p2", "Ada", "Ada Lovelace was born in River City."),
    ], answers=["River City"], gold_document_ids=["p1", "p2"])


def test_two_hop_pipeline_binds_variables_and_returns_verified_answer():
    llm = DeterministicMockLLM(json_responses=responses())
    config = ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200, retriever="bm25")
    prediction, retrieval, reasoning = StructuredReasoner(llm, BM25Retriever(example().passages), config).solve(example())
    assert prediction.status == RunStatus.ANSWER
    assert prediction.answer == "River City"
    assert reasoning[1]["bound_question"] == "Where was Ada Lovelace born?"
    assert prediction.usage.llm_calls == 6


def test_infrastructure_failure_is_not_abstention():
    llm = DeterministicMockLLM(fail=True)
    config = ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200)
    prediction, _, _ = StructuredReasoner(llm, BM25Retriever(example().passages), config).solve(example())
    assert prediction.status == RunStatus.INFRASTRUCTURE_FAILURE
    assert prediction.answer is None


class MalformedPlanner(DeterministicMockLLM):
    def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
        generation = Generation("{bad", prompt_tokens=11, completion_tokens=7)
        raise StructuredOutputError("invalid JSON", generation)


def test_structured_output_failure_records_completed_provider_call_and_tokens():
    prediction, _, _ = StructuredReasoner(
        MalformedPlanner(), BM25Retriever(example().passages),
        ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200),
    ).solve(example())
    assert prediction.status == RunStatus.INFRASTRUCTURE_FAILURE
    assert prediction.stop_reason == "structured_output_failure"
    assert prediction.usage.llm_calls == 1
    assert prediction.usage.provider_calls == 1
    assert prediction.usage.total_tokens == 18


def test_invalid_structured_field_type_fails_one_question_without_batch_exception():
    malformed = responses()
    malformed[1] = {**malformed[1], "confidence": {"not": "a number"}}
    prediction, _, _ = StructuredReasoner(
        DeterministicMockLLM(json_responses=malformed), BM25Retriever(example().passages),
        ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200),
    ).solve(example())
    assert prediction.status == RunStatus.INFRASTRUCTURE_FAILURE
    assert prediction.stop_reason == "invalid_structured_payload"
    assert prediction.usage.llm_calls == 2


def test_mock_execution_is_deterministic():
    outputs = []
    for _ in range(2):
        prediction, retrieval, reasoning = StructuredReasoner(
            DeterministicMockLLM(json_responses=responses()), BM25Retriever(example().passages),
            ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200),
        ).solve(example())
        outputs.append((prediction.answer, retrieval, reasoning))
    assert outputs[0] == outputs[1]


def test_direct_finalization_ablation_changes_execution_and_saves_call():
    llm = DeterministicMockLLM(json_responses=responses()[:-1])
    config = ResearchConfig(
        max_total_tokens=5000, final_reserve_tokens=200, retriever="bm25",
        finalization="direct",
    )
    prediction, _, _ = StructuredReasoner(llm, BM25Retriever(example().passages), config).solve(example())
    assert prediction.status == RunStatus.ANSWER
    assert prediction.answer == "River City"
    assert prediction.stop_reason == "direct_terminal_candidate_ablation"
    assert prediction.usage.llm_calls == 5


def test_no_memory_ablation_collapses_dag_instead_of_only_renaming_method():
    direct_responses = [
        responses()[0],
        {"answer": "River City", "subject": "", "relation": "", "answer_type": "entity", "source_document_ids": ["p2"], "source_spans": ["Ada Lovelace was born in River City"], "confidence": 0.9},
        {"evidence_relevance": 0.9, "relation_entailment": 0.9, "answer_type_match": 0.9, "dependency_consistency": 1.0, "contradiction_detected": False, "confidence": 0.85, "reasons": []},
    ]
    config = ResearchConfig(
        max_total_tokens=5000, final_reserve_tokens=200, memory_mode="none", finalization="direct",
    )
    prediction, _, reasoning = StructuredReasoner(
        DeterministicMockLLM(json_responses=direct_responses), BM25Retriever(example().passages), config,
    ).solve(example())
    assert prediction.plan.source == "ablation_no_memory"
    assert len(prediction.plan.slots) == 1
    assert len(reasoning) == 1


class CapturingMock(DeterministicMockLLM):
    def __init__(self, json_responses):
        super().__init__(json_responses=json_responses)
        self.messages_seen = []

    def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
        self.messages_seen.append((schema_name, messages))
        return super().generate_json(messages, schema_name, max_tokens, temperature)


def test_root_question_reaches_extractor_and_verifier_without_gold_fields():
    llm = CapturingMock(responses())
    StructuredReasoner(
        llm, BM25Retriever(example().passages),
        ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200),
    ).solve(example())
    relevant = [messages[-1]["content"] for schema, messages in llm.messages_seen if schema in {"claim_candidate_v1", "claim_verification_v1"}]
    assert relevant and all("Root question: Where was the leader of Alpha born?" in content for content in relevant)
    assert all("River City" not in content.split("Root question:", 1)[1].split("Current subquestion:", 1)[0] for content in relevant)
    verifier_prompts = [messages[-1]["content"] for schema, messages in llm.messages_seen if schema == "claim_verification_v1"]
    assert "Ada Lovelace" in verifier_prompts[-1]
