from tdca_research.config import ResearchConfig
from tdca_research.data import load_examples
from tdca_research.llm import DeterministicMockLLM
from tdca_research.models import RunStatus
from tdca_research.reasoning import StructuredReasoner
from tdca_research.retrieval import BM25Retriever


class EchoGroundedMock(DeterministicMockLLM):
    """Structural offline smoke, deliberately not a quality benchmark."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
        self.calls += 1
        question = messages[-1]["content"]
        if schema_name == "reasoning_plan_v1":
            value = {"plan_type": "direct_fallback", "slots": [{
                "slot_id": "root", "subquestion_template": question,
                "answer_type": "entity", "dependencies": [], "variable_bindings": [],
                "output_variable": "$answer", "terminal": True, "confidence": 0.2,
            }]}
        elif schema_name == "claim_candidate_v1":
            document_id = question.split("[")[1].split("]")[0]
            text = question.split("\n", 1)[1]
            span = next((line.strip() for line in text.splitlines() if line.strip() and not line.startswith("Expected") and not line.startswith("Passages")), "evidence")
            value = {"answer": "candidate", "subject": "root", "relation": "answer", "answer_type": "entity", "source_document_ids": [document_id], "source_spans": [span[:80]], "confidence": 0.3}
        elif schema_name == "claim_verification_v1":
            value = {"evidence_relevance": 0.2, "relation_entailment": 0.2, "answer_type_match": 0.5, "dependency_consistency": 1.0, "contradiction_detected": False, "confidence": 0.2, "reasons": ["offline smoke"]}
        else:
            value = {"answer": "", "confidence": 0.0, "supported": False, "reasons": ["offline smoke"]}
        return value, self._usage(messages, str(value))


def test_twenty_real_rows_finish_with_explicit_nonempty_status_and_within_budget():
    examples = load_examples("data/musique_subset_50.jsonl", "musique")[:20]
    for example in examples:
        config = ResearchConfig(max_total_tokens=5000, final_reserve_tokens=200, max_llm_calls=8, max_steps=2)
        prediction, retrieval, reasoning = StructuredReasoner(EchoGroundedMock(), BM25Retriever(example.passages), config).solve(example)
        assert prediction.status in {RunStatus.ANSWER, RunStatus.ABSTAIN, RunStatus.INFRASTRUCTURE_FAILURE}
        assert prediction.stop_reason
        assert prediction.answer != ""
        assert prediction.usage.llm_calls <= config.max_llm_calls
        assert prediction.usage.total_tokens <= config.max_total_tokens
        assert retrieval

