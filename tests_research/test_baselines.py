from tdca_research.baselines.ircot import _generate_step_with_one_repair
from tdca_research.baselines.simple import run_closed_book
from tdca_research.budget import Budget
from tdca_research.llm import BaseLLM, DeterministicMockLLM, Generation, StructuredOutputError
from tdca_research.models import QAExample, RunStatus
from tdca_research.models import Usage


def test_closed_book_infrastructure_failure_is_not_a_batch_crash_or_abstention():
    prediction = run_closed_book(QAExample("q", "Question?", []), DeterministicMockLLM(fail=True), 2, 2000)
    assert prediction.status == RunStatus.INFRASTRUCTURE_FAILURE
    assert prediction.answer is None


class _MalformedThenValidLLM(BaseLLM):
    model_name = "malformed-then-valid"

    def __init__(self):
        self.calls = 0

    def generate_json(self, messages, schema_name, max_tokens, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            generation = Generation('{"reasoning":"cut', 10, 400, "length")
            raise StructuredOutputError("invalid JSON", generation)
        generation = Generation('{"reasoning":"ok","next_query":"q","final_answer":""}', 8, 12, "stop")
        return {"reasoning": "ok", "next_query": "q", "final_answer": ""}, generation

    def generate_text(self, messages, max_tokens, temperature=0.0):
        raise AssertionError("not used")


def test_ircot_structured_output_repair_is_explicitly_budgeted():
    usage = Usage()
    budget = Budget(4, 2000, 0, usage)
    llm = _MalformedThenValidLLM()
    data = _generate_step_with_one_repair(
        llm,
        budget,
        [{"role": "user", "content": "question and evidence"}],
        400,
        0.0,
    )
    assert data["next_query"] == "q"
    assert llm.calls == 2
    assert usage.llm_calls == 2
    assert usage.completion_tokens == 412
