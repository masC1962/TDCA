import pytest

from tdca_research.budget import Budget, BudgetExceeded
from tdca_research.models import Usage, prediction_from_dict
from tdca_research.llm import Generation, InfrastructureError


def test_intermediate_calls_preserve_final_reserve():
    budget = Budget(max_llm_calls=3, max_total_tokens=100, final_reserve_tokens=30, usage=Usage(completion_tokens=60))
    with pytest.raises(BudgetExceeded):
        budget.require(20)
    budget.require(20, final=True)


def test_prompt_estimate_is_included_before_provider_call():
    budget = Budget(max_llm_calls=3, max_total_tokens=100, final_reserve_tokens=10, usage=Usage())
    with pytest.raises(BudgetExceeded):
        budget.require(20, estimated_prompt_tokens=75)


def test_provider_attempts_distinguish_retries_and_cache_hits():
    usage = Usage()
    budget = Budget(3, 1000, 0, usage)
    budget.record_generation(Generation("ok", 10, 2, metadata={"provider_attempts": 2}))
    budget.record_generation(Generation("cached", 10, 2, cached=True))
    budget.record_infrastructure_failure(InfrastructureError("timeout", provider_attempts=3))
    assert usage.llm_calls == 2
    assert usage.cache_hits == 1
    assert usage.provider_calls == 5
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 4
    assert usage.provider_prompt_tokens == 10
    assert usage.provider_completion_tokens == 2


def test_historical_prediction_rehydrates_uncached_logical_calls_as_provider_calls():
    prediction = prediction_from_dict({
        "qid": "q", "question": "?", "status": "answer", "answer": "a",
        "confidence": 0.5, "stop_reason": "done",
        "usage": {"llm_calls": 4, "cache_hits": 1},
    })
    assert prediction.usage.provider_calls == 3
