from __future__ import annotations

from dataclasses import dataclass

from .models import Usage


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_llm_calls: int
    max_total_tokens: int
    final_reserve_tokens: int
    usage: Usage

    def can_call(self, requested_completion: int, *, estimated_prompt_tokens: int = 0, final: bool = False) -> bool:
        if self.usage.llm_calls >= self.max_llm_calls:
            return False
        reserve = 0 if final else self.final_reserve_tokens
        return self.usage.total_tokens + max(0, estimated_prompt_tokens) + requested_completion + reserve <= self.max_total_tokens

    def require(self, requested_completion: int, *, estimated_prompt_tokens: int = 0, final: bool = False) -> None:
        if not self.can_call(requested_completion, estimated_prompt_tokens=estimated_prompt_tokens, final=final):
            raise BudgetExceeded("LLM call would violate call/token budget or final reserve")

    def record_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += max(0, int(prompt_tokens))
        self.usage.completion_tokens += max(0, int(completion_tokens))
        if self.usage.total_tokens > self.max_total_tokens:
            raise BudgetExceeded("API-reported usage exceeded the configured total token budget")

    def record_generation(self, generation) -> None:
        self.record_llm(generation.prompt_tokens, generation.completion_tokens)
        cached = bool(getattr(generation, "cached", False))
        self.usage.cache_hits += int(cached)
        if not cached:
            self.usage.provider_attempts += max(
                1, int(getattr(generation, "metadata", {}).get("provider_attempts", 1)),
            )
            self.usage.provider_prompt_tokens += max(0, int(generation.prompt_tokens))
            self.usage.provider_completion_tokens += max(0, int(generation.completion_tokens))

    def record_infrastructure_failure(self, error: BaseException) -> None:
        self.usage.provider_attempts += max(0, int(getattr(error, "provider_attempts", 0)))

    def record_retrieval(self) -> None:
        self.usage.retrieval_calls += 1
