from __future__ import annotations

from collections import deque
from typing import Any

from ..utils import stable_hash
from .base import BaseLLM, Generation, InfrastructureError


class DeterministicMockLLM(BaseLLM):
    model_name = "deterministic-mock"

    def __init__(self, json_responses: list[dict[str, Any]] | None = None, text_responses: list[str] | None = None, fail: bool = False) -> None:
        self.json_responses = deque(json_responses or [])
        self.text_responses = deque(text_responses or [])
        self.fail = fail

    def _usage(self, messages: list[dict[str, str]], text: str) -> Generation:
        prompt = " ".join(message.get("content", "") for message in messages)
        return Generation(text=text, prompt_tokens=max(1, len(prompt.split())), completion_tokens=max(1, len(text.split())), metadata={"fingerprint": stable_hash(messages)})

    def generate_json(self, messages: list[dict[str, str]], schema_name: str, max_tokens: int, temperature: float = 0.0) -> tuple[dict[str, Any], Generation]:
        if self.fail:
            raise InfrastructureError("mock infrastructure failure")
        if not self.json_responses:
            raise InfrastructureError(f"no mock JSON response for {schema_name}")
        value = self.json_responses.popleft()
        return value, self._usage(messages, str(value))

    def generate_text(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.0) -> Generation:
        if self.fail:
            raise InfrastructureError("mock infrastructure failure")
        if not self.text_responses:
            raise InfrastructureError("no mock text response")
        return self._usage(messages, self.text_responses.popleft())

