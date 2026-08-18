from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class InfrastructureError(RuntimeError):
    def __init__(self, message: str, provider_attempts: int = 0) -> None:
        super().__init__(message)
        self.provider_attempts = max(0, int(provider_attempts))


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredOutputError(InfrastructureError):
    """A provider response arrived but could not be decoded as structured output."""

    def __init__(self, message: str, generation: Generation) -> None:
        super().__init__(message)
        self.generation = generation


class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate_json(self, messages: list[dict[str, str]], schema_name: str, max_tokens: int, temperature: float = 0.0) -> tuple[dict[str, Any], Generation]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.0) -> Generation:
        raise NotImplementedError
