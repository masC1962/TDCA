from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..utils import safe_error, stable_hash, write_json
from .base import BaseLLM, Generation, InfrastructureError, StructuredOutputError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self, base_url: str, model_name: str, cache_dir: str, prompt_version: str,
        request_timeout_seconds: float = 120.0, max_api_attempts: int = 3,
    ) -> None:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise InfrastructureError("LLM_API_KEY or DASHSCOPE_API_KEY is required")
        self.model_name = model_name
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_version = prompt_version
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_api_attempts = int(max_api_attempts)
        # One explicit retry layer only; SDK retries would otherwise multiply the
        # configured attempt count and make latency/call accounting ambiguous.
        self.client = OpenAI(
            base_url=base_url, api_key=api_key,
            timeout=self.request_timeout_seconds, max_retries=0,
        )

    def _cache_key(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, schema_name: str = "") -> str:
        return stable_hash({
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "schema_name": schema_name,
            "response_format": "json_object" if schema_name else "text",
            "prompt_version": self.prompt_version,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_api_attempts": self.max_api_attempts,
        })

    def _request(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, schema_name: str = "") -> Generation:
        key = self._cache_key(messages, max_tokens, temperature, schema_name)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return Generation(**payload, cached=True)
        last_error: Exception | None = None
        for attempt in range(self.max_api_attempts):
            try:
                request = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if schema_name:
                    request["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    **request,
                )
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                generation = Generation(
                    text=(choice.message.content or "").strip(),
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    finish_reason=str(choice.finish_reason or ""),
                    metadata={"schema_name": schema_name, "provider_attempts": attempt + 1},
                )
                if not generation.text:
                    raise InfrastructureError("API returned an empty response")
                write_json(cache_path, {
                    "text": generation.text,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "finish_reason": generation.finish_reason,
                    "metadata": generation.metadata,
                })
                return generation
            except Exception as exc:
                last_error = exc
                if attempt < self.max_api_attempts - 1:
                    time.sleep(2**attempt)
        raise InfrastructureError(
            safe_error(last_error or RuntimeError("unknown API failure")),
            provider_attempts=self.max_api_attempts,
        )

    def generate_text(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.0) -> Generation:
        return self._request(messages, max_tokens, temperature)

    def generate_json(self, messages: list[dict[str, str]], schema_name: str, max_tokens: int, temperature: float = 0.0) -> tuple[dict[str, Any], Generation]:
        key = self._cache_key(messages, max_tokens, temperature, schema_name)
        generation = self._request(messages, max_tokens, temperature, schema_name)
        try:
            value = self._parse_json(generation.text)
        except json.JSONDecodeError as exc:
            # Invalid structured output is not a successful cached response. Do not
            # hide an extra provider call inside this method: callers own the budget.
            (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
            raise StructuredOutputError(
                f"invalid JSON for {schema_name}: {safe_error(exc)}",
                generation,
            ) from exc
        if not isinstance(value, dict):
            (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
            raise StructuredOutputError(f"JSON for {schema_name} must be an object", generation)
        return value, generation

    @staticmethod
    def _parse_json(raw: str) -> Any:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
        return json.loads(text)
