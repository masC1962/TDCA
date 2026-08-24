from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..campaign import CampaignBudgetExceeded, CampaignBudgetLedger
from ..utils import safe_error, stable_hash, write_json
from .base import (
    BaseLLM,
    Generation,
    InfrastructureError,
    ProviderRefusalError,
    StructuredOutputError,
)


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self, base_url: str, model_name: str, cache_dir: str, prompt_version: str,
        request_timeout_seconds: float = 120.0, max_api_attempts: int = 3,
        campaign_ledger: CampaignBudgetLedger | None = None,
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
        self.campaign_ledger = campaign_ledger
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
            reservation_id = None
            try:
                if self.campaign_ledger is not None:
                    # UTF-8 bytes plus the requested completion bound is a
                    # conservative provider-token reservation independent of a
                    # local tokenizer implementation.
                    prompt_upper_bound = len(json.dumps(
                        messages, ensure_ascii=False, separators=(",", ":"),
                    ).encode("utf-8"))
                    reservation_id = self.campaign_ledger.reserve(
                        cache_key=key,
                        cache_path=cache_path,
                        reserved_tokens=prompt_upper_bound + int(max_tokens),
                    )
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
                settlement = None
                if self.campaign_ledger is not None and reservation_id is not None:
                    settlement = self.campaign_ledger.settle(
                        reservation_id,
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                        outcome="success",
                    )
                    reservation_id = None
                if not generation.text:
                    raise InfrastructureError("API returned an empty response")
                write_json(cache_path, {
                    "text": generation.text,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "finish_reason": generation.finish_reason,
                    "metadata": generation.metadata,
                })
                if settlement is not None and settlement["over_cap"]:
                    raise CampaignBudgetExceeded(
                        "campaign provider token cap crossed by settled response",
                        settlement["snapshot"],
                    )
                return generation
            except CampaignBudgetExceeded:
                raise
            except Exception as exc:
                last_error = exc
                if self.campaign_ledger is not None and reservation_id is not None:
                    self.campaign_ledger.settle(
                        reservation_id, outcome=f"error:{type(exc).__name__}",
                    )
                if _is_provider_refusal(exc):
                    # Policy refusals are deterministic for an identical prompt;
                    # retrying only spends calls and still should not fail a whole
                    # reasoning episode as an infrastructure outage.
                    raise ProviderRefusalError(safe_error(exc), provider_attempts=attempt + 1) from exc
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
            value = self._parse_json(
                generation.text,
                allow_complete_array_prefix=generation.finish_reason in {"length", "max_tokens"},
            )
        except json.JSONDecodeError as exc:
            # Invalid structured output is not a successful cached response. Do not
            # hide an extra provider call inside this method: callers own the budget.
            (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
            raise StructuredOutputError(
                f"invalid JSON for {schema_name}: {safe_error(exc)}",
                generation,
            ) from exc
        if isinstance(value, list):
            root_key = _declared_array_root_key(schema_name)
            if root_key:
                value = {root_key: value}
        if not isinstance(value, dict):
            (self.cache_dir / f"{key}.json").unlink(missing_ok=True)
            raise StructuredOutputError(f"JSON for {schema_name} must be an object", generation)
        return value, generation

    @staticmethod
    def _parse_json(raw: str, *, allow_complete_array_prefix: bool = False) -> Any:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as original:
            # Deterministic syntax-only recovery.  Providers occasionally wrap a
            # valid object in a short preamble or leave a trailing comma.  This
            # never invents, deletes, or rewrites semantic field content.
            start = text.find("{")
            if start < 0:
                raise original
            candidate = re.sub(r",\s*([}\]])", r"\1", text[start:])
            try:
                value, _ = json.JSONDecoder().raw_decode(candidate)
                return value
            except json.JSONDecodeError:
                if allow_complete_array_prefix:
                    recovered = _complete_object_array_prefix(candidate)
                    if recovered is not None:
                        return recovered
                raise original


def _declared_array_root_key(schema_name: str) -> str:
    """Wrap a bare provider array only when the schema declares its root key."""
    prefixes = {
        "dynamic_v2_event_graph_editor": "operations",
        "dynamic_v2_typed_claim_extraction": "claims",
        "dynamic_v2_goal_conditioned_answer_projection": "claims",
        "dynamic_v2_independent_verification": "scores",
    }
    return next((
        root_key for prefix, root_key in prefixes.items()
        if str(schema_name).startswith(prefix)
    ), "")


def _complete_object_array_prefix(text: str) -> dict[str, list[dict[str, Any]]] | None:
    """Recover only fully decoded rows from a length-truncated claims/scores array.

    No incomplete row is repaired and no semantic field is invented.  Recovery
    is deliberately restricted to the two batched schemas whose consumers
    already define deterministic handling for omitted rows.
    """
    match = re.match(r'\s*\{\s*"(claims|scores)"\s*:\s*\[', text)
    if match is None:
        return None
    key = match.group(1)
    decoder = json.JSONDecoder()
    cursor = match.end()
    rows: list[dict[str, Any]] = []
    while cursor < len(text):
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            return None
        rows.append(value)
        cursor = end
    return {key: rows} if rows else None


def _is_provider_refusal(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return any(code in text for code in (
        "data_inspection_failed",
        "inappropriate content",
        "content_policy_violation",
        "content_filter",
        "safety refusal",
    ))
