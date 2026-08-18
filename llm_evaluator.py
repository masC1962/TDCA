from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from core_models import Node, NodeType, RetrievedContext
from prompts import build_scoring_prompt
from utils import clamp, extract_json_block, simple_tokenize, strip_think_blocks


@dataclass
class LLMGeneration:
    text: str
    raw_text: str
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def generation_empty(self) -> bool:
        return not bool((self.text or "").strip())


class BaseLLM:
    def __init__(self) -> None:
        self.call_count = 0
        self.total_generated_tokens = 0
        self.last_generation: LLMGeneration | None = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        raise NotImplementedError

    def generate_with_info(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> LLMGeneration:
        try:
            text = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        except Exception as exc:
            error_result = LLMGeneration(
                text="",
                raw_text="",
                finish_reason="",
                usage={},
                error=str(exc),
                debug={"exception_type": type(exc).__name__},
            )
            self.last_generation = error_result
            return error_result

        if isinstance(self.last_generation, LLMGeneration):
            return self.last_generation

        fallback = LLMGeneration(text=text or "", raw_text=text or "")
        self.last_generation = fallback
        return fallback

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int,
        default: Dict[str, Any],
        temperature: float = 0.1,
        do_sample: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        # Attempt 1: original prompt
        attempts = [
            prompt,
            prompt + "\n\nIMPORTANT: Return ONLY one valid JSON object. No markdown. No prose. No code fences.",
            prompt + "\n\nIMPORTANT: The previous response was invalid. Fix it and return ONLY one valid JSON object that matches the required schema exactly.",
        ]
        for i, p in enumerate(attempts[:max_retries]):
            raw = self.generate(
                prompt=p,
                max_new_tokens=max_new_tokens,
                temperature=temperature if i == 0 else 0.0,
                do_sample=do_sample if i == 0 else False,
            )
            parsed = extract_json_block(raw)
            if isinstance(parsed, dict):
                return parsed
            # try direct json parse as fallback
            try:
                maybe = json.loads(strip_think_blocks(raw).strip())
                if isinstance(maybe, dict):
                    return maybe
            except Exception:
                pass
        return default


class MockLLM(BaseLLM):
    @staticmethod
    def _mock_answer_from_prompt(prompt: str) -> str:
        lower = prompt.lower()
        if "what is 2+2" in lower:
            return "4"

        context_match = None
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "]: " not in stripped:
                # Matches: [1] Title: text
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    title = re.sub(r"^\[\d+\]\s*", "", parts[0]).strip()
                    if title:
                        context_match = title
                        break
        if context_match:
            return context_match

        question_match = re.search(r"Question:\s*(.+)", prompt, flags=re.I)
        if question_match:
            question = question_match.group(1).strip().rstrip("?")
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", question) if len(w) > 2]
            if words:
                return " ".join(words[:4])

        return "test"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        self.call_count += 1
        self.total_generated_tokens += min(max_new_tokens, 80)
        lower = prompt.lower()
        if '"sub_questions"' in prompt or 'tdca' in lower:
            raw = """{
  "sub_questions": [
    {"text": "Who directed the movie Inception?", "kind": "bridge", "priority": 0.96},
    {"text": "Where was Christopher Nolan born?", "kind": "retrieval", "priority": 0.90}
  ],
  "candidate_answer": "London",
  "stop": false,
  "confidence": 0.72
}"""
        elif '"task_progress"' in prompt:
            raw = """{
  "task_progress": 0.82,
  "evidence_support": 0.86,
  "memory_usefulness": 0.68,
  "answerability": 0.70,
  "uncertainty": 0.12
}"""
        else:
            raw = f"Final Answer: {self._mock_answer_from_prompt(prompt)}"

        self.last_generation = LLMGeneration(
            text=raw,
            raw_text=raw,
            finish_reason="stop",
            usage={
                "prompt_tokens": max(1, len(simple_tokenize(prompt))),
                "completion_tokens": min(max_new_tokens, 80),
                "total_tokens": max(1, len(simple_tokenize(prompt))) + min(max_new_tokens, 80),
            },
        )
        return raw


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        default_headers: Dict[str, str] | None = None,
        provider_preferences: Dict[str, Any] | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        super().__init__()
        from openai import OpenAI

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "No API key found. Set DASHSCOPE_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY "
                "before running with an OpenAI-compatible API backend."
            )
        self.api_key = resolved_api_key
        self.default_headers = {k: v for k, v in (default_headers or {}).items() if v}
        self.provider_preferences = provider_preferences or None
        self.reasoning_effort = (reasoning_effort or "none").strip() or "none"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=resolved_api_key,
            default_headers=self.default_headers or None,
        )

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        name = (model_name or "").strip().lower()
        if "/" in name:
            name = name.split("/")[-1]
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    @staticmethod
    def _message_to_text(message: Any) -> str:
        if message is None:
            return ""

        content = getattr(message, "content", None)
        if content in (None, "") and isinstance(message, dict):
            content = message.get("content")
        if content:
            return OpenAICompatibleLLM._coerce_text(content)

        reasoning = getattr(message, "reasoning", None)
        if reasoning in (None, "") and isinstance(message, dict):
            reasoning = message.get("reasoning")
        if reasoning:
            return OpenAICompatibleLLM._coerce_text(reasoning)

        refusal = getattr(message, "refusal", None)
        if refusal in (None, "") and isinstance(message, dict):
            refusal = message.get("refusal")
        if refusal:
            return OpenAICompatibleLLM._coerce_text(refusal)

        extras = OpenAICompatibleLLM._model_extras(message)
        if extras:
            for key in [
                "content",
                "output_text",
                "text",
                "reasoning",
                "refusal",
                "output",
                "response",
                "message",
            ]:
                txt = OpenAICompatibleLLM._coerce_text(extras.get(key))
                if txt:
                    return txt

        return ""

    @staticmethod
    def _model_extras(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {}
        extras = getattr(value, "model_extra", None)
        if isinstance(extras, dict) and extras:
            return extras
        extras = getattr(value, "__pydantic_extra__", None)
        if isinstance(extras, dict) and extras:
            return extras
        return {}

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [OpenAICompatibleLLM._to_plain(item) for item in value]
        if isinstance(value, dict):
            return {str(k): OpenAICompatibleLLM._to_plain(v) for k, v in value.items()}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return OpenAICompatibleLLM._to_plain(model_dump(exclude_none=True))
            except TypeError:
                return OpenAICompatibleLLM._to_plain(model_dump())
        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                return OpenAICompatibleLLM._to_plain(dict_method())
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            raw = {k: v for k, v in vars(value).items() if not k.startswith("_")}
            if raw:
                return OpenAICompatibleLLM._to_plain(raw)
        return str(value)

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                txt = OpenAICompatibleLLM._coerce_text(item)
                if txt:
                    parts.append(txt)
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            for key in ["text", "content", "reasoning", "output_text"]:
                txt = OpenAICompatibleLLM._coerce_text(value.get(key))
                if txt:
                    return txt
            if value.get("type") == "text":
                txt = OpenAICompatibleLLM._coerce_text(value.get("text"))
                if txt:
                    return txt
            return ""
        for attr in ["text", "content", "reasoning", "output_text"]:
            txt = OpenAICompatibleLLM._coerce_text(getattr(value, attr, None))
            if txt:
                return txt
        extras = OpenAICompatibleLLM._model_extras(value)
        if extras:
            for key in ["text", "content", "reasoning", "output_text", "refusal", "output", "response"]:
                txt = OpenAICompatibleLLM._coerce_text(extras.get(key))
                if txt:
                    return txt
        return ""

    @staticmethod
    def _get_field(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _usage_to_dict(self, usage: Any) -> Dict[str, Any]:
        plain = self._to_plain(usage)
        if isinstance(plain, dict):
            return plain
        return {}

    def _completion_tokens_from_usage(self, usage_dict: Dict[str, Any], text: str, max_new_tokens: int) -> int:
        for key in ["completion_tokens", "output_tokens", "generated_tokens"]:
            value = usage_dict.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return min(max_new_tokens, max(1, len(simple_tokenize(text or ""))))

    def _reasoning_payload(self) -> Dict[str, Any]:
        return {
            "effort": self.reasoning_effort,
            "exclude": True,
        }

    def _should_try_responses_api(self) -> bool:
        # OpenRouter-style proxies in this project expose chat completions reliably,
        # while their /responses compatibility can return provider wrapper metadata.
        return "api.openai.com" in (self.base_url or "").lower()

    def _should_send_provider_preferences(self) -> bool:
        return "openrouter.ai" in (self.base_url or "").lower()

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return str(exc or "")

    def _build_request_kwargs(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        response_format: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format:
            request_kwargs["response_format"] = response_format

        extra_body: Dict[str, Any] = {}
        if self.provider_preferences and self._should_send_provider_preferences():
            extra_body["provider"] = self.provider_preferences

        if self._is_reasoning_model(self.model_name):
            request_kwargs["max_completion_tokens"] = max_new_tokens
            request_kwargs["reasoning_effort"] = self.reasoning_effort
            # Some OpenAI-compatible SDK versions reject unknown top-level args like
            # `reasoning`, so send it through extra_body for OpenRouter-style APIs.
            extra_body["reasoning"] = self._reasoning_payload()
        else:
            request_kwargs["max_tokens"] = max_new_tokens
            request_kwargs["temperature"] = max(temperature, 1e-5) if do_sample else 0.0

        if extra_body:
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _build_responses_request_kwargs(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        response_format: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
        }
        if not self._is_reasoning_model(self.model_name):
            request_kwargs["temperature"] = max(temperature, 1e-5) if do_sample else 0.0
        elif self.reasoning_effort:
            request_kwargs["reasoning"] = self._reasoning_payload()

        if response_format and response_format.get("type") == "json_object":
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        extra_body: Dict[str, Any] = {}
        if self.provider_preferences and self._should_send_provider_preferences():
            extra_body["provider"] = self.provider_preferences
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _request_once(self, request_kwargs: Dict[str, Any]) -> Tuple[Any, str]:
        self.call_count += 1
        raw_response = self.client.chat.completions.with_raw_response.create(**request_kwargs)
        raw_body = raw_response.text
        return raw_response.parse(), raw_body

    def _request_once_responses(self, request_kwargs: Dict[str, Any]) -> Tuple[Any, str]:
        responses_api = getattr(self.client, "responses", None)
        if responses_api is None:
            raise RuntimeError("responses_api_unavailable")
        self.call_count += 1
        raw_response = responses_api.with_raw_response.create(**request_kwargs)
        raw_body = raw_response.text
        return raw_response.parse(), raw_body

    def _request_once_responses_http(self, request_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        self.call_count += 1
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers or {})
        body = json.dumps(request_kwargs).encode("utf-8")
        req = urllib_request.Request(url=url, data=body, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=120) as resp:
            raw_body = resp.read().decode("utf-8")
        return json.loads(raw_body), raw_body

    @staticmethod
    def _looks_like_retryable_error(error_text: str) -> bool:
        lower = (error_text or "").lower()
        return (
            "503" in lower
            or "qps/tpm" in lower
            or "maximum sustainable qps" in lower
            or "peak periods" in lower
            or "rate limit" in lower
            or "timeout" in lower
        )

    def _request_with_retries(self, request_kwargs: Dict[str, Any], max_attempts: int = 3) -> Tuple[Any, str]:
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self._request_once(request_kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts - 1 or not self._looks_like_retryable_error(self._error_text(exc)):
                    break
                time.sleep(min(4.0, 1.0 * (2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected empty retry state")

    def _request_with_retries_responses(self, request_kwargs: Dict[str, Any], max_attempts: int = 2) -> Tuple[Any, str]:
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self._request_once_responses(request_kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts - 1 or not self._looks_like_retryable_error(self._error_text(exc)):
                    break
                time.sleep(min(4.0, 1.0 * (2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected empty retry state")

    def _request_with_retries_responses_http(self, request_kwargs: Dict[str, Any], max_attempts: int = 2) -> Tuple[Dict[str, Any], str]:
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self._request_once_responses_http(request_kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts - 1 or not self._looks_like_retryable_error(self._error_text(exc)):
                    break
                time.sleep(min(4.0, 1.0 * (2 ** attempt)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected empty retry state")

    def _create_chat_completion(self, request_kwargs: Dict[str, Any]) -> Tuple[Any, str]:
        try:
            return self._request_with_retries(request_kwargs)
        except Exception as exc:
            error_text = self._error_text(exc)
            extra_body = request_kwargs.get("extra_body")
            if (
                isinstance(extra_body, dict)
                and isinstance(extra_body.get("reasoning"), dict)
                and "exclude" in extra_body["reasoning"]
                and "reasoning.exclude" in error_text
                and "unknown_parameter" in error_text
            ):
                fallback_kwargs = dict(request_kwargs)
                fallback_extra_body = dict(extra_body)
                fallback_reasoning = dict(fallback_extra_body["reasoning"])
                fallback_reasoning.pop("exclude", None)
                fallback_extra_body["reasoning"] = fallback_reasoning
                fallback_kwargs["extra_body"] = fallback_extra_body
                return self._request_with_retries(fallback_kwargs)
            if "reasoning_effort" in request_kwargs and "reasoning_effort" in error_text and "unknown_parameter" in error_text:
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("reasoning_effort", None)
                return self._request_with_retries(fallback_kwargs)
            if "max_completion_tokens" in request_kwargs:
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs["max_tokens"] = fallback_kwargs.pop("max_completion_tokens")
                return self._request_with_retries(fallback_kwargs)
            raise

    def _create_responses_completion(self, request_kwargs: Dict[str, Any]) -> Tuple[Any, str]:
        try:
            return self._request_with_retries_responses(request_kwargs)
        except Exception as exc:
            error_text = self._error_text(exc)
            extra_body = request_kwargs.get("extra_body")
            if (
                isinstance(extra_body, dict)
                and isinstance(extra_body.get("reasoning"), dict)
                and "exclude" in extra_body["reasoning"]
                and "reasoning.exclude" in error_text
                and "unknown_parameter" in error_text
            ):
                fallback_kwargs = dict(request_kwargs)
                fallback_extra_body = dict(extra_body)
                fallback_reasoning = dict(fallback_extra_body["reasoning"])
                fallback_reasoning.pop("exclude", None)
                fallback_extra_body["reasoning"] = fallback_reasoning
                fallback_kwargs["extra_body"] = fallback_extra_body
                return self._request_with_retries_responses(fallback_kwargs)
            if "reasoning" in request_kwargs and "reasoning" in error_text and "unknown_parameter" in error_text:
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("reasoning", None)
                return self._request_with_retries_responses(fallback_kwargs)
            raise

    def _create_responses_completion_http(self, request_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            return self._request_with_retries_responses_http(request_kwargs)
        except Exception as exc:
            error_text = self._error_text(exc)
            if isinstance(exc, urllib_error.HTTPError):
                try:
                    error_text = exc.read().decode("utf-8") or error_text
                except Exception:
                    pass
            if (
                isinstance(request_kwargs.get("reasoning"), dict)
                and "exclude" in request_kwargs["reasoning"]
                and "reasoning.exclude" in error_text
                and "unknown_parameter" in error_text
            ):
                fallback_kwargs = dict(request_kwargs)
                fallback_reasoning = dict(fallback_kwargs["reasoning"])
                fallback_reasoning.pop("exclude", None)
                if fallback_reasoning:
                    fallback_kwargs["reasoning"] = fallback_reasoning
                else:
                    fallback_kwargs.pop("reasoning", None)
                return self._request_with_retries_responses_http(fallback_kwargs)
            if "reasoning" in request_kwargs and "reasoning" in error_text and "unknown_parameter" in error_text:
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("reasoning", None)
                return self._request_with_retries_responses_http(fallback_kwargs)
            raise

    @staticmethod
    def _extract_response_output_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ["output_text", "text", "content", "reasoning", "summary", "message"]:
                txt = OpenAICompatibleLLM._coerce_text(value.get(key))
                if txt:
                    return txt
            output = value.get("output")
            if isinstance(output, list):
                for item in output:
                    txt = OpenAICompatibleLLM._extract_response_output_text(item)
                    if txt:
                        return txt
            content = value.get("content")
            if isinstance(content, list):
                for item in content:
                    txt = OpenAICompatibleLLM._extract_response_output_text(item)
                    if txt:
                        return txt
            return ""
        if isinstance(value, list):
            for item in value:
                txt = OpenAICompatibleLLM._extract_response_output_text(item)
                if txt:
                    return txt
            return ""
        txt = OpenAICompatibleLLM._coerce_text(getattr(value, "output_text", None))
        if txt:
            return txt
        for attr in ["text", "content", "reasoning", "summary", "message", "output"]:
            txt = OpenAICompatibleLLM._extract_response_output_text(getattr(value, attr, None))
            if txt:
                return txt
        return ""

    def _extract_generation(
        self,
        response: Any,
        max_new_tokens: int,
        raw_response_text: str = "",
        *,
        count_tokens: bool = True,
    ) -> LLMGeneration:
        response_dict: Any = None
        if raw_response_text:
            try:
                response_dict = json.loads(raw_response_text)
            except Exception:
                response_dict = None
        if not isinstance(response_dict, dict):
            response_dict = self._to_plain(response)
        choices = self._get_field(response, "choices", None)
        if choices is None and isinstance(response_dict, dict):
            choices = response_dict.get("choices") or []
        choice = choices[0] if choices else None
        message = self._get_field(choice, "message")
        raw_text = self._message_to_text(message)
        if not raw_text:
            raw_text = self._coerce_text(self._get_field(choice, "text"))
        if not raw_text and isinstance(response_dict, dict):
            raw_text = self._coerce_text(response_dict.get("output_text"))
        if not raw_text:
            raw_text = self._coerce_text(self._model_extras(choice))
        if not raw_text:
            raw_text = self._coerce_text(self._model_extras(message))
        if not raw_text and isinstance(response_dict, dict):
            raw_text = self._coerce_text(response_dict.get("output"))
        if not raw_text and isinstance(response_dict, dict):
            raw_text = self._coerce_text(response_dict.get("response"))
        used_debug_fallback = False
        if not raw_text:
            choice_plain = self._to_plain(choice)
            if isinstance(choice_plain, dict):
                debug_payload = {
                    key: value
                    for key, value in choice_plain.items()
                    if key not in {"index", "logprobs"} and value not in (None, "", [], {})
                }
                if debug_payload:
                    raw_text = json.dumps(debug_payload, ensure_ascii=False)
                    used_debug_fallback = True
        if raw_response_text and (not raw_text or used_debug_fallback):
            raw_text = raw_response_text.strip()
            used_debug_fallback = True

        finish_reason = str(self._get_field(choice, "finish_reason", "") or "")
        usage = self._usage_to_dict(self._get_field(response, "usage"))
        text = "" if used_debug_fallback else strip_think_blocks((raw_text or "").strip())
        completion_tokens = self._completion_tokens_from_usage(usage, text, max_new_tokens)
        if count_tokens:
            self.total_generated_tokens += int(completion_tokens)

        return LLMGeneration(
            text=text,
            raw_text=(raw_text or "").strip(),
            finish_reason=finish_reason,
            usage=usage,
        )

    def _extract_responses_generation(self, response: Any, max_new_tokens: int, raw_response_text: str = "") -> LLMGeneration:
        response_dict: Any = None
        if raw_response_text:
            try:
                response_dict = json.loads(raw_response_text)
            except Exception:
                response_dict = None
        if not isinstance(response_dict, dict):
            response_dict = self._to_plain(response)

        raw_text = self._coerce_text(getattr(response, "output_text", None))
        if not raw_text and isinstance(response_dict, dict):
            raw_text = self._coerce_text(response_dict.get("output_text"))
        if not raw_text:
            raw_text = self._extract_response_output_text(response)
        if not raw_text and isinstance(response_dict, dict):
            raw_text = self._extract_response_output_text(response_dict)

        finish_reason = str(
            getattr(response, "status", None)
            or (response_dict.get("status") if isinstance(response_dict, dict) else "")
            or ""
        )
        usage = self._usage_to_dict(
            getattr(response, "usage", None)
            or (response_dict.get("usage") if isinstance(response_dict, dict) else None)
        )

        used_debug_fallback = False
        if not raw_text and raw_response_text:
            raw_text = raw_response_text.strip()
            used_debug_fallback = True

        text = "" if used_debug_fallback else strip_think_blocks((raw_text or "").strip())
        completion_tokens = self._completion_tokens_from_usage(usage, text, max_new_tokens)
        self.total_generated_tokens += int(completion_tokens)

        return LLMGeneration(
            text=text,
            raw_text=(raw_text or "").strip(),
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _with_debug(generation: LLMGeneration, extra_debug: Dict[str, Any]) -> LLMGeneration:
        merged = dict(generation.debug or {})
        merged.update({k: v for k, v in (extra_debug or {}).items() if v not in (None, "", [], {})})
        return LLMGeneration(
            text=generation.text,
            raw_text=generation.raw_text,
            finish_reason=generation.finish_reason,
            usage=generation.usage,
            error=generation.error,
            debug=merged,
        )

    def _empty_generation_recovery(self, request_kwargs: Dict[str, Any], max_new_tokens: int) -> Tuple[Any | None, str, Dict[str, Any]] | None:
        attempts: List[Dict[str, Any]] = []
        candidates: List[Tuple[str, Dict[str, Any]]] = []

        def add_candidate(label: str, candidate: Dict[str, Any]) -> None:
            signature = json.dumps(candidate, sort_keys=True, default=str)
            if any(json.dumps(existing, sort_keys=True, default=str) == signature for _, existing in candidates):
                return
            candidates.append((label, candidate))

        original_extra_body = request_kwargs.get("extra_body")
        original_reasoning = {}
        if isinstance(original_extra_body, dict) and isinstance(original_extra_body.get("reasoning"), dict):
            original_reasoning = dict(original_extra_body["reasoning"])
        if original_reasoning:
            visible_reasoning_kwargs = dict(request_kwargs)
            visible_extra_body = dict(original_extra_body) if isinstance(original_extra_body, dict) else {}
            visible_reasoning = dict(original_reasoning)
            visible_reasoning["exclude"] = False
            visible_extra_body["reasoning"] = visible_reasoning
            visible_reasoning_kwargs["extra_body"] = visible_extra_body
            visible_reasoning_kwargs.pop("reasoning_effort", None)
            add_candidate("chat_with_reasoning_visible", visible_reasoning_kwargs)

        stripped_kwargs = dict(request_kwargs)
        extra_body = stripped_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            fallback_extra_body = dict(extra_body)
            fallback_extra_body.pop("reasoning", None)
            if fallback_extra_body:
                stripped_kwargs["extra_body"] = fallback_extra_body
            else:
                stripped_kwargs.pop("extra_body", None)
        stripped_kwargs.pop("reasoning_effort", None)
        if stripped_kwargs != request_kwargs:
            add_candidate("chat_without_reasoning_controls", stripped_kwargs)

        bare_kwargs = dict(stripped_kwargs)
        bare_kwargs.pop("extra_body", None)
        if bare_kwargs != stripped_kwargs:
            add_candidate("bare_chat_without_extra_body", bare_kwargs)

        if "max_completion_tokens" in stripped_kwargs:
            max_tokens_kwargs = dict(stripped_kwargs)
            max_tokens_kwargs["max_tokens"] = max_tokens_kwargs.pop("max_completion_tokens")
            add_candidate("chat_without_reasoning_controls_max_tokens", max_tokens_kwargs)

        if "max_completion_tokens" in bare_kwargs:
            bare_max_tokens_kwargs = dict(bare_kwargs)
            bare_max_tokens_kwargs["max_tokens"] = bare_max_tokens_kwargs.pop("max_completion_tokens")
            add_candidate("bare_chat_max_tokens", bare_max_tokens_kwargs)

        for label, candidate in candidates:
            attempt_info = {
                "label": label,
                "has_reasoning_extra_body": isinstance(candidate.get("extra_body"), dict)
                and "reasoning" in candidate.get("extra_body", {}),
                "has_reasoning_effort": "reasoning_effort" in candidate,
                "uses_max_completion_tokens": "max_completion_tokens" in candidate,
                "uses_max_tokens": "max_tokens" in candidate,
            }
            try:
                response, raw_response_text = self._create_chat_completion(candidate)
                probe = self._extract_generation(
                    response,
                    max_new_tokens=max_new_tokens,
                    raw_response_text=raw_response_text,
                    count_tokens=False,
                )
                attempts.append({**attempt_info, "ok": True, "generation_empty": probe.generation_empty})
                if not probe.generation_empty:
                    return response, raw_response_text, {
                        "chat_empty_recovery_attempted": True,
                        "chat_empty_recovery_succeeded": label,
                        "chat_empty_recovery_attempts": attempts,
                    }
            except Exception as exc:
                attempts.append({**attempt_info, "ok": False, "error": self._error_text(exc)})

        if attempts:
            return None, "", {
                "chat_empty_recovery_attempted": True,
                "chat_empty_recovery_succeeded": "",
                "chat_empty_recovery_attempts": attempts,
            }
        return None

    def _responses_generation_recovery(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        response_format: Dict[str, Any] | None = None,
    ) -> LLMGeneration | None:
        request_kwargs = self._build_responses_request_kwargs(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            response_format=response_format,
        )
        debug: Dict[str, Any] = {
            "responses_attempted": True,
            "responses_request_has_reasoning": "reasoning" in request_kwargs,
            "responses_request_has_text_format": bool(request_kwargs.get("text")),
        }

        try:
            response, raw_response_text = self._create_responses_completion(request_kwargs)
            generation = self._extract_responses_generation(response, max_new_tokens=max_new_tokens, raw_response_text=raw_response_text)
            generation = self._with_debug(
                generation,
                {
                    **debug,
                    "responses_path": "sdk",
                    "responses_sdk_raw_len": len(raw_response_text or ""),
                    "responses_sdk_finish_reason": generation.finish_reason,
                    "responses_sdk_empty": generation.generation_empty,
                },
            )
            if not generation.generation_empty:
                return generation
            debug.update(
                {
                    "responses_sdk_raw_len": len(raw_response_text or ""),
                    "responses_sdk_finish_reason": generation.finish_reason,
                    "responses_sdk_empty": True,
                }
            )
        except Exception as exc:
            debug["responses_sdk_error"] = self._error_text(exc)

        try:
            response, raw_response_text = self._create_responses_completion_http(request_kwargs)
            generation = self._extract_responses_generation(response, max_new_tokens=max_new_tokens, raw_response_text=raw_response_text)
            generation = self._with_debug(
                generation,
                {
                    **debug,
                    "responses_path": "http",
                    "responses_http_raw_len": len(raw_response_text or ""),
                    "responses_http_finish_reason": generation.finish_reason,
                    "responses_http_empty": generation.generation_empty,
                },
            )
            return generation
        except Exception as exc:
            debug["responses_http_error"] = self._error_text(exc)

        return LLMGeneration(
            text="",
            raw_text="",
            finish_reason="",
            usage={},
            error=None,
            debug=debug,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        responses_generation: LLMGeneration | None = None
        if self._is_reasoning_model(self.model_name) and self._should_try_responses_api():
            responses_generation = self._responses_generation_recovery(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            if responses_generation is not None and not responses_generation.generation_empty:
                self.last_generation = self._with_debug(
                    responses_generation,
                    {"api_path": "responses"},
                )
                return self.last_generation.text

        request_kwargs = self._build_request_kwargs(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        response, raw_response_text = self._create_chat_completion(request_kwargs)
        generation = self._extract_generation(response, max_new_tokens=max_new_tokens, raw_response_text=raw_response_text)
        recovery_debug: Dict[str, Any] = {}
        if generation.generation_empty and generation.finish_reason == "stop":
            recovered = self._empty_generation_recovery(request_kwargs, max_new_tokens=max_new_tokens)
            if recovered is not None:
                response, raw_response_text, recovery_debug = recovered
                if response is not None:
                    generation = self._extract_generation(response, max_new_tokens=max_new_tokens, raw_response_text=raw_response_text)
        debug = {
            "api_path": "chat_completions",
            "chat_request_has_reasoning_extra_body": isinstance(request_kwargs.get("extra_body"), dict)
            and "reasoning" in request_kwargs.get("extra_body", {}),
            "chat_request_has_reasoning_effort": "reasoning_effort" in request_kwargs,
            "chat_request_uses_max_completion_tokens": "max_completion_tokens" in request_kwargs,
            **recovery_debug,
        }
        if responses_generation is not None:
            debug.update(responses_generation.debug)
        self.last_generation = self._with_debug(generation, debug)
        return generation.text

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int,
        default: Dict[str, Any],
        temperature: float = 0.1,
        do_sample: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        # Try up to 3 times, stronger constraints each time
        prompts = [
            prompt + "\n\nIMPORTANT: Return ONLY one valid JSON object. No markdown. No prose.",
            prompt + "\n\nIMPORTANT: Your previous response may be invalid. Return ONLY one valid JSON object. Do not include code fences, explanations, or any extra text.",
            prompt + "\n\nFINAL ATTEMPT. Return EXACTLY one valid JSON object matching the schema. If unsure, fill missing values with defaults but keep valid JSON.",
        ]

        for i in range(min(max_retries, 3)):
            try:
                if self._is_reasoning_model(self.model_name) and self._should_try_responses_api():
                    response_generation = self._responses_generation_recovery(
                        prompt=prompts[i],
                        max_new_tokens=max_new_tokens,
                        temperature=(0.0 if i > 0 else temperature),
                        do_sample=(do_sample if i == 0 else False),
                        response_format={"type": "json_object"},
                    )
                    if response_generation is not None:
                        self.last_generation = self._with_debug(response_generation, {"api_path": "responses"})
                        parsed = extract_json_block(response_generation.text)
                        if isinstance(parsed, dict):
                            return parsed
                        if response_generation.text.strip():
                            maybe = json.loads(response_generation.text.strip())
                            if isinstance(maybe, dict):
                                return maybe

                req = self._build_request_kwargs(
                    prompt=prompts[i],
                    max_new_tokens=max_new_tokens,
                    temperature=(0.0 if i > 0 else temperature),
                    do_sample=(do_sample if i == 0 else False),
                    response_format={"type": "json_object"},
                )
                response, raw_response_text = self._create_chat_completion(req)
                generation = self._extract_generation(response, max_new_tokens=max_new_tokens, raw_response_text=raw_response_text)
                self.last_generation = self._with_debug(generation, {"api_path": "chat_completions"})

                parsed = extract_json_block(generation.text)
                if isinstance(parsed, dict):
                    return parsed
                maybe = json.loads(generation.text.strip())
                if isinstance(maybe, dict):
                    return maybe
            except Exception:
                continue

        return default


class LocalLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        prefer_bfloat16: bool = True,
        device_preference: str = "auto",
        min_free_gb_for_cuda: float = 8.0,
    ) -> None:
        super().__init__()
        self.model_path = model_path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        pref = (device_preference or "auto").lower().strip()
        if pref not in {"auto", "cuda", "cpu"}:
            pref = "auto"

        dtype = torch.float32
        self.device = "cpu"
        self.device_reason = "CPU fallback"

        if pref != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"[TDCA] CUDA visible. free={free_gb:.2f} GiB / total={total_gb:.2f} GiB")

            if pref == "cuda" or free_gb >= float(min_free_gb_for_cuda):
                self.device = "cuda"
                self.device_reason = f"CUDA selected, free VRAM {free_gb:.2f} GiB"
                if prefer_bfloat16 and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            elif pref == "cuda":
                raise RuntimeError("--local_device cuda was requested, but CUDA is not available or lacks free memory.")

        print(f"[TDCA] LocalLLM device = {self.device} ({self.device_reason})")

        load_kwargs = dict(
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        self.call_count += 1
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        self.total_generated_tokens += int(generated.shape[0])
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        cleaned = strip_think_blocks(raw)
        self.last_generation = LLMGeneration(
            text=cleaned,
            raw_text=raw,
            finish_reason="stop",
            usage={"completion_tokens": int(generated.shape[0])},
        )
        return cleaned


class ValueEvaluator:
    def __init__(self, llm: BaseLLM, value_weights: Dict[str, float]) -> None:
        self.llm = llm
        self.value_weights = value_weights

    def evaluate(
        self,
        question: str,
        node: Node,
        evidence_items: List[RetrievedContext],
        memory_items: List[RetrievedContext],
        scoring_mode: str = "hybrid",
        max_new_tokens_score: int = 128,
    ) -> Tuple[float, Dict[str, float]]:
        if node.node_type == NodeType.KG:
            return 1.0, {
                "task_progress": 0.0,
                "evidence_support": 1.0,
                "memory_usefulness": 0.0,
                "answerability": 0.0,
                "uncertainty": 0.0,
            }

        if scoring_mode == "llm":
            metrics = self._score_with_llm(
                question=question,
                state_text=node.content,
                evidence_items=evidence_items,
                memory_items=memory_items,
                max_new_tokens_score=max_new_tokens_score,
            )
        else:
            metrics = self._score_hybrid(
                question=question,
                state_text=node.content,
                evidence_items=evidence_items,
                memory_items=memory_items,
            )

        value = (
            self.value_weights["task_progress"] * metrics["task_progress"]
            + self.value_weights["evidence_support"] * metrics["evidence_support"]
            + self.value_weights["memory_usefulness"] * metrics["memory_usefulness"]
            + self.value_weights["answerability"] * metrics["answerability"]
            - self.value_weights["uncertainty_penalty"] * metrics["uncertainty"]
        )
        return clamp(value, 0.0, 1.0), metrics

    def _score_with_llm(
        self,
        question: str,
        state_text: str,
        evidence_items: List[RetrievedContext],
        memory_items: List[RetrievedContext],
        max_new_tokens_score: int,
    ) -> Dict[str, float]:
        default = {
            "task_progress": 0.5,
            "evidence_support": 0.5,
            "memory_usefulness": 0.5,
            "answerability": 0.3,
            "uncertainty": 0.5,
        }
        prompt = build_scoring_prompt(
            question=question,
            state_text=state_text,
            evidence_items=evidence_items,
            memory_items=memory_items,
        )
        result = self.llm.generate_json(
            prompt=prompt,
            max_new_tokens=max_new_tokens_score,
            default=default,
            temperature=0.1,
            do_sample=False,
            max_retries=3,
        )
        out = {}
        for key, default_value in default.items():
            out[key] = clamp(float(result.get(key, default_value)))
        return out

    def _score_hybrid(
        self,
        question: str,
        state_text: str,
        evidence_items: List[RetrievedContext],
        memory_items: List[RetrievedContext],
    ) -> Dict[str, float]:
        q_tokens = set(simple_tokenize(question))
        s_tokens = set(simple_tokenize(state_text))

        overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
        evidence_support = sum(item.score for item in evidence_items[:2]) / max(1, min(2, len(evidence_items))) if evidence_items else 0.0
        memory_usefulness = sum(item.score for item in memory_items[:2]) / max(1, min(2, len(memory_items))) if memory_items else 0.0

        answer_markers = ["therefore", "answer", "final", "thus", " is ", " are ", "who is", "what is", "where was"]
        answerability = 0.20 + 0.50 * evidence_support + 0.20 * overlap
        if any(marker in f" {state_text.lower()} " for marker in answer_markers):
            answerability += 0.10

        uncertainty_terms = ["maybe", "possibly", "probably", "unclear", "not sure", "guess", "might"]
        uncertainty = 0.12 + 0.18 * max(0.0, 1.0 - evidence_support)
        if any(term in state_text.lower() for term in uncertainty_terms):
            uncertainty += 0.25

        task_progress = 0.28 + 0.38 * overlap + 0.34 * evidence_support

        return {
            "task_progress": clamp(task_progress),
            "evidence_support": clamp(evidence_support),
            "memory_usefulness": clamp(memory_usefulness),
            "answerability": clamp(answerability),
            "uncertainty": clamp(uncertainty),
        }
