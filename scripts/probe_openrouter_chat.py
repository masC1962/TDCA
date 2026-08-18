from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


def _dump_obj(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _message_field(message: Any, name: str) -> Any:
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _response_payload(response: Any) -> Dict[str, Any]:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice is not None else None
    return {
        "finish_reason": getattr(choice, "finish_reason", None),
        "content": _message_field(message, "content"),
        "reasoning": _message_field(message, "reasoning"),
        "refusal": _message_field(message, "refusal"),
        "usage": _dump_obj(getattr(response, "usage", None)),
    }


def _print_result(name: str, payload: Dict[str, Any]) -> None:
    print(f"\n### {name}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://yh.m7ai.com/v1"
    )
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("DASHSCOPE_MODEL")
        or os.getenv("SERVED_MODEL_NAME")
        or os.getenv("OPENROUTER_MODEL")
        or "gpt-5.4"
    )
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY or OPENAI_API_KEY")

    print(f"base_url={base_url}")
    print(f"model={model}")
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "user", "content": "Return exactly this single line: Final Answer: yes"}]
    variants = [
        ("max_completion_tokens", {"max_completion_tokens": 64}),
        ("max_tokens", {"max_tokens": 64}),
        (
            "reasoning_none_exclude_true",
            {
                "max_completion_tokens": 64,
                "extra_body": {"reasoning": {"effort": "none", "exclude": True}},
            },
        ),
        (
            "reasoning_none_exclude_false",
            {
                "max_completion_tokens": 64,
                "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
            },
        ),
        (
            "reasoning_minimal_exclude_true",
            {
                "max_completion_tokens": 64,
                "extra_body": {"reasoning": {"effort": "minimal", "exclude": True}},
            },
        ),
        ("temperature_zero_max_completion", {"max_completion_tokens": 64, "temperature": 0}),
        ("temperature_zero_max_tokens", {"max_tokens": 64, "temperature": 0}),
    ]

    results = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "model": model,
        "variants": [],
    }
    for name, kwargs in variants:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            payload = _response_payload(response)
            results["variants"].append({"name": name, "request_kwargs": kwargs, "response": payload})
            _print_result(name, payload)
        except Exception as exc:
            payload = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            results["variants"].append({"name": name, "request_kwargs": kwargs, "response": payload})
            print(f"\n### {name}")
            print(f"ERROR {payload['error_type']}: {payload['error']}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"probe_openrouter_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    nonempty = [
        item["name"]
        for item in results["variants"]
        if (item.get("response", {}).get("content") or item.get("response", {}).get("reasoning") or "").strip()
    ]
    print("\n### summary")
    print(f"saved={output_path}")
    if nonempty:
        print("nonempty_variants=" + ", ".join(nonempty))
    else:
        print("nonempty_variants=NONE")
        print("diagnosis=all tested chat/completions variants returned empty visible text")


if __name__ == "__main__":
    main()
