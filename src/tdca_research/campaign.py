from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "tdca-campaign-budget-v1"


class CampaignBudgetExceeded(RuntimeError):
    """A shared provider campaign cannot safely authorize another request."""

    def __init__(self, message: str, snapshot: dict[str, Any]) -> None:
        super().__init__(message)
        self.snapshot = snapshot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Cross-process advisory lock used around every read-modify-write cycle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - Linux is the experiment platform.
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CampaignBudgetLedger:
    """Durable provider-attempt and provider-token accounting.

    Calls are charged before an HTTP request starts. Tokens are conservatively
    reserved before the request and replaced by provider-reported actual usage
    after it returns. A killed process therefore cannot silently lose a call or
    authorize work beyond the declared cap.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str,
        provider_call_cap: int,
        provider_token_cap: int,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.campaign_id = str(campaign_id)
        self.provider_call_cap = int(provider_call_cap)
        self.provider_token_cap = int(provider_token_cap)
        if not self.campaign_id:
            raise ValueError("campaign_id is required for a campaign ledger")
        if self.provider_call_cap <= 0 or self.provider_token_cap <= 0:
            raise ValueError("campaign provider caps must be positive")
        with _exclusive_lock(self.lock_path):
            if self.path.exists():
                payload = self._read()
                self._validate_identity(payload)
            else:
                _atomic_json(self.path, self._new_payload())
        self.reconcile_cached_responses()

    def _new_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "limits": {
                "provider_calls": self.provider_call_cap,
                "provider_reported_tokens": self.provider_token_cap,
            },
            "usage": {
                "provider_calls": 0,
                "provider_reported_tokens": 0,
                "pending_reserved_tokens": 0,
                "effective_provider_tokens": 0,
            },
            "pending": {},
            "events": [],
            "status": "active",
            "last_stop_reason": None,
        }

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _validate_identity(self, payload: dict[str, Any]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "limits": {
                "provider_calls": self.provider_call_cap,
                "provider_reported_tokens": self.provider_token_cap,
            },
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"campaign ledger {key} does not match configured campaign")

    @staticmethod
    def _refresh(payload: dict[str, Any]) -> None:
        pending_tokens = sum(
            int(row.get("reserved_tokens", 0))
            for row in payload.get("pending", {}).values()
        )
        usage = payload.setdefault("usage", {})
        usage["pending_reserved_tokens"] = pending_tokens
        usage["effective_provider_tokens"] = (
            int(usage.get("provider_reported_tokens", 0)) + pending_tokens
        )
        payload["updated_at_utc"] = _now()

    def snapshot(self) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            payload = self._read()
            self._validate_identity(payload)
            self._refresh(payload)
            _atomic_json(self.path, payload)
            return payload

    def reserve(
        self,
        *,
        cache_key: str,
        cache_path: str | Path,
        reserved_tokens: int,
    ) -> str:
        reserved_tokens = max(1, int(reserved_tokens))
        denied: CampaignBudgetExceeded | None = None
        request_id = uuid.uuid4().hex
        with _exclusive_lock(self.lock_path):
            payload = self._read()
            self._validate_identity(payload)
            self._refresh(payload)
            usage = payload["usage"]
            next_calls = int(usage.get("provider_calls", 0)) + 1
            next_tokens = int(usage.get("effective_provider_tokens", 0)) + reserved_tokens
            reasons = []
            if next_calls > self.provider_call_cap:
                reasons.append("provider_call_cap")
            if next_tokens > self.provider_token_cap:
                reasons.append("provider_token_cap_preflight")
            if reasons:
                payload["status"] = "exhausted"
                payload["last_stop_reason"] = "+".join(reasons)
                payload["events"].append({
                    "event": "request_denied", "at_utc": _now(),
                    "reason": payload["last_stop_reason"],
                    "requested_reserved_tokens": reserved_tokens,
                })
                self._refresh(payload)
                _atomic_json(self.path, payload)
                denied = CampaignBudgetExceeded(
                    f"campaign budget denied provider request: {payload['last_stop_reason']}",
                    payload,
                )
            else:
                usage["provider_calls"] = next_calls
                payload["pending"][request_id] = {
                    "cache_key": str(cache_key),
                    "cache_path": str(Path(cache_path)),
                    "reserved_tokens": reserved_tokens,
                    "started_at_utc": _now(),
                }
                payload["events"].append({
                    "event": "request_reserved", "at_utc": _now(),
                    "request_id": request_id, "cache_key": str(cache_key),
                    "reserved_tokens": reserved_tokens,
                })
                self._refresh(payload)
                _atomic_json(self.path, payload)
        if denied is not None:
            raise denied
        return request_id

    def settle(
        self,
        request_id: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        outcome: str,
    ) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            payload = self._read()
            self._validate_identity(payload)
            pending = payload.get("pending", {}).pop(str(request_id), None)
            if pending is None:
                raise ValueError(f"unknown or already settled campaign request: {request_id}")
            actual_tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
            usage = payload["usage"]
            usage["provider_reported_tokens"] = (
                int(usage.get("provider_reported_tokens", 0)) + actual_tokens
            )
            payload["events"].append({
                "event": "request_settled", "at_utc": _now(),
                "request_id": str(request_id), "cache_key": pending.get("cache_key", ""),
                "outcome": str(outcome), "provider_reported_tokens": actual_tokens,
            })
            self._refresh(payload)
            over_cap = int(usage["effective_provider_tokens"]) > self.provider_token_cap
            if over_cap:
                payload["status"] = "exhausted"
                payload["last_stop_reason"] = "provider_token_cap_post_response"
            _atomic_json(self.path, payload)
            return {"over_cap": over_cap, "snapshot": payload}

    def reconcile_cached_responses(self) -> int:
        """Settle killed-after-response reservations when their cache is durable."""
        reconciled = 0
        with _exclusive_lock(self.lock_path):
            payload = self._read()
            self._validate_identity(payload)
            for request_id, pending in list(payload.get("pending", {}).items()):
                cache_path = Path(str(pending.get("cache_path", "")))
                if not cache_path.exists():
                    continue
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    prompt_tokens = int(cached.get("prompt_tokens", 0))
                    completion_tokens = int(cached.get("completion_tokens", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                payload["pending"].pop(request_id, None)
                actual = max(0, prompt_tokens) + max(0, completion_tokens)
                payload["usage"]["provider_reported_tokens"] = (
                    int(payload["usage"].get("provider_reported_tokens", 0)) + actual
                )
                payload["events"].append({
                    "event": "request_reconciled_from_cache", "at_utc": _now(),
                    "request_id": request_id, "cache_key": pending.get("cache_key", ""),
                    "provider_reported_tokens": actual,
                })
                reconciled += 1
            self._refresh(payload)
            if int(payload["usage"]["effective_provider_tokens"]) > self.provider_token_cap:
                payload["status"] = "exhausted"
                payload["last_stop_reason"] = "provider_token_cap_reconciled"
            _atomic_json(self.path, payload)
        return reconciled
