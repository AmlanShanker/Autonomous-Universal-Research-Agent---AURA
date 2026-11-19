"""API authentication, role authorization, and process-local rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass


_ALLOWED_ROLES = {"reader", "operator", "admin"}
_ROLE_LEVEL = {"reader": 1, "operator": 2, "admin": 3}


@dataclass(frozen=True)
class APIPrincipal:
    key_id: str
    role: str


class APIKeyAuthenticator:
    """Authenticate bearer API keys without retaining or logging raw key values."""

    def __init__(self, *, enabled: bool, keys: list[dict[str, str]] | None = None):
        self.enabled = enabled
        self._keys: list[tuple[str, str, str]] = []
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for entry in keys or []:
            raw_id = entry.get("id", "")
            raw_key = entry.get("key", "")
            raw_role = entry.get("role", "")
            if not isinstance(raw_id, str) or not isinstance(raw_role, str):
                raise ValueError("Each API key id and role must be strings.")
            key_id = raw_id.strip()
            role = raw_role.strip().lower()
            if not key_id or len(key_id) > 80:
                raise ValueError("Each API key requires an id of 1 to 80 characters.")
            if not isinstance(raw_key, str) or len(raw_key) < 32:
                raise ValueError("API keys must contain at least 32 characters.")
            if role not in _ALLOWED_ROLES:
                raise ValueError("API key role must be reader, operator, or admin.")
            digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            if key_id in seen_ids or digest in seen_digests:
                raise ValueError("API key ids and key values must be unique.")
            seen_ids.add(key_id)
            seen_digests.add(digest)
            self._keys.append((key_id, digest, role))
        if self.enabled and not self._keys:
            raise RuntimeError(
                "AURA_API_AUTH_ENABLED=true requires at least one entry in AURA_API_KEYS."
            )

    @classmethod
    def from_environment(cls) -> "APIKeyAuthenticator":
        enabled = os.getenv("AURA_API_AUTH_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        raw_config = os.getenv("AURA_API_KEYS", "[]")
        try:
            keys = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AURA_API_KEYS must be a valid JSON array.") from exc
        if not isinstance(keys, list) or any(not isinstance(item, dict) for item in keys):
            raise RuntimeError("AURA_API_KEYS must be a JSON array of key objects.")
        try:
            return cls(enabled=enabled, keys=keys)
        except ValueError as exc:
            raise RuntimeError(f"Invalid AURA_API_KEYS configuration: {exc}") from exc

    def authenticate(self, authorization: str | None) -> APIPrincipal | None:
        if not self.enabled:
            return APIPrincipal(key_id="development", role="admin")
        if not authorization:
            return None
        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token.strip():
            return None
        candidate = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
        matched: APIPrincipal | None = None
        # Compare every configured digest to avoid stopping at the matching key.
        for key_id, digest, role in self._keys:
            if hmac.compare_digest(candidate, digest):
                matched = APIPrincipal(key_id=key_id, role=role)
        return matched


class InMemoryRateLimiter:
    """Thread-safe fixed-window limiter; counters are local to this API process."""

    def __init__(self, limit: int = 120, window_seconds: int = 60):
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("Rate limit must be a positive integer.")
        if isinstance(window_seconds, bool) or window_seconds < 1:
            raise ValueError("Rate limit window must be a positive integer.")
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "InMemoryRateLimiter":
        try:
            limit = int(os.getenv("AURA_API_RATE_LIMIT", "120"))
            window = int(os.getenv("AURA_API_RATE_WINDOW_SECONDS", "60"))
            return cls(limit=limit, window_seconds=window)
        except ValueError as exc:
            raise RuntimeError("AURA_API_RATE_LIMIT and AURA_API_RATE_WINDOW_SECONDS must be positive integers.") from exc

    def check(self, identity: str, now: float | None = None) -> tuple[bool, int, int]:
        current = time.monotonic() if now is None else now
        with self._lock:
            window_start, count = self._windows.get(identity, (current, 0))
            if current - window_start >= self.window_seconds:
                window_start, count = current, 0
            if count >= self.limit:
                retry_after = max(1, int(self.window_seconds - (current - window_start) + 0.999))
                return False, 0, retry_after
            count += 1
            self._windows[identity] = (window_start, count)
            if len(self._windows) > 10000:
                cutoff = current - self.window_seconds
                self._windows = {
                    key: value for key, value in self._windows.items()
                    if value[0] >= cutoff
                }
            return True, self.limit - count, max(
                0, int(self.window_seconds - (current - window_start))
            )


def client_identity(client_host: str | None) -> str:
    """Return a bounded fallback identity; do not trust forwarded IP headers."""
    return f"ip:{client_host or 'unknown'}"
