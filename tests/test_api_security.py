import json

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.security import APIKeyAuthenticator, InMemoryRateLimiter
from storage.database.memory import InMemoryDatabase


READER_KEY = "reader-key-0123456789-abcdefghijkl"
OPERATOR_KEY = "operator-key-0123456789-abcdefghi"


def configure_auth(monkeypatch, *, limit="120"):
    keys = [
        {"id": "reader-client", "key": READER_KEY, "role": "reader"},
        {"id": "operator-client", "key": OPERATOR_KEY, "role": "operator"},
    ]
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("AURA_API_KEYS", json.dumps(keys))
    monkeypatch.setenv("AURA_API_RATE_LIMIT", limit)
    monkeypatch.setenv("AURA_API_RATE_WINDOW_SECONDS", "60")


def make_client():
    app = create_app(
        database=InMemoryDatabase(),
        director=object(),
        experiment_adapters={},
    )
    return TestClient(app)


def test_authentication_is_required_except_for_health(monkeypatch):
    configure_auth(monkeypatch)
    client = make_client()

    assert client.get("/health").status_code == 200
    assert client.get("/research/runs/missing").status_code == 401
    assert client.get(
        "/research/runs/missing",
        headers={"Authorization": "Bearer invalid-key"},
    ).status_code == 401


def test_reader_can_read_but_cannot_mutate(monkeypatch):
    configure_auth(monkeypatch)
    client = make_client()
    headers = {"Authorization": f"Bearer {READER_KEY}"}

    assert client.get("/research/runs/missing", headers=headers).status_code == 404
    response = client.post("/research/runs/missing/execute", headers=headers)
    assert response.status_code == 403


def test_operator_is_authorized_for_mutations(monkeypatch):
    configure_auth(monkeypatch)
    client = make_client()
    headers = {"Authorization": f"Bearer {OPERATOR_KEY}"}

    # The request reaches the route handler; the run itself is intentionally absent.
    assert client.post("/research/runs/missing/execute", headers=headers).status_code == 404


def test_authentication_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("AURA_API_KEYS", "[]")

    with pytest.raises(RuntimeError, match="requires at least one"):
        make_client()


def test_rate_limit_returns_429_and_retry_after(monkeypatch):
    configure_auth(monkeypatch, limit="2")
    client = make_client()

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    response = client.get("/health")

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limiter_resets_after_window():
    limiter = InMemoryRateLimiter(limit=1, window_seconds=10)

    assert limiter.check("client", now=100) == (True, 0, 10)
    assert limiter.check("client", now=101)[0] is False
    assert limiter.check("client", now=110) == (True, 0, 10)


def test_api_keys_require_unique_ids_and_strong_values():
    with pytest.raises(ValueError, match="at least 32"):
        APIKeyAuthenticator(
            enabled=True,
            keys=[{"id": "weak", "key": "short", "role": "admin"}],
        )
    with pytest.raises(ValueError, match="unique"):
        APIKeyAuthenticator(
            enabled=True,
            keys=[
                {"id": "same", "key": "a" * 32, "role": "reader"},
                {"id": "same", "key": "b" * 32, "role": "operator"},
            ],
        )
