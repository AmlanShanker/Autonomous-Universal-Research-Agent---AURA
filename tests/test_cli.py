import sys

import pytest

from aura import main


def test_cli_requires_a_service_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aura"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_cli_starts_api_with_configured_options(monkeypatch):
    import uvicorn

    calls = {}
    monkeypatch.setattr(sys, "argv", ["aura", "api", "--host", "0.0.0.0", "--port", "8123", "--reload"])
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))

    main()

    assert calls == {
        "app": "api.app:app",
        "host": "0.0.0.0",
        "port": 8123,
        "reload": True,
    }


def test_cli_starts_worker(monkeypatch):
    from workers import research_worker

    calls = []
    monkeypatch.setattr(sys, "argv", ["aura", "worker"])
    monkeypatch.setattr(research_worker, "main", lambda: calls.append("started"))

    main()

    assert calls == ["started"]
