from config.database import database_uri_from_environment


def test_explicit_aura_database_uri_has_priority(monkeypatch):
    monkeypatch.setenv("AURA_DATABASE_URI", "mongodb://primary.example/aura")
    monkeypatch.setenv("MONGO_URL", "mongodb://legacy.example/aura")
    monkeypatch.setenv("MONGODB_URI", "mongodb://fallback.example/aura")
    assert database_uri_from_environment() == "mongodb://primary.example/aura"


def test_legacy_mongo_url_remains_supported(monkeypatch):
    monkeypatch.delenv("AURA_DATABASE_URI", raising=False)
    monkeypatch.setenv("MONGO_URL", "mongodb://legacy.example/aura")
    monkeypatch.setenv("MONGODB_URI", "mongodb://fallback.example/aura")
    assert database_uri_from_environment() == "mongodb://legacy.example/aura"


def test_mongodb_uri_is_supported_as_fallback(monkeypatch):
    monkeypatch.delenv("AURA_DATABASE_URI", raising=False)
    monkeypatch.delenv("MONGO_URL", raising=False)
    monkeypatch.setenv("MONGODB_URI", "mongodb://fallback.example/aura")
    assert database_uri_from_environment() == "mongodb://fallback.example/aura"


def test_missing_database_uri_returns_none(monkeypatch):
    monkeypatch.delenv("AURA_DATABASE_URI", raising=False)
    monkeypatch.delenv("MONGO_URL", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert database_uri_from_environment() is None


def test_legacy_settings_import_without_database_uri(monkeypatch):
    import importlib

    monkeypatch.delenv("MONGO_URL", raising=False)
    import config.settings as settings

    reloaded = importlib.reload(settings)
    assert reloaded.MONGO_URL is None
