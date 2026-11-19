from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openai import APIConnectionError, BadRequestError

from agents.llm_client import LLMClient, LLMFallbackError


def isolate_environment(monkeypatch):
    # Do not let a developer's local .env file leak into configuration tests.
    monkeypatch.setattr("agents.llm_client.load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    # Legacy fallback tests verify a single primary attempt; retry behavior is
    # exercised explicitly by tests that set LLM_RETRY_ATTEMPTS.
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "0")


def test_defaults_to_groq_for_backward_compatibility(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    openai_client = MagicMock()
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_client)

    client = LLMClient()

    assert client.provider == "groq"
    assert client.model == "openai/gpt-oss-120b"
    assert client.fallback_provider is None
    assert client.fallback_model is None
    openai_client.assert_called_once_with(
        api_key="test-groq-key",
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
        max_retries=0,
    )


def test_groq_model_can_be_overridden(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "custom-groq-model")

    monkeypatch.setattr("agents.llm_client.OpenAI", MagicMock())

    client = LLMClient()

    assert client.model == "custom-groq-model"


def test_configures_adaption_provider(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    openai_client = MagicMock()
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_client)

    client = LLMClient()

    assert client.provider == "adaption"
    assert client.model == "qwen3.5-397b"
    openai_client.assert_called_once_with(
        api_key="test-adaption-key",
        base_url="https://api.prod.adaptionlabs.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )


def test_generate_uses_selected_provider_model_and_prompt(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    openai_client = MagicMock()
    openai_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="AURA test response")
                )
            ]
        )
    )
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_client)

    client = LLMClient()
    result = client.generate("Explain autonomous research.")

    openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="qwen3.5-397b",
        messages=[
            {
                "role": "user",
                "content": "Explain autonomous research.",
            }
        ],
    )
    assert result == "AURA test response"


def test_missing_provider_api_key_raises_clear_error(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    with pytest.raises(RuntimeError, match="ADAPTION_API_KEY"):
        LLMClient()


def test_missing_adaption_model_raises_clear_error(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.delenv("ADAPTION_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="ADAPTION_MODEL"):
        LLMClient()


def test_unsupported_provider_raises_clear_error(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        LLMClient()


def test_empty_model_setting_raises_clear_error(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "   ")

    with pytest.raises(RuntimeError, match="ADAPTION_MODEL"):
        LLMClient()


def test_empty_provider_response_raises_clear_error(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    openai_client = MagicMock()
    openai_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
    )
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_client)

    client = LLMClient()

    with pytest.raises(RuntimeError, match="empty response"):
        client.generate("Test prompt")


def test_configures_groq_fallback_for_adaption_primary(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    openai_client = MagicMock()
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_client)

    client = LLMClient()

    assert client.provider == "adaption"
    assert client.model == "qwen3.5-397b"
    assert client.fallback_provider == "groq"
    assert client.fallback_model == "openai/gpt-oss-120b"
    assert client._fallback_base_url == "https://api.groq.com/openai/v1"
    assert client.fallback_client is not None
    assert openai_client.call_count == 2
    openai_client.assert_any_call(
        api_key="test-adaption-key",
        base_url="https://api.prod.adaptionlabs.ai/api/v1",
        timeout=30.0,
        max_retries=0,
    )
    openai_client.assert_any_call(
        api_key="test-groq-key",
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
        max_retries=0,
    )


def test_configures_adaption_fallback_for_groq_primary(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    monkeypatch.setattr("agents.llm_client.OpenAI", MagicMock())

    client = LLMClient()

    assert client.fallback_provider == "adaption"
    assert client.fallback_model == "qwen3.5-397b"
    assert client._fallback_base_url == "https://api.prod.adaptionlabs.ai/api/v1"
    assert client.fallback_client is not None


def test_rejects_same_primary_and_fallback_provider(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")

    with pytest.raises(ValueError, match="must be different"):
        LLMClient()


def test_rejects_unsupported_fallback_provider(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_FALLBACK_PROVIDER"):
        LLMClient()


def test_fallback_requires_its_own_api_key(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        LLMClient()


def test_fallback_requires_model_when_provider_has_no_default(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.delenv("ADAPTION_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="ADAPTION_MODEL"):
        LLMClient()


def test_generate_falls_back_after_primary_connection_failure(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    primary = MagicMock()
    primary.chat.completions.create.side_effect = APIConnectionError(
        message="primary unavailable", request=MagicMock()
    )
    fallback = MagicMock()
    fallback.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fallback response"))]
    )
    openai_factory = MagicMock(side_effect=[primary, fallback])
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_factory)

    client = LLMClient()
    result = client.generate("research prompt")

    assert result == "fallback response"
    primary.chat.completions.create.assert_called_once_with(
        model="qwen3.5-397b",
        messages=[{"role": "user", "content": "research prompt"}],
    )
    fallback.chat.completions.create.assert_called_once_with(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "research prompt"}],
    )


def test_generate_does_not_fallback_for_bad_request(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    primary = MagicMock()
    primary.chat.completions.create.side_effect = BadRequestError(
        "invalid request",
        response=MagicMock(),
        body=None,
    )
    fallback = MagicMock()
    openai_factory = MagicMock(side_effect=[primary, fallback])
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_factory)

    client = LLMClient()
    with pytest.raises(BadRequestError):
        client.generate("invalid prompt")

    primary.chat.completions.create.assert_called_once()
    fallback.chat.completions.create.assert_not_called()


def test_generate_preserves_primary_error_when_fallback_is_disabled(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    primary = MagicMock()
    error = APIConnectionError(message="primary unavailable", request=MagicMock())
    primary.chat.completions.create.side_effect = error
    openai_factory = MagicMock(return_value=primary)
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_factory)

    client = LLMClient()
    with pytest.raises(APIConnectionError) as raised:
        client.generate("research prompt")

    assert raised.value is error
    primary.chat.completions.create.assert_called_once()


def test_fallback_failure_is_not_retried_again(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")

    primary = MagicMock()
    primary.chat.completions.create.side_effect = APIConnectionError(
        message="primary unavailable", request=MagicMock()
    )
    fallback = MagicMock()
    fallback_error = APIConnectionError(
        message="fallback unavailable", request=MagicMock()
    )
    fallback.chat.completions.create.side_effect = fallback_error
    openai_factory = MagicMock(side_effect=[primary, fallback])
    monkeypatch.setattr("agents.llm_client.OpenAI", openai_factory)

    client = LLMClient()
    with pytest.raises(LLMFallbackError) as raised:
        client.generate("research prompt")

    assert raised.value.primary_error is not None
    assert raised.value.fallback_error is fallback_error
    primary.chat.completions.create.assert_called_once()
    fallback.chat.completions.create.assert_called_once()
    assert openai_factory.call_count == 2



def test_retries_transient_primary_error_with_exponential_backoff(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0.25")
    sleeps = []
    monkeypatch.setattr("agents.llm_client.time.sleep", sleeps.append)

    provider = MagicMock()
    provider.chat.completions.create.side_effect = [
        APIConnectionError(message="temporary", request=MagicMock()),
        APIConnectionError(message="temporary", request=MagicMock()),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]),
    ]
    monkeypatch.setattr("agents.llm_client.OpenAI", MagicMock(return_value=provider))

    client = LLMClient()
    assert client.generate("prompt") == "ok"
    assert provider.chat.completions.create.call_count == 3
    assert sleeps == [0.25, 0.5]


def test_retry_attempts_zero_disables_retries(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "0")
    provider = MagicMock()
    provider.chat.completions.create.side_effect = APIConnectionError(
        message="unavailable", request=MagicMock()
    )
    monkeypatch.setattr("agents.llm_client.OpenAI", MagicMock(return_value=provider))

    client = LLMClient()
    with pytest.raises(APIConnectionError):
        client.generate("prompt")
    provider.chat.completions.create.assert_called_once()


def test_timeout_and_retry_settings_are_applied_to_both_clients(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_MODEL", "qwen3.5-397b")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "4")
    factory = MagicMock()
    monkeypatch.setattr("agents.llm_client.OpenAI", factory)

    client = LLMClient()

    assert client.timeout_seconds == 12.5
    assert client.max_retries == 4
    assert factory.call_count == 2
    for call in factory.call_args_list:
        assert call.kwargs["timeout"] == 12.5
        assert call.kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_RETRY_ATTEMPTS", "-1", "at least 0"),
        ("LLM_RETRY_ATTEMPTS", "many", "must be an integer"),
        ("LLM_TIMEOUT_SECONDS", "0", "at least 0.1"),
        ("LLM_TIMEOUT_SECONDS", "fast", "must be a number"),
        ("LLM_RETRY_BACKOFF_SECONDS", "-0.1", "at least 0.0"),
    ],
)
def test_rejects_invalid_reliability_settings(monkeypatch, name, value, message):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        LLMClient()


def test_non_transient_primary_error_is_not_retried(monkeypatch):
    isolate_environment(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "3")
    provider = MagicMock()
    error = BadRequestError("invalid", response=MagicMock(), body=None)
    provider.chat.completions.create.side_effect = error
    monkeypatch.setattr("agents.llm_client.OpenAI", MagicMock(return_value=provider))

    client = LLMClient()
    with pytest.raises(BadRequestError) as raised:
        client.generate("prompt")
    assert raised.value is error
    provider.chat.completions.create.assert_called_once()
