from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.llm_client import LLMClient


def isolate_environment(monkeypatch):
    # Do not let a developer's local .env file leak into configuration tests.
    monkeypatch.setattr("agents.llm_client.load_dotenv", lambda: None)


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
    openai_client.assert_called_once_with(
        api_key="test-groq-key",
        base_url="https://api.groq.com/openai/v1",
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
