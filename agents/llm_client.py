import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    Provider-independent interface for communicating with an LLM.

    Supported providers:
    - Groq (default)
    - Adaption Labs

    Select the primary provider with LLM_PROVIDER. An optional, distinct
    provider can be configured with LLM_FALLBACK_PROVIDER. Fallback execution
    is handled separately; this class validates and stores its configuration.
    """

    PROVIDERS = {
        "groq": {
            "api_key_env": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
            "model_env": "GROQ_MODEL",
            "default_model": "openai/gpt-oss-120b",
        },
        "adaption": {
            "api_key_env": "ADAPTION_API_KEY",
            "base_url": "https://api.prod.adaptionlabs.ai/api/v1",
            "model_env": "ADAPTION_MODEL",
            "default_model": None,
        },
    }

    def __init__(self):
        load_dotenv()

        self.provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        primary_config = self._provider_config(self.provider, "LLM_PROVIDER")
        self.model = self._configured_model(primary_config, self.provider)
        api_key = self._configured_api_key(primary_config, self.provider)

        self.client = OpenAI(
            api_key=api_key,
            base_url=primary_config["base_url"],
        )

        self.fallback_provider = None
        self.fallback_model = None
        self._fallback_api_key = None
        self._fallback_base_url = None
        self._load_fallback_configuration()

    @classmethod
    def _provider_config(cls, provider: str, setting_name: str) -> dict:
        if provider not in cls.PROVIDERS:
            supported = ", ".join(sorted(cls.PROVIDERS))
            raise ValueError(
                f"Unsupported {setting_name} '{provider}'. "
                f"Supported providers: {supported}."
            )
        return cls.PROVIDERS[provider]

    @staticmethod
    def _configured_api_key(config: dict, provider: str) -> str:
        api_key = os.getenv(config["api_key_env"])
        if not api_key or not api_key.strip():
            raise RuntimeError(
                f"{config['api_key_env']} is not configured "
                f"for provider '{provider}'."
            )
        return api_key.strip()

    @staticmethod
    def _configured_model(config: dict, provider: str) -> str:
        model = os.getenv(config["model_env"]) or config["default_model"]
        if not model or not model.strip():
            raise RuntimeError(
                f"{config['model_env']} is not configured "
                f"for provider '{provider}'."
            )
        return model.strip()

    def _load_fallback_configuration(self) -> None:
        fallback_provider = (
            os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
        )
        if not fallback_provider:
            return

        fallback_config = self._provider_config(
            fallback_provider,
            "LLM_FALLBACK_PROVIDER",
        )
        if fallback_provider == self.provider:
            raise ValueError(
                "LLM_FALLBACK_PROVIDER must be different from LLM_PROVIDER."
            )

        self._fallback_api_key = self._configured_api_key(
            fallback_config,
            fallback_provider,
        )
        self.fallback_model = self._configured_model(
            fallback_config,
            fallback_provider,
        )
        self.fallback_provider = fallback_provider
        self._fallback_base_url = fallback_config["base_url"]

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured primary provider and return its response.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"The {self.provider} provider returned an empty response."
            )

        return content
