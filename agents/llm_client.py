import os
import time

from dotenv import load_dotenv
from observability import METRICS, configure_logging, span
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


class LLMFallbackError(RuntimeError):
    """Raised when both configured providers fail; retains both exceptions."""

    def __init__(self, primary_provider, fallback_provider, primary_error, fallback_error):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"LLM request failed on primary provider '{primary_provider}' "
            f"({type(primary_error).__name__}) and fallback provider "
            f"'{fallback_provider}' ({type(fallback_error).__name__})."
        )


class LLMClient:
    """Provider-independent LLM client with bounded retries and one fallback."""

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

    FALLBACK_ELIGIBLE_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )

    def __init__(self):
        configure_logging()
        load_dotenv()
        self.provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        primary_config = self._provider_config(self.provider, "LLM_PROVIDER")
        self.model = self._configured_model(primary_config, self.provider)
        api_key = self._configured_api_key(primary_config, self.provider)

        self.max_retries = self._integer_setting("LLM_RETRY_ATTEMPTS", 2, minimum=0)
        self.timeout_seconds = self._float_setting("LLM_TIMEOUT_SECONDS", 30.0, minimum=0.1)
        self.retry_backoff_seconds = self._float_setting(
            "LLM_RETRY_BACKOFF_SECONDS", 0.5, minimum=0.0
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=primary_config["base_url"],
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        self.fallback_provider = None
        self.fallback_model = None
        self.fallback_client = None
        self._fallback_base_url = None
        self._load_fallback_configuration()

    @classmethod
    def _provider_config(cls, provider: str, setting_name: str) -> dict:
        if provider not in cls.PROVIDERS:
            supported = ", ".join(sorted(cls.PROVIDERS))
            raise ValueError(
                f"Unsupported {setting_name} '{provider}'. Supported providers: {supported}."
            )
        return cls.PROVIDERS[provider]

    @staticmethod
    def _configured_api_key(config: dict, provider: str) -> str:
        api_key = os.getenv(config["api_key_env"])
        if not api_key or not api_key.strip():
            raise RuntimeError(
                f"{config['api_key_env']} is not configured for provider '{provider}'."
            )
        return api_key.strip()

    @staticmethod
    def _configured_model(config: dict, provider: str) -> str:
        model = os.getenv(config["model_env"]) or config["default_model"]
        if not model or not model.strip():
            raise RuntimeError(
                f"{config['model_env']} is not configured for provider '{provider}'."
            )
        return model.strip()

    @staticmethod
    def _integer_setting(name: str, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        try:
            value = default if raw is None or not raw.strip() else int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return value

    @staticmethod
    def _float_setting(name: str, default: float, minimum: float) -> float:
        raw = os.getenv(name)
        try:
            value = default if raw is None or not raw.strip() else float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return value

    def _load_fallback_configuration(self) -> None:
        fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
        if not fallback_provider:
            return
        fallback_config = self._provider_config(
            fallback_provider, "LLM_FALLBACK_PROVIDER"
        )
        if fallback_provider == self.provider:
            raise ValueError(
                "LLM_FALLBACK_PROVIDER must be different from LLM_PROVIDER."
            )
        fallback_key = self._configured_api_key(fallback_config, fallback_provider)
        self.fallback_model = self._configured_model(fallback_config, fallback_provider)
        self.fallback_provider = fallback_provider
        self._fallback_base_url = fallback_config["base_url"]
        self.fallback_client = OpenAI(
            api_key=fallback_key,
            base_url=fallback_config["base_url"],
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _response_content(response, provider: str) -> str:
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"The {provider} provider returned an empty response.")
        return content

    @staticmethod
    def _messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _request_with_retries(self, client, model: str, prompt: str):
        """Retry only transient errors; SDK retries are disabled on our clients."""
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=self._messages(prompt),
                )
                return response
            except self.FALLBACK_ELIGIBLE_ERRORS:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff_seconds * (2 ** attempt)
                if delay:
                    time.sleep(delay)

    def generate(self, prompt: str) -> str:
        """Generate with bounded retries while recording provider-level telemetry."""
        METRICS.increment("aura_llm_requests", provider=self.provider, outcome="started")
        try:
            with span("llm.generate", component="llm", provider=self.provider):
                result = self._generate(prompt)
        except Exception:
            METRICS.increment("aura_llm_requests", provider=self.provider, outcome="error")
            raise
        METRICS.increment("aura_llm_requests", provider=self.provider, outcome="success")
        return result

    def _generate(self, prompt: str) -> str:
        """Generate with bounded retries and at most one fallback provider attempt."""
        try:
            response = self._request_with_retries(self.client, self.model, prompt)
        except self.FALLBACK_ELIGIBLE_ERRORS as primary_error:
            if self.fallback_client is None:
                raise
            try:
                response = self._request_with_retries(
                    self.fallback_client, self.fallback_model, prompt
                )
            except self.FALLBACK_ELIGIBLE_ERRORS as fallback_error:
                raise LLMFallbackError(
                    self.provider,
                    self.fallback_provider,
                    primary_error,
                    fallback_error,
                ) from fallback_error
            return self._response_content(response, self.fallback_provider)
        return self._response_content(response, self.provider)
