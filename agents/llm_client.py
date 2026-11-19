import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    Provider-independent interface for communicating with an LLM.

    Supported providers:
    - Groq (default)
    - Adaption Labs

    Select a provider with LLM_PROVIDER. Existing deployments continue to use
    Groq when LLM_PROVIDER is not set.
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
        if self.provider not in self.PROVIDERS:
            supported = ", ".join(sorted(self.PROVIDERS))
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{self.provider}'. "
                f"Supported providers: {supported}."
            )

        config = self.PROVIDERS[self.provider]
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"{config['api_key_env']} is not configured."
            )

        self.model = (
            os.getenv(config["model_env"]) or config["default_model"]
        )
        if not self.model or not self.model.strip():
            raise RuntimeError(
                f"{config['model_env']} is not configured "
                f"for provider '{self.provider}'."
            )
        self.model = self.model.strip()

        self.client = OpenAI(
            api_key=api_key,
            base_url=config["base_url"],
        )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured provider and return its response.
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
