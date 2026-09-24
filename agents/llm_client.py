import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    Provider-independent interface for communicating with an LLM.

    Groq is used as the current provider.
    """

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured LLM and return its response.
        """

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content