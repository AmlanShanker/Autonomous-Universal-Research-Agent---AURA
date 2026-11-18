from typing import Protocol


class EmbeddingProvider(Protocol):
    """Generate vector embeddings without coupling AURA to one vendor."""

    def embed(self, text: str) -> list[float]:
        """Return a finite, non-empty embedding for the supplied text."""
        ...


def build_memory_embedding_text(
    research_question: str,
    content: str,
    tags: list[str] | None = None,
) -> str:
    """Build the consistent text representation embedded for a memory."""
    return "\n".join(
        part
        for part in (
            research_question.strip(),
            content.strip(),
            " ".join(tags or []).strip(),
        )
        if part
    )


def validate_embedding(embedding: list[float]) -> list[float]:
    """Validate and normalize an embedding returned by a provider."""
    import math

    if not isinstance(embedding, (list, tuple)) or not embedding:
        raise ValueError("Embedding provider must return a non-empty vector.")

    values = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Embedding values must be finite numbers.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Embedding values must be finite numbers.")
        values.append(number)

    return values
