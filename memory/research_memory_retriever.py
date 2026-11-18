from datetime import datetime, timezone
import math
import re

from memory.embedding_provider import (
    EmbeddingProvider,
    build_memory_embedding_text,
    validate_embedding,
)
from memory.research_memory import ResearchMemoryItem


class ResearchMemoryRetriever:
    """
    Retrieve reusable findings using normalized keyword and optional semantic
    relevance, then adjust ranking using memory quality signals.

    Without an embedding provider, retrieval remains deterministic and
    keyword-only. With one, all active memories are considered so findings
    with different wording can still be discovered.
    """

    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "does",
        "for", "from", "how", "in", "into", "is", "it", "of",
        "on", "or", "that", "the", "this", "to", "what", "when",
        "where", "which", "with",
    }

    def __init__(
        self,
        database,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_weight: float = 0.7,
        quality_weight: float = 0.15,
    ):
        self._validate_weight("semantic_weight", semantic_weight)
        self._validate_weight("quality_weight", quality_weight)
        self.database = database
        self.embedding_provider = embedding_provider
        self.semantic_weight = float(semantic_weight)
        self.quality_weight = float(quality_weight)

    @staticmethod
    def _validate_weight(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be a number between 0 and 1.")

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: str | None = None,
    ) -> list[ResearchMemoryItem]:
        return [
            item for item, _ in self.retrieve_scored(
                query, limit=limit, exclude_run_id=exclude_run_id
            )
        ]

    def retrieve_scored(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: str | None = None,
    ) -> list[tuple[ResearchMemoryItem, float]]:
        """Return findings paired with normalized final scores in [0, 1]."""
        if not isinstance(query, str):
            raise TypeError("Research-memory query must be a string.")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError(
                "Research-memory result limit must be a positive integer."
            )

        query_terms = self._tokenize(query)
        if not query_terms and self.embedding_provider is None:
            return []
        if not query.strip():
            return []

        now = datetime.now(timezone.utc)
        if self.embedding_provider is None:
            candidates = self.database.search_research_memory(query)
            query_embedding = None
        else:
            candidates = self.database.list_research_memory()
            query_embedding = validate_embedding(
                self.embedding_provider.embed(query)
            )

        ranked = []
        seen_ids = set()
        for item in candidates:
            if item.memory_id in seen_ids:
                continue
            seen_ids.add(item.memory_id)

            if item.archived or (
                item.expires_at is not None and item.expires_at <= now
            ):
                continue
            if exclude_run_id is not None and item.run_id == exclude_run_id:
                continue

            lexical_score = self._normalized_lexical_score(item, query_terms)
            if query_embedding is None:
                relevance = lexical_score
            else:
                item_embedding = item.embedding
                if item_embedding is None:
                    text = build_memory_embedding_text(
                        item.research_question, item.content, item.tags
                    )
                    item_embedding = validate_embedding(
                        self.embedding_provider.embed(text)
                    )
                semantic_score = self._cosine_similarity(
                    query_embedding, item_embedding
                )
                relevance = (
                    self.semantic_weight * semantic_score
                    + (1.0 - self.semantic_weight) * lexical_score
                )

            if relevance <= 0.0:
                continue

            quality = self._quality_score(item, now)
            final_score = relevance * (
                (1.0 - self.quality_weight)
                + self.quality_weight * quality
            )
            ranked.append((item, final_score))

        ranked.sort(key=lambda entry: (-entry[1], entry[0].memory_id))
        return ranked[:limit]

    @classmethod
    def _normalized_lexical_score(
        cls,
        item: ResearchMemoryItem,
        query_terms: set[str],
    ) -> float:
        if not query_terms:
            return 0.0
        tag_terms = cls._tokenize(" ".join(item.tags))
        content_terms = cls._tokenize(item.content)
        question_terms = cls._tokenize(item.research_question)
        raw_score = (
            3 * len(query_terms.intersection(tag_terms))
            + 2 * len(query_terms.intersection(content_terms))
            + len(query_terms.intersection(question_terms))
        )
        return min(1.0, raw_score / (6.0 * len(query_terms)))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError(
                "Embedding dimensions do not match; use one embedding model "
                "for all research memories and queries."
            )
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = sum(a * b for a, b in zip(left, right)) / (
            left_norm * right_norm
        )
        return max(0.0, min(1.0, cosine))

    @staticmethod
    def _quality_score(item: ResearchMemoryItem, now: datetime) -> float:
        confidence = item.confidence
        provenance = (
            0.5 * bool(item.source_result_id)
            + 0.5 * bool(item.source_tool_id)
        )
        age_days = max(
            0,
            int((now - item.updated_at).total_seconds() // 86400),
        )
        freshness = math.pow(0.5, age_days / 365.0)
        return 0.70 * confidence + 0.15 * provenance + 0.15 * freshness

    @classmethod
    def _score(cls, item: ResearchMemoryItem, query_terms: set[str]) -> int:
        """Backward-compatible raw lexical score helper."""
        return round(
            cls._normalized_lexical_score(item, query_terms)
            * 6 * max(1, len(query_terms))
        )

    @classmethod
    def _tokenize(cls, value: str) -> set[str]:
        return {
            term for term in re.findall(r"[a-z0-9]+", value.lower())
            if term not in cls._STOP_WORDS
        }
