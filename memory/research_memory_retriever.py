import re

from memory.research_memory import ResearchMemoryItem


class ResearchMemoryRetriever:
    """
    Retrieve and rank reusable findings from previous research.

    Retrieval is deterministic and keyword-based. It uses the
    database search operation to find candidates, then ranks them
    by where query terms occur:

    - tags: 3 points per matching term
    - finding content: 2 points per matching term
    - original research question: 1 point per matching term

    Semantic/vector retrieval can be added behind this interface
    in a later milestone.
    """

    _STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }

    def __init__(self, database):
        self.database = database

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        exclude_run_id: str | None = None,
    ) -> list[ResearchMemoryItem]:
        """
        Return the most relevant stored findings for a query.

        Args:
            query: Research question or topic to search for.
            limit: Maximum number of findings to return.
            exclude_run_id: Optional run whose findings should
                not be returned, useful when retrieving prior
                knowledge for an in-progress run.

        Results are ordered by relevance, then by memory ID
        to make ties deterministic.
        """

        if not isinstance(query, str):
            raise TypeError("Research-memory query must be a string.")

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError("Research-memory result limit must be a positive integer.")

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        candidates = self.database.search_research_memory(query)

        ranked_items = []
        seen_memory_ids = set()

        for item in candidates:
            if item.memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(item.memory_id)

            if (
                exclude_run_id is not None
                and item.run_id == exclude_run_id
            ):
                continue

            score = self._score(item, query_terms)

            if score <= 0:
                continue

            ranked_items.append((score, item))

        ranked_items.sort(
            key=lambda entry: (
                -entry[0],
                entry[1].memory_id,
            )
        )

        return [
            item
            for _, item in ranked_items[:limit]
        ]

    @classmethod
    def _score(
        cls,
        item: ResearchMemoryItem,
        query_terms: set[str],
    ) -> int:
        tag_terms = cls._tokenize(" ".join(item.tags))
        content_terms = cls._tokenize(item.content)
        question_terms = cls._tokenize(item.research_question)

        return (
            3 * len(query_terms.intersection(tag_terms))
            + 2 * len(query_terms.intersection(content_terms))
            + len(query_terms.intersection(question_terms))
        )

    @classmethod
    def _tokenize(cls, value: str) -> set[str]:
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", value.lower())
            if term not in cls._STOP_WORDS
        }
        return terms
