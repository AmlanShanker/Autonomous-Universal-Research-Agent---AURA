import re

from pydantic import BaseModel, Field

from memory.research_memory import ResearchMemoryItem


class ResearchMemoryMatch(BaseModel):
    """A stored finding related to a candidate finding."""

    memory_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    reason: str


class ResearchMemoryAnalysis(BaseModel):
    """Deterministic duplicate and potential-conflict matches."""

    duplicates: list[ResearchMemoryMatch] = Field(default_factory=list)
    potential_conflicts: list[ResearchMemoryMatch] = Field(default_factory=list)


class ResearchMemoryAnalyzer:
    """
    Find likely duplicate and potentially conflicting research findings.

    This first version is deliberately deterministic: it compares normalized
    word sets and explicit polarity markers. A conflict is a review signal,
    not proof that either finding is incorrect.
    """

    _NEGATION = {"not", "no", "never", "without", "n't"}
    _POSITIVE = {
        "increase", "increases", "increased", "improve", "improves",
        "improved", "boost", "boosts", "higher", "high", "positive",
        "support", "supports", "supported", "effective", "safe",
    }
    _NEGATIVE = {
        "decrease", "decreases", "decreased", "reduce", "reduces",
        "reduced", "worsen", "worsens", "lower", "low", "negative",
        "refute", "refutes", "refuted", "ineffective", "unsafe",
    }

    def __init__(
        self,
        database,
        duplicate_threshold: float = 0.9,
        conflict_threshold: float = 0.5,
    ):
        for name, value in (
            ("duplicate_threshold", duplicate_threshold),
            ("conflict_threshold", conflict_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be greater than 0 and at most 1.")

        self.database = database
        self.duplicate_threshold = float(duplicate_threshold)
        self.conflict_threshold = float(conflict_threshold)

    def analyze(
        self,
        item: ResearchMemoryItem,
    ) -> ResearchMemoryAnalysis:
        """
        Compare a candidate finding with searchable stored findings.

        The candidate itself is excluded when it is already stored. Matches
        are sorted by descending similarity and then by memory ID.
        """
        candidate_terms = self._tokens(item.content)
        if not candidate_terms:
            return ResearchMemoryAnalysis()

        candidates = self.database.search_research_memory(item.content)
        duplicates = []
        conflicts = []
        seen_ids = set()

        for existing in candidates:
            if (
                existing.memory_id == item.memory_id
                or existing.memory_id in seen_ids
            ):
                continue
            seen_ids.add(existing.memory_id)

            existing_terms = self._tokens(existing.content)
            similarity = self._jaccard(candidate_terms, existing_terms)
            if similarity == 0.0:
                continue

            if similarity >= self.duplicate_threshold:
                duplicates.append(
                    ResearchMemoryMatch(
                        memory_id=existing.memory_id,
                        similarity=similarity,
                        reason="Content is lexically near-identical.",
                    )
                )
            elif (
                similarity >= self.conflict_threshold
                and self._opposite_polarity(candidate_terms, existing_terms)
            ):
                conflicts.append(
                    ResearchMemoryMatch(
                        memory_id=existing.memory_id,
                        similarity=similarity,
                        reason="Similar content contains opposing polarity markers.",
                    )
                )

        sort_key = lambda match: (-match.similarity, match.memory_id)
        duplicates.sort(key=sort_key)
        conflicts.sort(key=sort_key)

        return ResearchMemoryAnalysis(
            duplicates=duplicates,
            potential_conflicts=conflicts,
        )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower()))

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @classmethod
    def _opposite_polarity(
        cls,
        left: set[str],
        right: set[str],
    ) -> bool:
        left_negated = bool(left & cls._NEGATION)
        right_negated = bool(right & cls._NEGATION)
        if left_negated != right_negated:
            return True

        left_positive = bool(left & cls._POSITIVE)
        left_negative = bool(left & cls._NEGATIVE)
        right_positive = bool(right & cls._POSITIVE)
        right_negative = bool(right & cls._NEGATIVE)

        return (
            (left_positive and right_negative)
            or (left_negative and right_positive)
        )
