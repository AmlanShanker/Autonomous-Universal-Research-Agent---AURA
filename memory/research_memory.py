from datetime import datetime, timezone
import math

from pydantic import BaseModel, Field, field_validator


class ResearchMemoryItem(BaseModel):
    """
    A persistent piece of knowledge produced by a research run.

    Embeddings are optional so existing records and installations remain
    compatible when no semantic embedding provider is configured.
    """

    memory_id: str
    run_id: str
    task_id: str
    research_question: str
    content: str
    memory_type: str = "finding"
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated support strength, from 0.0 to 1.0.",
    )
    source_result_id: str | None = None
    source_tool_id: str | None = None
    embedding: list[float] | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime | None = None
    archived: bool = False

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value):
        if value is None:
            return value
        if not value:
            raise ValueError("Research-memory embedding cannot be empty.")
        normalized = []
        for number in value:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError("Research-memory embedding values must be finite numbers.")
            number = float(number)
            if not math.isfinite(number):
                raise ValueError("Research-memory embedding values must be finite numbers.")
            normalized.append(number)
        return normalized

    @field_validator("created_at", "updated_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Research-memory timestamps must be timezone-aware.")
        return value


class ResearchMemory:
    """In-process research-memory abstraction used by lightweight workflows."""

    def __init__(self):
        self.items: dict[str, ResearchMemoryItem] = {}

    def add(self, item: ResearchMemoryItem) -> None:
        if item.memory_id in self.items:
            raise ValueError(
                f"Research memory item '{item.memory_id}' already exists."
            )
        self.items[item.memory_id] = item

    def get(self, memory_id: str) -> ResearchMemoryItem | None:
        return self.items.get(memory_id)

    def list_for_run(self, run_id: str) -> list[ResearchMemoryItem]:
        return [
            item for item in self.items.values()
            if item.run_id == run_id
        ]

    def search(self, query: str) -> list[ResearchMemoryItem]:
        query_terms = {
            term.lower() for term in query.split() if term.strip()
        }
        if not query_terms:
            return []

        now = datetime.now(timezone.utc)
        results = []
        for item in self.items.values():
            if item.archived or (
                item.expires_at is not None and item.expires_at <= now
            ):
                continue
            searchable_text = (
                f"{item.research_question} {item.content} "
                f"{' '.join(item.tags)}"
            ).lower()
            if any(term in searchable_text for term in query_terms):
                results.append(item)
        return results
