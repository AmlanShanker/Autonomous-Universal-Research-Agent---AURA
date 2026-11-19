from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ResearchSource(BaseModel):
    """A normalized scholarly source with explicit retrieval provenance."""

    source_id: str
    canonical_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    url: str
    doi: str | None = None
    arxiv_id: str | None = None
    publisher: str | None = None
    publication_year: int | None = None
    source_provider: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieved_query: str | None = None
    credibility_score: float = 0.0
    credibility_reasons: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
