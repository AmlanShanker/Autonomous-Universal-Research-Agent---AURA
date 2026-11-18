import uuid
from datetime import datetime, timezone

from memory.embedding_provider import (
    EmbeddingProvider,
    build_memory_embedding_text,
    validate_embedding,
)
from memory.research_memory import ResearchMemoryItem
from storage.database.base import Database


class ResearchMemoryWriter:
    """Persist reusable findings and optional semantic embeddings."""

    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.database = database
        self.embedding_provider = embedding_provider

    def write_finding(
        self,
        run_id: str,
        task_id: str,
        research_question: str,
        content: str,
        memory_type: str = "finding",
        tags: list[str] | None = None,
        metadata: dict | None = None,
        confidence: float = 0.5,
        source_result_id: str | None = None,
        source_tool_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> ResearchMemoryItem:
        if not content.strip():
            raise ValueError("Research memory content cannot be empty.")

        normalized_tags = tags or []
        embedding = None
        if self.embedding_provider is not None:
            text = build_memory_embedding_text(
                research_question, content, normalized_tags
            )
            embedding = validate_embedding(
                self.embedding_provider.embed(text)
            )

        item = ResearchMemoryItem(
            memory_id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            research_question=research_question,
            content=content,
            memory_type=memory_type,
            tags=normalized_tags,
            metadata=metadata or {},
            confidence=confidence,
            source_result_id=source_result_id,
            source_tool_id=source_tool_id,
            embedding=embedding,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.database.save_research_memory(item)
        return item
