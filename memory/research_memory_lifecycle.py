from datetime import datetime, timezone

from memory.embedding_provider import (
    EmbeddingProvider,
    build_memory_embedding_text,
    validate_embedding,
)
from memory.research_memory import ResearchMemoryItem
from storage.database.base import Database


class ResearchMemoryLifecycle:
    """Manage updates, archival, expiry, and deletion of stored findings."""

    _MUTABLE_FIELDS = {
        "research_question",
        "content",
        "memory_type",
        "tags",
        "metadata",
        "confidence",
        "expires_at",
        "archived",
    }

    def __init__(
        self,
        database: Database,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.database = database
        self.embedding_provider = embedding_provider

    def update(self, memory_id: str, **changes) -> ResearchMemoryItem:
        unknown = set(changes) - self._MUTABLE_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported research-memory fields: {names}")

        existing = self.database.get_research_memory(memory_id)
        if existing is None:
            raise KeyError(f"Research memory '{memory_id}' was not found.")

        values = existing.model_dump()
        values.update(changes)
        values["updated_at"] = datetime.now(timezone.utc)

        text_fields_changed = bool(
            {"research_question", "content", "tags"} & set(changes)
        )
        if text_fields_changed and self.embedding_provider is not None:
            text = build_memory_embedding_text(
                values["research_question"],
                values["content"],
                values["tags"],
            )
            values["embedding"] = validate_embedding(
                self.embedding_provider.embed(text)
            )

        updated = ResearchMemoryItem.model_validate(values)
        self.database.update_research_memory(updated)
        return updated

    def archive(self, memory_id: str) -> ResearchMemoryItem:
        return self.update(memory_id, archived=True)

    def restore(self, memory_id: str) -> ResearchMemoryItem:
        return self.update(memory_id, archived=False)

    def delete(self, memory_id: str) -> bool:
        return self.database.delete_research_memory(memory_id)

    def expire_due(
        self,
        now: datetime | None = None,
    ) -> list[str]:
        """Archive all active memories whose expiry timestamp has passed."""
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("Expiry comparison time must be timezone-aware.")

        expired_ids = []
        for item in self.database.list_research_memory():
            expires_at = item.expires_at
            if (
                not item.archived
                and expires_at is not None
                and expires_at <= current_time
            ):
                self.update(item.memory_id, archived=True)
                expired_ids.append(item.memory_id)

        return expired_ids
