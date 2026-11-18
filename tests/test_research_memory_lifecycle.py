from datetime import datetime, timedelta, timezone

import pytest

from memory.research_memory import ResearchMemoryItem
from memory.research_memory_lifecycle import ResearchMemoryLifecycle
from storage.database.memory import InMemoryDatabase


def add_item(database, memory_id="memory_1", **kwargs):
    item = ResearchMemoryItem(
        memory_id=memory_id,
        run_id="run_1",
        task_id="task_1",
        research_question="Research question",
        content="Original finding",
        **kwargs,
    )
    database.save_research_memory(item)
    return item


def test_lifecycle_updates_finding_and_timestamp():
    database = InMemoryDatabase()
    original = add_item(database)
    updated = ResearchMemoryLifecycle(database).update(
        original.memory_id,
        content="Updated finding",
        confidence=0.9,
    )

    assert updated.content == "Updated finding"
    assert updated.confidence == 0.9
    assert updated.updated_at >= original.updated_at
    assert database.get_research_memory(original.memory_id).content == "Updated finding"


def test_lifecycle_rejects_identity_and_unknown_field_updates():
    database = InMemoryDatabase()
    item = add_item(database)
    lifecycle = ResearchMemoryLifecycle(database)

    with pytest.raises(ValueError, match="Unsupported"):
        lifecycle.update(item.memory_id, run_id="another_run")
    with pytest.raises(KeyError, match="not found"):
        lifecycle.update("missing", content="No record")


def test_lifecycle_archive_and_restore_control_retrieval():
    database = InMemoryDatabase()
    item = add_item(database)
    lifecycle = ResearchMemoryLifecycle(database)

    lifecycle.archive(item.memory_id)
    assert database.search_research_memory("finding") == []

    lifecycle.restore(item.memory_id)
    assert [result.memory_id for result in database.search_research_memory("finding")] == [
        item.memory_id
    ]


def test_expired_memory_is_archived_and_excluded_from_search():
    database = InMemoryDatabase()
    expired = add_item(
        database,
        memory_id="expired",
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    add_item(database, memory_id="active")
    lifecycle = ResearchMemoryLifecycle(database)

    expired_ids = lifecycle.expire_due()

    assert expired_ids == [expired.memory_id]
    assert database.get_research_memory(expired.memory_id).archived is True
    assert [item.memory_id for item in database.search_research_memory("finding")] == [
        "active"
    ]


def test_expiry_requires_timezone_aware_now():
    database = InMemoryDatabase()
    lifecycle = ResearchMemoryLifecycle(database)

    with pytest.raises(ValueError, match="timezone-aware"):
        lifecycle.expire_due(datetime(2026, 1, 1))


def test_lifecycle_deletes_memory_and_reports_missing_items():
    database = InMemoryDatabase()
    item = add_item(database)
    lifecycle = ResearchMemoryLifecycle(database)

    assert lifecycle.delete(item.memory_id) is True
    assert lifecycle.delete(item.memory_id) is False
    assert database.get_research_memory(item.memory_id) is None


def test_lifecycle_recomputes_embedding_when_text_changes():
    class Provider:
        def embed(self, text):
            return [float(len(text)), 1.0]

    database = InMemoryDatabase()
    item = add_item(database, embedding=[1.0, 1.0])
    updated = ResearchMemoryLifecycle(
        database, embedding_provider=Provider()
    ).update(item.memory_id, content="A longer updated finding")

    assert updated.embedding == [float(len("Research question\nA longer updated finding")), 1.0]
