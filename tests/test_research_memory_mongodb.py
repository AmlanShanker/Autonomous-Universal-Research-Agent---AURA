import pytest

from config.settings import MONGO_URL
from memory.research_memory import ResearchMemoryItem
from storage.database.mongo import MongoDatabase


@pytest.fixture
def database():
    database = MongoDatabase(
        connection_string=MONGO_URL,
        database_name="aura_test",
    )

    if not database.ping():
        pytest.skip("MongoDB is not reachable.")

    database.research_memory.delete_many({})

    yield database

    database.research_memory.delete_many({})
    database.client.close()


def create_memory_item(
    memory_id: str = "memory_001",
) -> ResearchMemoryItem:
    return ResearchMemoryItem(
        memory_id=memory_id,
        run_id="run_001",
        task_id="task_a",
        research_question="Investigate retrieval performance.",
        content="Dense retrieval improved recall.",
        tags=["retrieval", "recall"],
    )


def test_mongodb_persists_research_memory(database):
    item = create_memory_item()

    database.save_research_memory(item)

    stored = database.get_research_memory("memory_001")

    assert stored is not None
    assert stored.memory_id == "memory_001"
    assert stored.content == "Dense retrieval improved recall."


def test_mongodb_lists_memory_for_run(database):
    database.save_research_memory(
        create_memory_item("memory_001")
    )

    second = create_memory_item("memory_002")
    second.run_id = "run_002"

    database.save_research_memory(second)

    results = database.list_research_memory_for_run(
        "run_001"
    )

    assert len(results) == 1
    assert results[0].memory_id == "memory_001"


def test_mongodb_searches_research_memory(database):
    database.save_research_memory(
        create_memory_item()
    )

    second = ResearchMemoryItem(
        memory_id="memory_002",
        run_id="run_002",
        task_id="task_b",
        research_question="Investigate database performance.",
        content="Indexing improved latency.",
        tags=["database", "latency"],
    )

    database.save_research_memory(second)

    results = database.search_research_memory(
        "retrieval recall"
    )

    assert len(results) == 1
    assert results[0].memory_id == "memory_001"


def test_mongodb_updates_existing_memory(database):
    item = create_memory_item()

    database.save_research_memory(item)

    item.content = "Dense retrieval significantly improved recall."

    database.save_research_memory(item)

    stored = database.get_research_memory(
        "memory_001"
    )

    assert stored is not None
    assert stored.content == (
        "Dense retrieval significantly improved recall."
    )