from memory.research_memory import (
    ResearchMemory,
    ResearchMemoryItem,
)


def test_memory_stores_and_retrieves_item():
    memory = ResearchMemory()

    item = ResearchMemoryItem(
        memory_id="memory_001",
        run_id="run_001",
        task_id="task_a",
        research_question=(
            "Investigate retrieval performance."
        ),
        content=(
            "Dense retrieval improved recall."
        ),
        tags=[
            "retrieval",
            "recall",
        ],
    )

    memory.add(item)

    stored = memory.get(
        "memory_001"
    )

    assert stored is not None

    assert stored.content == (
        "Dense retrieval improved recall."
    )


def test_memory_lists_items_for_run():
    memory = ResearchMemory()

    memory.add(
        ResearchMemoryItem(
            memory_id="memory_001",
            run_id="run_001",
            task_id="task_a",
            research_question="Question A",
            content="Finding A",
        )
    )

    memory.add(
        ResearchMemoryItem(
            memory_id="memory_002",
            run_id="run_002",
            task_id="task_b",
            research_question="Question B",
            content="Finding B",
        )
    )

    results = memory.list_for_run(
        "run_001"
    )

    assert len(results) == 1

    assert results[0].memory_id == (
        "memory_001"
    )


def test_memory_search_finds_relevant_items():
    memory = ResearchMemory()

    memory.add(
        ResearchMemoryItem(
            memory_id="memory_001",
            run_id="run_001",
            task_id="task_a",
            research_question=(
                "Investigate retrieval performance."
            ),
            content=(
                "Dense retrieval improved recall."
            ),
            tags=[
                "retrieval",
                "recall",
            ],
        )
    )

    memory.add(
        ResearchMemoryItem(
            memory_id="memory_002",
            run_id="run_002",
            task_id="task_b",
            research_question=(
                "Investigate database performance."
            ),
            content=(
                "Database indexing improved latency."
            ),
            tags=[
                "database",
                "latency",
            ],
        )
    )

    results = memory.search(
        "retrieval recall"
    )

    assert len(results) == 1

    assert results[0].memory_id == (
        "memory_001"
    )


def test_memory_search_returns_empty_for_unknown_query():
    memory = ResearchMemory()

    memory.add(
        ResearchMemoryItem(
            memory_id="memory_001",
            run_id="run_001",
            task_id="task_a",
            research_question="Research question",
            content="Research finding",
        )
    )

    results = memory.search(
        "quantum computing"
    )

    assert results == []


def test_memory_rejects_duplicate_memory_ids():
    memory = ResearchMemory()

    item = ResearchMemoryItem(
        memory_id="memory_001",
        run_id="run_001",
        task_id="task_a",
        research_question="Question",
        content="Finding",
    )

    memory.add(item)

    try:
        memory.add(item)
        assert False, (
            "Expected duplicate memory ID "
            "to raise ValueError."
        )
    except ValueError as error:
        assert (
            "already exists"
            in str(error)
        )