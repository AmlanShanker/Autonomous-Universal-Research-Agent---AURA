from memory.research_memory_writer import ResearchMemoryWriter
from storage.database.memory import InMemoryDatabase


def test_writer_persists_research_finding():
    database = InMemoryDatabase()
    writer = ResearchMemoryWriter(database)

    item = writer.write_finding(
        run_id="run_001",
        task_id="task_a",
        research_question="Investigate retrieval performance.",
        content="Dense retrieval improved recall.",
        tags=["retrieval", "recall"],
    )

    stored = database.get_research_memory(
        item.memory_id
    )

    assert stored is not None
    assert stored.run_id == "run_001"
    assert stored.task_id == "task_a"
    assert stored.content == (
        "Dense retrieval improved recall."
    )
    assert stored.tags == [
        "retrieval",
        "recall",
    ]


def test_writer_supports_memory_type_and_metadata():
    database = InMemoryDatabase()
    writer = ResearchMemoryWriter(database)

    item = writer.write_finding(
        run_id="run_001",
        task_id="task_a",
        research_question="Investigate retrieval performance.",
        content="The experiment supports the hypothesis.",
        memory_type="conclusion",
        tags=["retrieval"],
        metadata={
            "confidence": "high",
            "source": "experiment",
        },
    )

    assert item.memory_type == "conclusion"
    assert item.metadata["confidence"] == "high"
    assert item.metadata["source"] == "experiment"


def test_writer_rejects_empty_content():
    database = InMemoryDatabase()
    writer = ResearchMemoryWriter(database)

    try:
        writer.write_finding(
            run_id="run_001",
            task_id="task_a",
            research_question="Test question.",
            content="",
        )
        assert False, (
            "Expected empty memory content "
            "to raise ValueError."
        )
    except ValueError as error:
        assert "cannot be empty" in str(error)