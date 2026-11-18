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
    assert item.confidence == 0.5
    assert item.source_result_id is None
    assert item.source_tool_id is None


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

def test_writer_persists_confidence_and_execution_provenance():
    database = InMemoryDatabase()
    writer = ResearchMemoryWriter(database)

    item = writer.write_finding(
        run_id="run_001",
        task_id="task_a",
        research_question="Investigate retrieval performance.",
        content="Dense retrieval improved recall.",
        confidence=0.9,
        source_result_id="result_123",
        source_tool_id="literature_search",
    )

    stored = database.get_research_memory(item.memory_id)

    assert stored is not None
    assert stored.confidence == 0.9
    assert stored.source_result_id == "result_123"
    assert stored.source_tool_id == "literature_search"


def test_writer_rejects_confidence_outside_zero_to_one():
    database = InMemoryDatabase()
    writer = ResearchMemoryWriter(database)

    try:
        writer.write_finding(
            run_id="run_001",
            task_id="task_a",
            research_question="Test question.",
            content="Finding.",
            confidence=1.5,
        )
        assert False, "Expected invalid confidence to be rejected."
    except ValueError:
        pass
