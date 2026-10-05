from agents.task_executor import TaskExecutor

from memory.research_memory_writer import ResearchMemoryWriter

from storage.database.memory import InMemoryDatabase

from storage.models.research import (
    ResearchRun,
    ResearchTask,
    ToolDefinition,
)

from tools.registry import ToolRegistry


class FakeFindingExtractor:
    def __init__(self):
        self.calls = []

    def extract(
        self,
        research_question,
        execution_result,
    ):
        self.calls.append(
            {
                "research_question": research_question,
                "execution_result": execution_result,
            }
        )

        return [
            {
                "content": (
                    "Chunk size affects retrieval performance."
                ),
                "memory_type": "finding",
                "tags": [
                    "rag",
                    "retrieval",
                ],
                "metadata": {
                    "source": "test",
                },
            }
        ]


def test_task_executor_extracts_and_persists_research_memory():
    database = InMemoryDatabase()

    run = ResearchRun(
        run_id="run-memory-integration",
        research_question=(
            "How does chunk size affect "
            "RAG retrieval performance?"
        ),
    )

    database.save_research_run(
        run
    )

    task = ResearchTask(
        task_id="task-literature",
        description=(
            "Search literature about RAG "
            "chunk size."
        ),
        task_type="research",
        required_tools=[
            "literature_search"
        ],
        tool_inputs={
            "literature_search": {
                "query": (
                    "RAG chunk size "
                    "retrieval performance"
                ),
                "max_results": 1,
            }
        },
    )

    database.save_research_task(
        task
    )

    definition = ToolDefinition(
        tool_id="literature_search",
        name="literature_search",
        capability="literature_search",
        description=(
            "Search scholarly literature."
        ),
        purpose=(
            "Find relevant research papers."
        ),
        status="draft",
    )

    database.save_tool(
        definition
    )

    registry = ToolRegistry(
        database=database
    )

    tool = registry.get(
        "literature_search"
    )

    assert tool is not None

    def fake_execute(
        query,
        max_results=5,
    ):
        return {
            "query": query,
            "results": [
                {
                    "title": "Test research paper",
                    "publication_year": 2025,
                }
            ],
            "count": 1,
        }

    tool.execute = fake_execute

    validation_result = (
        registry.validate_tool(
            "literature_search"
        )
    )

    assert validation_result.valid is True

    finding_extractor = (
        FakeFindingExtractor()
    )

    memory_writer = (
        ResearchMemoryWriter(
            database
        )
    )

    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id=run.run_id,
        tool_registry=registry,
        finding_extractor=(
            finding_extractor
        ),
        memory_writer=(
            memory_writer
        ),
    )

    assert (
        executor.execute_task(
            task
        )
        is True
    )

    assert (
        len(
            finding_extractor.calls
        )
        == 1
    )

    extraction_call = (
        finding_extractor.calls[0]
    )

    assert (
        extraction_call[
            "research_question"
        ]
        == run.research_question
    )

    execution_result = (
        extraction_call[
            "execution_result"
        ]
    )

    assert (
        execution_result.status
        == "success"
    )

    memories = (
        database.list_research_memory_for_run(
            run.run_id
        )
    )

    assert len(memories) == 1

    memory = memories[0]

    assert (
        memory.content
        == "Chunk size affects "
        "retrieval performance."
    )

    assert (
        memory.memory_type
        == "finding"
    )

    assert memory.tags == [
        "rag",
        "retrieval",
    ]

    assert memory.metadata == {
        "source": "test",
    }

    assert (
        memory.run_id
        == run.run_id
    )

    assert (
        memory.task_id
        == task.task_id
    )
