from memory.finding_extractor import ResearchFindingExtractor
from memory.research_memory_writer import ResearchMemoryWriter
from storage.database.memory import InMemoryDatabase
from storage.models.execution import ExecutionResult


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def create_execution_result() -> ExecutionResult:
    return ExecutionResult(
        result_id="result_001",
        run_id="run_001",
        task_id="task_a",
        tool_id="literature_search",
        implementation_id="literature_search_builtin",
        output={
            "query": "retrieval performance",
            "results": [
                {
                    "title": "Dense Retrieval Study",
                    "publication_year": 2024,
                }
            ],
            "count": 1,
        },
    )


def test_finding_extraction_and_memory_persistence():
    llm = FakeLLM(
        """
        {
            "findings": [
                {
                    "content": "Dense retrieval research includes a 2024 study.",
                    "memory_type": "finding",
                    "tags": ["retrieval", "dense"],
                    "metadata": {
                        "source": "literature_search"
                    }
                }
            ]
        }
        """
    )

    database = InMemoryDatabase()

    extractor = ResearchFindingExtractor(llm)
    writer = ResearchMemoryWriter(database)

    execution_result = create_execution_result()

    findings = extractor.extract(
        research_question="Investigate retrieval performance.",
        execution_result=execution_result,
    )

    assert len(findings) == 1

    memory_item = writer.write_finding(
        run_id=execution_result.run_id,
        task_id=execution_result.task_id,
        research_question="Investigate retrieval performance.",
        content=findings[0]["content"],
        memory_type=findings[0]["memory_type"],
        tags=findings[0]["tags"],
        metadata=findings[0]["metadata"],
    )

    assert memory_item.run_id == "run_001"
    assert memory_item.task_id == "task_a"
    assert memory_item.content == (
        "Dense retrieval research includes a 2024 study."
    )

    stored = database.get_research_memory(
        memory_item.memory_id
    )

    assert stored is not None
    assert stored.content == memory_item.content
    assert stored.tags == ["retrieval", "dense"]
    assert stored.metadata["source"] == "literature_search"


def test_pipeline_supports_multiple_findings():
    llm = FakeLLM(
        """
        {
            "findings": [
                {
                    "content": "Finding one.",
                    "memory_type": "finding",
                    "tags": ["one"],
                    "metadata": {}
                },
                {
                    "content": "Finding two.",
                    "memory_type": "hypothesis",
                    "tags": ["two"],
                    "metadata": {
                        "confidence": "medium"
                    }
                }
            ]
        }
        """
    )

    database = InMemoryDatabase()

    extractor = ResearchFindingExtractor(llm)
    writer = ResearchMemoryWriter(database)

    execution_result = create_execution_result()

    findings = extractor.extract(
        research_question="Investigate retrieval performance.",
        execution_result=execution_result,
    )

    assert len(findings) == 2

    memory_items = []

    for finding in findings:
        item = writer.write_finding(
            run_id=execution_result.run_id,
            task_id=execution_result.task_id,
            research_question="Investigate retrieval performance.",
            content=finding["content"],
            memory_type=finding["memory_type"],
            tags=finding["tags"],
            metadata=finding["metadata"],
        )

        memory_items.append(item)

    stored_items = database.list_research_memory_for_run(
        "run_001"
    )

    assert len(stored_items) == 2
    assert {
        item.content for item in stored_items
    } == {
        "Finding one.",
        "Finding two.",
    }