import pytest

from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.research import (
    ResearchTask,
    ToolDefinition,
)
from tools.registry import ToolRegistry


def create_executor(
    task: ResearchTask,
    database: InMemoryDatabase,
) -> TaskExecutor:
    registry = ToolRegistry(
        database=database
    )

    definition = ToolDefinition(
        tool_id="literature_search",
        name="literature_search",
        capability="literature_search",
        description=(
            "Search scholarly literature on arXiv "
            "and return relevant research papers."
        ),
        purpose=(
            "Find relevant scholarly research papers."
        ),
    )

    registry.register_definition(
        definition
    )

    database.save_tool(
        definition
    )

    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id="run_001",
        tool_registry=registry,
    )

    return executor


def test_successful_tool_execution_persists_execution_result(
    monkeypatch,
):
    database = InMemoryDatabase()

    task = ResearchTask(
        task_id="task_001",
        description="Search research literature.",
        task_type="literature_search",
        required_tools=[
            "literature_search"
        ],
        tool_inputs={
            "literature_search": {
                "query": "transformer efficiency",
                "max_results": 2,
            }
        },
    )

    executor = create_executor(
        task,
        database,
    )

    def fake_execute(
        *,
        query,
        max_results,
    ):
        return {
            "query": query,
            "count": 2,
            "results": [
                {
                    "title": "Paper One",
                },
                {
                    "title": "Paper Two",
                },
            ],
        }

    monkeypatch.setattr(
        executor.tool_registry
        .get("literature_search"),
        "execute",
        fake_execute,
    )

    completed = executor.execute_task(
        task
    )

    assert completed is True

    results = list(
        database.execution_results.values()
    )

    assert len(results) == 1

    result = results[0]

    assert result.run_id == "run_001"
    assert result.task_id == "task_001"
    assert result.tool_id == "literature_search"
    assert (
        result.implementation_id
        == "literature_search_builtin_v1"
    )
    assert result.implementation_version == "1.0.0"
    assert result.status == "success"

    assert (
        result.output["query"]
        == "transformer efficiency"
    )

    assert result.output["count"] == 2

    assert (
        result.metadata["execution_type"]
        == "tool"
    )


def test_failed_tool_execution_persists_failed_result(
    monkeypatch,
):
    database = InMemoryDatabase()

    task = ResearchTask(
        task_id="task_002",
        description="Search research literature.",
        task_type="literature_search",
        required_tools=[
            "literature_search"
        ],
        tool_inputs={
            "literature_search": {
                "query": "transformer efficiency",
                "max_results": 2,
            }
        },
    )

    executor = create_executor(
        task,
        database,
    )

    def failing_execute(
        *,
        query,
        max_results,
    ):
        raise RuntimeError(
            "Simulated literature search failure."
        )

    monkeypatch.setattr(
        executor.tool_registry
        .get("literature_search"),
        "execute",
        failing_execute,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated literature search failure.",
    ):
        executor.execute_task(
            task
        )

    results = list(
        database.execution_results.values()
    )

    assert len(results) == 1

    result = results[0]

    assert result.run_id == "run_001"
    assert result.task_id == "task_002"
    assert result.tool_id == "literature_search"
    assert (
        result.implementation_id
        == "literature_search_builtin_v1"
    )
    assert result.implementation_version == "1.0.0"
    assert result.status == "failed"

    assert (
        result.output["error"]
        == "Simulated literature search failure."
    )

    assert (
        result.metadata["execution_type"]
        == "tool"
    )


def test_execution_results_use_unique_result_ids(
    monkeypatch,
):
    database = InMemoryDatabase()

    task = ResearchTask(
        task_id="task_003",
        description="Search research literature.",
        task_type="literature_search",
        required_tools=[
            "literature_search"
        ],
        tool_inputs={
            "literature_search": {
                "query": "transformer efficiency",
                "max_results": 1,
            }
        },
    )

    executor = create_executor(
        task,
        database,
    )

    def fake_execute(
        *,
        query,
        max_results,
    ):
        return {
            "query": query,
            "count": 1,
        }

    monkeypatch.setattr(
        executor.tool_registry
        .get("literature_search"),
        "execute",
        fake_execute,
    )

    executor.execute_task(
        task
    )

    task.status = "pending"
    executor.completed_tasks.clear()

    executor.execute_task(
        task
    )

    results = list(
        database.execution_results.values()
    )

    assert len(results) == 2

    result_ids = {
        result.result_id
        for result in results
    }

    assert len(result_ids) == 2
