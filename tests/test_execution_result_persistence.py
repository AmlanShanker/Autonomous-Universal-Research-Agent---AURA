from storage.database.memory import InMemoryDatabase
from storage.models.execution import ExecutionResult


def test_execution_result_persistence():
    database = InMemoryDatabase()

    result = ExecutionResult(
        result_id="result_001",
        run_id="run_001",
        task_id="task_001",
        tool_id="literature_search",
        implementation_id="literature_search_builtin_v1",
        implementation_version="1.0.0",
        status="success",
        output={
            "query": "transformer efficiency",
            "count": 2,
        },
        metadata={
            "source": "arXiv",
        },
    )

    database.save_execution_result(result)

    stored = database.get_execution_result(
        "result_001"
    )

    assert stored is not None
    assert stored.result_id == "result_001"
    assert stored.run_id == "run_001"
    assert stored.task_id == "task_001"
    assert stored.tool_id == "literature_search"
    assert (
        stored.implementation_id
        == "literature_search_builtin_v1"
    )
    assert stored.status == "success"
    assert (
        stored.output["query"]
        == "transformer efficiency"
    )
    assert stored.output["count"] == 2
    assert stored.metadata["source"] == "arXiv"


def test_missing_execution_result_returns_none():
    database = InMemoryDatabase()

    result = database.get_execution_result(
        "does_not_exist"
    )

    assert result is None
