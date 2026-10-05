from agents.task_scheduler import TaskScheduler
from storage.models.context import ResearchContext
from storage.models.research import ResearchTask


def test_context_starts_empty():
    context = ResearchContext(run_id="run_001")

    assert context.run_id == "run_001"
    assert context.items == []


def test_context_can_store_execution_result():
    context = ResearchContext(run_id="run_001")

    context.add_result(
        result_id="result_001",
        task_id="task_a",
        tool_id="literature_search",
        status="success",
        output={
            "papers": [
                {"title": "Example Paper"}
            ]
        },
    )

    assert len(context.items) == 1
    assert context.items[0].result_id == "result_001"
    assert context.items[0].task_id == "task_a"
    assert context.items[0].output["papers"][0]["title"] == "Example Paper"


def test_context_can_filter_results_by_task():
    context = ResearchContext(run_id="run_001")

    context.add_result(
        result_id="result_a",
        task_id="task_a",
        tool_id="tool_a",
        status="success",
        output={"value": "A"},
    )

    context.add_result(
        result_id="result_b",
        task_id="task_b",
        tool_id="tool_b",
        status="success",
        output={"value": "B"},
    )

    results = context.get_results_for_task("task_a")

    assert len(results) == 1
    assert results[0].result_id == "result_a"


def test_context_returns_dependency_evidence():
    context = ResearchContext(run_id="run_001")

    context.add_result(
        result_id="result_a",
        task_id="task_a",
        tool_id="tool_a",
        status="success",
        output={"value": "A"},
    )

    context.add_result(
        result_id="result_b",
        task_id="task_b",
        tool_id="tool_b",
        status="success",
        output={"value": "B"},
    )

    context.add_result(
        result_id="result_c",
        task_id="task_c",
        tool_id="tool_c",
        status="success",
        output={"value": "C"},
    )

    task_d = ResearchTask(
        task_id="task_d",
        description="Use previous findings.",
        task_type="analysis",
        dependencies=["task_a", "task_b"],
    )

    dependency_context = context.get_dependency_context(task_d)

    assert len(dependency_context) == 2

    result_ids = {
        item.result_id
        for item in dependency_context
    }

    assert result_ids == {"result_a", "result_b"}


def test_context_ignores_unrelated_results():
    context = ResearchContext(run_id="run_001")

    context.add_result(
        result_id="result_a",
        task_id="task_a",
        tool_id="tool_a",
        status="success",
        output={"value": "A"},
    )

    context.add_result(
        result_id="result_c",
        task_id="task_c",
        tool_id="tool_c",
        status="success",
        output={"value": "C"},
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Use task A.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    dependency_context = context.get_dependency_context(task_b)

    assert len(dependency_context) == 1
    assert dependency_context[0].task_id == "task_a"