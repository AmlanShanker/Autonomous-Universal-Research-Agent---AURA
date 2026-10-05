from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchTask
from tools.registry import ToolRegistry


def create_executor(
    database,
    tasks,
):
    tool_registry = ToolRegistry()

    return TaskExecutor(
        database=database,
        tasks=tasks,
        run_id="run_001",
        tool_registry=tool_registry,
    )


def test_executor_context_starts_with_run_id():
    database = InMemoryDatabase()

    tasks = [
        ResearchTask(
            task_id="task_a",
            description="First task.",
            task_type="research",
        )
    ]

    executor = create_executor(
        database,
        tasks,
    )

    assert executor.context.run_id == "run_001"
    assert executor.context.items == []


def test_executor_can_access_dependency_context():
    database = InMemoryDatabase()

    task_a = ResearchTask(
        task_id="task_a",
        description="First task.",
        task_type="research",
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Second task.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    executor = create_executor(
        database,
        [task_a, task_b],
    )

    executor.context.add_result(
        result_id="result_a",
        task_id="task_a",
        tool_id="tool_a",
        status="success",
        output={
            "finding": "Important research finding"
        },
    )

    dependency_context = executor.get_task_context(
        task_b
    )

    assert len(dependency_context) == 1
    assert dependency_context[0].task_id == "task_a"
    assert (
        dependency_context[0].output["finding"]
        == "Important research finding"
    )


def test_executor_does_not_expose_unrelated_context():
    database = InMemoryDatabase()

    task_a = ResearchTask(
        task_id="task_a",
        description="First task.",
        task_type="research",
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Second task.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    task_c = ResearchTask(
        task_id="task_c",
        description="Unrelated task.",
        task_type="research",
    )

    executor = create_executor(
        database,
        [task_a, task_b, task_c],
    )

    executor.context.add_result(
        result_id="result_a",
        task_id="task_a",
        tool_id="tool_a",
        status="success",
        output={
            "finding": "A"
        },
    )

    executor.context.add_result(
        result_id="result_c",
        task_id="task_c",
        tool_id="tool_c",
        status="success",
        output={
            "finding": "C"
        },
    )

    dependency_context = executor.get_task_context(
        task_b
    )

    assert len(dependency_context) == 1
    assert dependency_context[0].task_id == "task_a"