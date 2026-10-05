import pytest

from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.research import (
    ResearchRun,
    ResearchRunStatus,
    ResearchTask,
)


class FakeToolRegistry:
    pass


def create_executor(
    database: InMemoryDatabase,
    run_id: str = "run_001",
):
    task = ResearchTask(
        task_id="task_001",
        description="Test task",
        task_type="test",
        required_tools=[],
    )

    return TaskExecutor(
        tasks=[task],
        database=database,
        run_id=run_id,
        tool_registry=FakeToolRegistry(),
    )


def test_executor_run_transitions_planned_to_completed():
    database = InMemoryDatabase()

    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.PLANNED,
    )

    database.save_research_run(run)

    executor = create_executor(
        database
    )

    executor.run()

    stored_run = database.get_research_run(
        "run_001"
    )

    assert stored_run is not None
    assert stored_run.status == ResearchRunStatus.COMPLETED
    assert stored_run.completed_tasks == [
        "task_001"
    ]


def test_executor_can_transition_blocked_run_back_to_running():
    database = InMemoryDatabase()

    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.BLOCKED,
    )

    database.save_research_run(run)

    executor = create_executor(
        database
    )

    executor.run()

    stored_run = database.get_research_run(
        "run_001"
    )

    assert stored_run is not None
    assert stored_run.status == ResearchRunStatus.COMPLETED


def test_executor_marks_run_failed_when_task_execution_fails():
    database = InMemoryDatabase()

    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.PLANNED,
    )

    database.save_research_run(run)

    task = ResearchTask(
        task_id="task_001",
        description="Test failing task",
        task_type="test",
        required_tools=[],
    )

    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id="run_001",
        tool_registry=FakeToolRegistry(),
    )

    def failing_logic(task):
        raise RuntimeError(
            "simulated task failure"
        )

    executor._execute_task_logic = failing_logic

    with pytest.raises(RuntimeError):
        executor.run()

    stored_run = database.get_research_run(
        "run_001"
    )

    assert stored_run is not None
    assert stored_run.status == ResearchRunStatus.FAILED
    assert stored_run.failed_tasks == [
        "task_001"
    ]


def test_executor_rejects_starting_a_completed_run():
    database = InMemoryDatabase()

    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.COMPLETED,
    )

    database.save_research_run(run)

    executor = create_executor(
        database
    )

    with pytest.raises(ValueError):
        executor.run()

    stored_run = database.get_research_run(
        "run_001"
    )

    assert stored_run is not None
    assert stored_run.status == ResearchRunStatus.COMPLETED
