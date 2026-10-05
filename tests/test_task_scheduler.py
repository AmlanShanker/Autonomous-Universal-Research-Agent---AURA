import pytest

from agents.task_scheduler import TaskScheduler
from storage.models.research import ResearchTask


def create_task(
    task_id: str,
    dependencies: list[str] | None = None,
    status: str = "pending",
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        description=f"Task {task_id}",
        task_type="test",
        dependencies=dependencies or [],
        status=status,
    )


def test_task_without_dependencies_is_ready():
    tasks = [
        create_task("task_a"),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task_a"
    ]


def test_task_with_completed_dependency_is_ready():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
        create_task(
            "task_b",
            dependencies=["task_a"],
        ),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task_b"
    ]


def test_task_with_pending_dependency_is_not_ready():
    tasks = [
        create_task("task_a"),
        create_task(
            "task_b",
            dependencies=["task_a"],
        ),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert ready == [
        tasks[0]
    ]


def test_multiple_independent_tasks_are_ready():
    tasks = [
        create_task("task_a"),
        create_task("task_b"),
        create_task("task_c"),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert {
        task.task_id
        for task in ready
    } == {
        "task_a",
        "task_b",
        "task_c",
    }


def test_completed_tasks_are_not_returned_as_ready():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert ready == []


def test_task_with_multiple_completed_dependencies_is_ready():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
        create_task(
            "task_b",
            status="completed",
        ),
        create_task(
            "task_c",
            dependencies=[
                "task_a",
                "task_b",
            ],
        ),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task_c"
    ]


def test_task_with_one_incomplete_dependency_is_not_ready():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
        create_task(
            "task_b",
        ),
        create_task(
            "task_c",
            dependencies=[
                "task_a",
                "task_b",
            ],
        ),
    ]

    scheduler = TaskScheduler(tasks)

    ready = scheduler.get_ready_tasks()

    assert {
        task.task_id
        for task in ready
    } == {
        "task_b"
    }


def test_unknown_dependency_is_rejected():
    tasks = [
        create_task(
            "task_a",
            dependencies=["missing_task"],
        ),
    ]

    scheduler = TaskScheduler(tasks)

    with pytest.raises(ValueError):
        scheduler.get_ready_tasks()


def test_scheduler_detects_pending_tasks():
    tasks = [
        create_task("task_a"),
        create_task(
            "task_b",
            status="completed",
        ),
    ]

    scheduler = TaskScheduler(tasks)

    assert scheduler.has_pending_tasks() is True


def test_scheduler_detects_when_no_tasks_are_pending():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
    ]

    scheduler = TaskScheduler(tasks)

    assert scheduler.has_pending_tasks() is False


def test_scheduler_detects_ready_tasks():
    tasks = [
        create_task("task_a"),
    ]

    scheduler = TaskScheduler(tasks)

    assert scheduler.has_ready_tasks() is True


def test_scheduler_detects_when_no_tasks_are_ready():
    tasks = [
        create_task(
            "task_a",
            status="completed",
        ),
    ]

    scheduler = TaskScheduler(tasks)

    assert scheduler.has_ready_tasks() is False