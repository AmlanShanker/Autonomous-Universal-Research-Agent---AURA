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
    tasks: list[ResearchTask],
    run_id: str = "run_001",
):
    return TaskExecutor(
        tasks=tasks,
        database=database,
        run_id=run_id,
        tool_registry=FakeToolRegistry(),
    )


def test_executor_respects_task_dependencies():
    database = InMemoryDatabase()

    tasks = [
        ResearchTask(
            task_id="task_a",
            description="First task",
            task_type="test",
        ),
        ResearchTask(
            task_id="task_b",
            description="Second task",
            task_type="test",
            dependencies=["task_a"],
        ),
        ResearchTask(
            task_id="task_c",
            description="Third task",
            task_type="test",
            dependencies=["task_b"],
        ),
    ]

    run = ResearchRun(
        run_id="run_001",
        research_question="Dependency test",
        status=ResearchRunStatus.PLANNED,
        tasks=[
            "task_a",
            "task_b",
            "task_c",
        ],
    )

    database.save_research_run(run)

    executor = create_executor(
        database,
        tasks,
    )

    execution_order = []

    original_execute = (
        executor._execute_task_logic
    )

    def tracking_execute(task):
        execution_order.append(
            task.task_id
        )
        return original_execute(task)

    executor._execute_task_logic = (
        tracking_execute
    )

    executor.run()

    assert execution_order == [
        "task_a",
        "task_b",
        "task_c",
    ]

    stored_run = database.get_research_run(
        "run_001"
    )

    assert stored_run is not None
    assert stored_run.status == (
        ResearchRunStatus.COMPLETED
    )


def test_executor_runs_independent_tasks_in_same_ready_batch():
    database = InMemoryDatabase()

    tasks = [
        ResearchTask(
            task_id="task_a",
            description="Independent A",
            task_type="test",
        ),
        ResearchTask(
            task_id="task_b",
            description="Independent B",
            task_type="test",
        ),
        ResearchTask(
            task_id="task_c",
            description="Depends on A and B",
            task_type="test",
            dependencies=[
                "task_a",
                "task_b",
            ],
        ),
    ]

    run = ResearchRun(
        run_id="run_001",
        research_question="Parallel readiness test",
        status=ResearchRunStatus.PLANNED,
        tasks=[
            "task_a",
            "task_b",
            "task_c",
        ],
    )

    database.save_research_run(run)

    executor = create_executor(
        database,
        tasks,
    )

    first_ready_tasks = executor.get_ready_tasks()

    assert {
        task.task_id
        for task in first_ready_tasks
    } == {
        "task_a",
        "task_b",
    }


def test_executor_does_not_execute_dependent_task_early():
    database = InMemoryDatabase()

    tasks = [
        ResearchTask(
            task_id="task_a",
            description="First task",
            task_type="test",
        ),
        ResearchTask(
            task_id="task_b",
            description="Dependent task",
            task_type="test",
            dependencies=["task_a"],
        ),
    ]

    executor = create_executor(
        database,
        tasks,
    )

    ready = executor.get_ready_tasks()

    assert [
        task.task_id
        for task in ready
    ] == [
        "task_a"
    ]

    assert tasks[1].status == "pending"


def test_executor_recalculates_readiness_after_completion():
    database = InMemoryDatabase()

    tasks = [
        ResearchTask(
            task_id="task_a",
            description="First task",
            task_type="test",
        ),
        ResearchTask(
            task_id="task_b",
            description="Dependent task",
            task_type="test",
            dependencies=["task_a"],
        ),
    ]

    executor = create_executor(
        database,
        tasks,
    )

    initial_ready = executor.get_ready_tasks()

    assert [
        task.task_id
        for task in initial_ready
    ] == [
        "task_a"
    ]

    tasks[0].status = "completed"

    executor.completed_tasks.add(
        "task_a"
    )

    next_ready = executor.get_ready_tasks()

    assert [
        task.task_id
        for task in next_ready
    ] == [
        "task_b"
    ]