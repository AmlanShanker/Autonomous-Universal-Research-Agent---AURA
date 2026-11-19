from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchTask
from storage.models.task_checkpoint import TaskCheckpoint
from tools.registry import ToolRegistry


def make_executor(database, task):
    return TaskExecutor(
        tasks=[task],
        database=database,
        run_id="checkpoint-run",
        tool_registry=ToolRegistry(database=database),
    )


def test_checkpoint_round_trips_as_an_isolated_snapshot():
    database = InMemoryDatabase()
    checkpoint = TaskCheckpoint(
        checkpoint_id="checkpoint-run:task-1",
        run_id="checkpoint-run",
        task_id="task-1",
        input_fingerprint="fingerprint-1",
        status="completed",
        result={"answer": "persisted"},
    )

    database.save_task_checkpoint(checkpoint)
    loaded = database.get_task_checkpoint("checkpoint-run", "task-1")
    loaded.result["answer"] = "mutated"

    assert database.get_task_checkpoint("checkpoint-run", "task-1").result == {
        "answer": "persisted"
    }


def test_completed_checkpoint_skips_task_after_executor_restart():
    database = InMemoryDatabase()
    original = ResearchTask(
        task_id="task-1",
        description="A deterministic test task",
        task_type="test",
    )
    first = make_executor(database, original)
    assert first.execute_task(original) is True
    persisted_result_count = len(database.list_execution_results_for_run("checkpoint-run"))

    resumed_task = ResearchTask(
        task_id="task-1",
        description="A deterministic test task",
        task_type="test",
    )
    resumed = make_executor(database, resumed_task)

    assert resumed_task.status == "pending"
    assert resumed.execute_task(resumed_task) is True
    assert resumed_task.status == "completed"
    assert len(database.list_execution_results_for_run("checkpoint-run")) == persisted_result_count


def test_changed_task_inputs_invalidate_completed_checkpoint():
    database = InMemoryDatabase()
    original = ResearchTask(
        task_id="task-1",
        description="Original task description",
        task_type="test",
    )
    executor = make_executor(database, original)
    executor._save_task_checkpoint(original, "completed", {"task_result": {"ok": True}})

    changed = ResearchTask(
        task_id="task-1",
        description="Changed task description",
        task_type="test",
        status="completed",
    )
    resumed = make_executor(database, changed)

    assert changed.status == "pending"
    assert resumed.get_ready_tasks() == [changed]


def test_checkpoint_state_is_updated_for_each_task_attempt():
    database = InMemoryDatabase()
    task = ResearchTask(
        task_id="task-1",
        description="A deterministic test task",
        task_type="test",
    )
    executor = make_executor(database, task)

    assert executor.execute_task(task) is True
    checkpoint = database.get_task_checkpoint("checkpoint-run", "task-1")

    assert checkpoint.status == "completed"
    assert checkpoint.input_fingerprint == executor._task_fingerprint(task)
    assert checkpoint.result["task_id"] == "task-1"
