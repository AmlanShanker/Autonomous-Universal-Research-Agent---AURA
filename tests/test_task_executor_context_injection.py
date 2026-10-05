from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.context import ResearchContextItem
from storage.models.research import ResearchTask
from tools.registry import ToolRegistry


def create_executor(
    database,
    tasks,
):
    return TaskExecutor(
        database=database,
        tasks=tasks,
        run_id="run_001",
        tool_registry=ToolRegistry(),
    )


def test_executor_builds_dependency_context_for_downstream_task():
    database = InMemoryDatabase()

    task_a = ResearchTask(
        task_id="task_a",
        description="Research the topic.",
        task_type="research",
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Analyze the research.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    executor = create_executor(
        database,
        [task_a, task_b],
    )

    executor.context.items.append(
        ResearchContextItem(
            result_id="result_a",
            task_id="task_a",
            tool_id="tool_a",
            status="success",
            output={
                "finding": "Important finding"
            },
        )
    )

    context = executor.build_task_context(
        task_b
    )

    assert len(
        context["dependencies"]
    ) == 1

    assert (
        context["dependencies"][0]["task_id"]
        == "task_a"
    )

    assert (
        context["dependencies"][0]["output"]["finding"]
        == "Important finding"
    )


def test_executor_exposes_context_in_tool_inputs():
    database = InMemoryDatabase()

    task_a = ResearchTask(
        task_id="task_a",
        description="Research the topic.",
        task_type="research",
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Analyze the research.",
        task_type="analysis",
        dependencies=["task_a"],
        tool_inputs={
            "literature_search": {
                "query": "follow-up research",
                "max_results": 3,
            }
        },
    )

    executor = create_executor(
        database,
        [task_a, task_b],
    )

    executor.context.items.append(
        ResearchContextItem(
            result_id="result_a",
            task_id="task_a",
            tool_id="tool_a",
            status="success",
            output={
                "finding": "Previous evidence"
            },
        )
    )

    inputs = executor._build_tool_inputs(
        task_b,
        "literature_search",
    )

    assert inputs["query"] == "follow-up research"
    assert inputs["max_results"] == 3

    assert (
        len(
            inputs["research_context"]["dependencies"]
        )
        == 1
    )

    assert (
        inputs["research_context"]["dependencies"][0][
            "output"
        ]["finding"]
        == "Previous evidence"
    )


def test_executor_does_not_inject_unrelated_context():
    database = InMemoryDatabase()

    task_a = ResearchTask(
        task_id="task_a",
        description="Research the topic.",
        task_type="research",
    )

    task_b = ResearchTask(
        task_id="task_b",
        description="Analyze the research.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    task_c = ResearchTask(
        task_id="task_c",
        description="Unrelated research.",
        task_type="research",
    )

    executor = create_executor(
        database,
        [task_a, task_b, task_c],
    )

    executor.context.items.extend(
        [
            ResearchContextItem(
                result_id="result_a",
                task_id="task_a",
                tool_id="tool_a",
                status="success",
                output={
                    "finding": "Relevant"
                },
            ),
            ResearchContextItem(
                result_id="result_c",
                task_id="task_c",
                tool_id="tool_c",
                status="success",
                output={
                    "finding": "Unrelated"
                },
            ),
        ]
    )

    inputs = executor._build_tool_inputs(
        task_b,
        "literature_search",
    )

    dependencies = (
        inputs["research_context"]["dependencies"]
    )

    assert len(dependencies) == 1
    assert (
        dependencies[0]["task_id"]
        == "task_a"
    )