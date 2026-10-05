from agents.context_builder import TaskContextBuilder
from storage.models.context import ResearchContextItem
from storage.models.research import ResearchTask


def test_builder_includes_dependency_context():
    task = ResearchTask(
        task_id="task_b",
        description="Analyze previous research.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    context_items = [
        ResearchContextItem(
            result_id="result_a",
            task_id="task_a",
            tool_id="tool_a",
            status="success",
            output={
                "finding": "Important finding"
            },
        )
    ]

    builder = TaskContextBuilder()

    result = builder.build(
        task,
        context_items,
    )

    assert len(result["dependencies"]) == 1
    assert result["dependencies"][0]["task_id"] == "task_a"
    assert (
        result["dependencies"][0]["output"]["finding"]
        == "Important finding"
    )


def test_builder_excludes_unrelated_context():
    task = ResearchTask(
        task_id="task_b",
        description="Analyze previous research.",
        task_type="analysis",
        dependencies=["task_a"],
    )

    context_items = [
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

    builder = TaskContextBuilder()

    result = builder.build(
        task,
        context_items,
    )

    assert len(result["dependencies"]) == 1
    assert result["dependencies"][0]["task_id"] == "task_a"


def test_builder_returns_empty_context_without_dependencies():
    task = ResearchTask(
        task_id="task_a",
        description="Initial research.",
        task_type="research",
    )

    context_items = [
        ResearchContextItem(
            result_id="result_x",
            task_id="task_x",
            tool_id="tool_x",
            status="success",
            output={
                "finding": "Something"
            },
        )
    ]

    builder = TaskContextBuilder()

    result = builder.build(
        task,
        context_items,
    )

    assert result == {
        "dependencies": []
    }