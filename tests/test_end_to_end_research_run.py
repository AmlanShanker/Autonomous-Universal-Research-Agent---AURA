from agents.research_director import ResearchDirector
from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.execution import ExecutionResult
from storage.models.research import (
    ResearchRunStatus,
    ToolDefinition,
)
from tools.registry import ToolRegistry


class FakeLLM:
    """
    Deterministic LLM replacement for the end-to-end test.

    This keeps the test independent of an external LLM provider.
    """

    def generate(self, prompt: str) -> str:
        if "Create a structured research plan" in prompt:
            return """
{
    "question": "Investigate retrieval performance.",
    "objectives": [
        "Research the topic",
        "Analyze the research findings"
    ],
    "hypotheses": [
        "The research evidence will provide useful findings"
    ],
    "required_capabilities": [
        "literature_search"
    ],
    "experiments": [
        "Search the scholarly literature"
    ],
    "success_criteria": [
        "Research tasks complete successfully"
    ]
}
"""

        if "Create an executable research task graph" in prompt:
            return """
{
    "tasks": [
        {
            "task_id": "task_a",
            "description": "Search the scholarly literature.",
            "task_type": "research",
            "dependencies": [],
            "required_tools": ["literature_search"],
            "tool_inputs": {
                "literature_search": {
                    "query": "retrieval performance",
                    "max_results": 3
                }
            }
        },
        {
            "task_id": "task_b",
            "description": "Analyze the literature findings.",
            "task_type": "analysis",
            "dependencies": ["task_a"],
            "required_tools": [],
            "tool_inputs": {}
        }
    ]
}
"""

        raise AssertionError(
            "FakeLLM received an unexpected prompt."
        )


def create_tool_registry(
    database,
) -> ToolRegistry:
    """
    Create a registry containing the real built-in
    literature-search tool and its persistent definition.
    """

    registry = ToolRegistry()

    definition = ToolDefinition(
        tool_id="literature_search",
        name="literature_search",
        description=(
            "Search scholarly literature and return "
            "relevant research papers."
        ),
        capability="literature_search",
        input_schema={
            "query": "string",
            "max_results": "integer",
        },
        output_schema={
            "query": "string",
            "results": "list",
            "count": "integer",
        },
        status="active",
        implementation_ids=[],
    )

    registry.register_definition(
        definition
    )

    database.save_tool(
        definition
    )

    validation = registry.validate_tool(
        "literature_search"
    )

    assert validation.valid
    assert registry.has_validated(
        "literature_search"
    )

    return registry


def test_end_to_end_research_run_completes_with_context():
    database = InMemoryDatabase()

    director = ResearchDirector(
        database=database,
        llm_client=FakeLLM(),
    )

    plan = director.create_plan(
        "Investigate retrieval performance."
    )

    tasks = director.create_tasks(plan)

    assert len(tasks) == 2

    run = director.start_run(
        run_id="run_001",
        question=plan.question,
    )

    run.tasks = [
        task.task_id
        for task in tasks
    ]

    database.save_research_run(run)

    tool_registry = create_tool_registry(
        database
    )

    executor = TaskExecutor(
        database=database,
        tasks=tasks,
        run_id=run.run_id,
        tool_registry=tool_registry,
    )

    executor.run()

    completed_run = database.get_research_run(
        "run_001"
    )

    assert completed_run is not None

    assert (
        completed_run.status
        == ResearchRunStatus.COMPLETED
    )

    assert set(
        completed_run.completed_tasks
    ) == {
        "task_a",
        "task_b",
    }

    assert completed_run.failed_tasks == []

    stored_result = next(
        result
        for result in database.execution_results.values()
        if result.task_id == "task_a"
    )

    assert isinstance(
        stored_result,
        ExecutionResult,
    )

    assert stored_result.status == "success"

    assert (
        stored_result.output["query"]
        == "retrieval performance"
    )

    dependency_context = (
        executor.get_task_context(
            tasks[1]
        )
    )

    assert len(dependency_context) == 1

    assert (
        dependency_context[0].task_id
        == "task_a"
    )

    assert (
        dependency_context[0].output["query"]
        == "retrieval performance"
    )