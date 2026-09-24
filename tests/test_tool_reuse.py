from config.settings import MONGO_URL

from agents.task_executor import TaskExecutor

from storage.database.mongo import MongoDatabase
from storage.models.research import (
    ResearchRun,
    ResearchTask,
    ToolDefinition,
)

from tools.registry import ToolRegistry


class FailingToolBuilder:
    """
    Test double used to prove that TaskExecutor does
    not invoke the Tool Builder when a matching
    persistent definition already exists.
    """

    def build(self, request):
        raise AssertionError(
            "ToolBuilder should not be called when "
            "a matching tool definition already exists."
        )


def main():

    # -----------------------------------------
    # Database
    # -----------------------------------------

    database = MongoDatabase(
        connection_string=MONGO_URL,
        database_name="aura",
    )

    # -----------------------------------------
    # Create a Known Tool Definition
    # -----------------------------------------

    capability = "dataset_download"

    definition = ToolDefinition(
        tool_id="DatasetDownloader",
        name="DatasetDownloader",
        capability=capability,
        description=(
            "Downloads datasets from remote sources."
        ),
        purpose=(
            "Provides AURA with a reusable dataset "
            "download capability."
        ),
        input_schema={
            "inputs": [
                "source_url",
                "destination_path",
            ]
        },
        output_schema={
            "outputs": [
                "local_path",
                "download_status",
            ]
        },
        dependencies=[
            "requests>=2.28.0",
        ],
        validation_requirements=[
            "Verify downloaded file exists.",
            "Verify checksum when provided.",
        ],
        status="draft",
    )

    database.save_tool(
        definition
    )

    print(
        "\nTool definition saved:"
    )

    print(
        f"- {definition.name}"
    )

    # -----------------------------------------
    # Create Research Run
    # -----------------------------------------

    run = ResearchRun(
        run_id="test_reuse_run",
        research_question=(
            "Test capability-aware tool reuse."
        ),
        status="planned",
    )

    database.save_research_run(
        run
    )

    # -----------------------------------------
    # Create Research Task
    # -----------------------------------------

    task = ResearchTask(
        task_id="reuse_task_001",
        description=(
            "Download a benchmark dataset."
        ),
        task_type="data_acquisition",
        dependencies=[],
        required_tools=[
            capability
        ],
    )

    database.save_research_task(
        task
    )

    # -----------------------------------------
    # Create Registry
    # -----------------------------------------

    registry = ToolRegistry(
        database=database
    )

    # -----------------------------------------
    # Verify Definition Exists
    # -----------------------------------------

    found = (
        registry.find_definition_by_capability(
            capability
        )
    )

    print(
        "\n=== PERSISTED CAPABILITY ==="
    )

    print(
        f"Requested capability: "
        f"{capability}"
    )

    if found is None:

        raise RuntimeError(
            "Expected persisted tool definition "
            "was not found."
        )

    print(
        f"Found tool: {found.name}"
    )

    print(
        f"Status: {found.status}"
    )

    # -----------------------------------------
    # Create Executor
    # -----------------------------------------

    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id="test_reuse_run",
        tool_registry=registry,
    )

    # Replace the real Tool Builder with a test
    # implementation that fails if called.

    executor.tool_builder = (
        FailingToolBuilder()
    )

    # -----------------------------------------
    # Execute Task
    # -----------------------------------------

    print(
        "\n=== TASK EXECUTION ==="
    )

    completed = executor.execute_task(
        task
    )

    # -----------------------------------------
    # Verify Task Was Blocked
    # -----------------------------------------

    if completed:

        raise RuntimeError(
            "Task incorrectly completed even though "
            "the known tool definition is not executable."
        )

    # -----------------------------------------
    # Verify Reuse
    # -----------------------------------------

    if capability not in (
        executor.reused_tool_definitions
    ):

        raise RuntimeError(
            "Existing tool definition was not "
            "recorded as reused."
        )

    reused = (
        executor.reused_tool_definitions[
            capability
        ]
    )

    if reused.name != definition.name:

        raise RuntimeError(
            "Incorrect tool definition was reused."
        )

    # -----------------------------------------
    # Verify Builder Was Not Called
    # -----------------------------------------

    if executor.generated_tool_specs:

        raise RuntimeError(
            "A new tool specification was generated "
            "even though a matching definition already existed."
        )

    # -----------------------------------------
    # Verify Task Status
    # -----------------------------------------

    stored_task = (
        database.get_research_task(
            task.task_id
        )
    )

    if stored_task is None:

        raise RuntimeError(
            "Research task was not persisted."
        )

    if stored_task.status != "blocked":

        raise RuntimeError(
            "Task should be blocked because the "
            "known definition is not executable."
        )

    # -----------------------------------------
    # Verify Run Status
    # -----------------------------------------

    stored_run = (
        database.get_research_run(
            "test_reuse_run"
        )
    )

    if stored_run is None:

        raise RuntimeError(
            "Research run was not persisted."
        )

    # execute_task() only blocks the task.
    # TaskExecutor.run() is responsible for setting
    # the overall research run to blocked.

    print(
        "\n=== REUSE RESULT ==="
    )

    print(
        "Existing definition was reused."
    )

    print(
        "Tool Builder was not called."
    )

    print(
        "Task remained blocked because the "
        "definition is not executable."
    )

    print(
        "\nTool reuse execution test passed!"
    )


if __name__ == "__main__":
    main()
