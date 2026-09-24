from storage.database.memory import InMemoryDatabase
from storage.models.research import (
    ResearchTask,
    ToolDefinition,
)

from tools.registry import ToolRegistry


def test_executable_tool_to_usable_implementation_lifecycle():
    database = InMemoryDatabase()

    registry = ToolRegistry(
        database=database
    )

    definition = ToolDefinition(
        tool_id="literature_search",
        name="literature_search",
        capability="literature_search",
        description=(
            "Search scholarly literature on arXiv "
            "and return relevant research papers."
        ),
    )

    registry.register_definition(
        definition
    )

    # -----------------------------------------
    # Before validation
    # -----------------------------------------

    assert (
        registry.get_usable_implementations(
            "literature_search"
        )
        == []
    )

    # -----------------------------------------
    # Validate executable tool
    # -----------------------------------------

    validation = registry.validate_tool(
        "literature_search"
    )

    assert validation.valid is True

    # -----------------------------------------
    # Implementation must now exist
    # -----------------------------------------

    implementation = (
        registry.get_implementation(
            "literature_search_builtin_v1"
        )
    )

    assert implementation is not None

    assert (
        implementation.tool_id
        == "literature_search"
    )

    assert (
        implementation.implementation_type
        == "builtin"
    )

    assert (
        implementation.validation_status
        == "validated"
    )

    assert implementation.status == "active"

    # -----------------------------------------
    # Definition must reference implementation
    # -----------------------------------------

    definition = registry.get_definition(
        "literature_search"
    )

    assert definition is not None

    assert (
        "literature_search_builtin_v1"
        in definition.implementation_ids
    )

    # -----------------------------------------
    # Implementation must be usable
    # -----------------------------------------

    usable = (
        registry.get_usable_implementations(
            "literature_search"
        )
    )

    assert len(usable) == 1

    assert (
        usable[0].implementation_id
        == "literature_search_builtin_v1"
    )

    # -----------------------------------------
    # Implementation must be selectable
    # -----------------------------------------

    selected = registry.select_implementation(
        "literature_search"
    )

    assert selected is not None

    assert (
        selected.implementation_id
        == "literature_search_builtin_v1"
    )

    # -----------------------------------------
    # Implementation must resolve to executable
    # tool
    # -----------------------------------------

    resolved = registry.resolve_implementation(
        selected
    )

    assert resolved is not None

    assert (
        resolved.name
        == "literature_search"
    )

    # -----------------------------------------
    # Combined selection + resolution
    # -----------------------------------------

    executable = (
        registry.get_executable_implementation(
            "literature_search"
        )
    )

    assert executable is not None

    resolved_implementation, tool = executable

    assert (
        resolved_implementation.implementation_id
        == "literature_search_builtin_v1"
    )

    assert (
        tool.name
        == "literature_search"
    )


def test_generated_implementation_is_not_executable():
    database = InMemoryDatabase()

    registry = ToolRegistry(
        database=database
    )

    definition = ToolDefinition(
        tool_id="experiment_designer",
        name="experiment_designer",
        capability="experiment_design",
        description=(
            "Design standardized research experiments."
        ),
    )

    registry.register_definition(
        definition
    )

    from storage.models.research import (
        ToolImplementation,
    )

    implementation = ToolImplementation(
        implementation_id=(
            "experiment_designer_generated_v1"
        ),
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="generated",
        validation_status="validated",
        status="active",
    )

    registry.register_implementation(
        implementation
    )

    selected = registry.select_implementation(
        "experiment_designer"
    )

    assert selected is not None

    assert (
        selected.implementation_id
        == "experiment_designer_generated_v1"
    )

    resolved = registry.resolve_implementation(
        selected
    )

    assert resolved is None

    executable = (
        registry.get_executable_implementation(
            "experiment_designer"
        )
    )

    assert executable is None


def test_task_model_supports_tool_inputs_for_lifecycle_execution():
    task = ResearchTask(
        task_id="task_001",
        description="Search for RAG research.",
        task_type="literature_search",
        required_tools=[
            "literature_search"
        ],
        tool_inputs={
            "literature_search": {
                "query": (
                    "RAG document retrieval"
                ),
                "max_results": 2,
            }
        },
    )

    assert (
        task.tool_inputs[
            "literature_search"
        ]["max_results"]
        == 2
    )
