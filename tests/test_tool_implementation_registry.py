from storage.database.memory import InMemoryDatabase
from storage.models.research import (
    ToolDefinition,
    ToolImplementation,
)
from tools.registry import ToolRegistry


def test_register_implementation_links_it_to_definition():
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

    implementation = ToolImplementation(
        implementation_id=(
            "experiment_designer_builtin_v1"
        ),
        tool_id="experiment_designer",
        implementation_type="builtin",
        status="active",
    )

    registry.register_implementation(
        implementation
    )

    assert registry.has_implementation(
        "experiment_designer_builtin_v1"
    )

    assert (
        "experiment_designer_builtin_v1"
        in definition.implementation_ids
    )

    implementations = (
        registry.get_implementations_for_tool(
            "experiment_designer"
        )
    )

    assert len(implementations) == 1

    assert (
        implementations[0].implementation_id
        == "experiment_designer_builtin_v1"
    )

    persisted = (
        database.get_tool_implementation(
            "experiment_designer_builtin_v1"
        )
    )

    assert persisted is not None


def test_multiple_implementations_can_belong_to_one_tool():
    registry = ToolRegistry()

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

    first = ToolImplementation(
        implementation_id=(
            "experiment_designer_builtin_v1"
        ),
        tool_id="experiment_designer",
        implementation_type="builtin",
        status="active",
    )

    second = ToolImplementation(
        implementation_id=(
            "experiment_designer_generated_v1"
        ),
        tool_id="experiment_designer",
        implementation_type="generated",
        status="draft",
    )

    registry.register_implementation(
        first
    )

    registry.register_implementation(
        second
    )

    implementations = (
        registry.get_implementations_for_tool(
            "experiment_designer"
        )
    )

    assert len(implementations) == 2

    implementation_ids = {
        implementation.implementation_id
        for implementation in implementations
    }

    assert implementation_ids == {
        "experiment_designer_builtin_v1",
        "experiment_designer_generated_v1",
    }


def test_unknown_tool_definition_rejects_implementation():
    registry = ToolRegistry()

    implementation = ToolImplementation(
        implementation_id="unknown_v1",
        tool_id="unknown_tool",
        implementation_type="builtin",
    )

    try:
        registry.register_implementation(
            implementation
        )
        assert False
    except ValueError as error:
        assert (
            "tool definition"
            in str(error).lower()
        )


def test_duplicate_implementation_is_rejected():
    registry = ToolRegistry()

    definition = ToolDefinition(
        tool_id="experiment_designer",
        name="experiment_designer",
        description=(
            "Design standardized research experiments."
        ),
    )

    registry.register_definition(
        definition
    )

    implementation = ToolImplementation(
        implementation_id=(
            "experiment_designer_builtin_v1"
        ),
        tool_id="experiment_designer",
        implementation_type="builtin",
    )

    registry.register_implementation(
        implementation
    )

    try:
        registry.register_implementation(
            implementation
        )
        assert False
    except ValueError as error:
        assert (
            "already registered"
            in str(error).lower()
        )
