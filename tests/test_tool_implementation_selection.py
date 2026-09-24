from storage.models.research import (
    ToolDefinition,
    ToolImplementation,
)
from tools.registry import ToolRegistry


def create_registry_with_definition():
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

    return registry


def test_only_validated_active_implementations_are_usable():
    registry = create_registry_with_definition()

    usable = ToolImplementation(
        implementation_id=(
            "experiment_designer_v1"
        ),
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    draft = ToolImplementation(
        implementation_id=(
            "experiment_designer_v2"
        ),
        tool_id="experiment_designer",
        version="2.0.0",
        implementation_type="generated",
        validation_status="unvalidated",
        status="draft",
    )

    failed = ToolImplementation(
        implementation_id=(
            "experiment_designer_v3"
        ),
        tool_id="experiment_designer",
        version="3.0.0",
        implementation_type="generated",
        validation_status="failed",
        status="inactive",
    )

    registry.register_implementation(
        usable
    )

    registry.register_implementation(
        draft
    )

    registry.register_implementation(
        failed
    )

    result = (
        registry.get_usable_implementations(
            "experiment_designer"
        )
    )

    assert len(result) == 1

    assert (
        result[0].implementation_id
        == "experiment_designer_v1"
    )


def test_select_implementation_returns_usable_implementation():
    registry = create_registry_with_definition()

    implementation = ToolImplementation(
        implementation_id=(
            "experiment_designer_v1"
        ),
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="builtin",
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
        == "experiment_designer_v1"
    )


def test_select_implementation_returns_none_when_no_usable_implementation_exists():
    registry = create_registry_with_definition()

    implementation = ToolImplementation(
        implementation_id=(
            "experiment_designer_v1"
        ),
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="generated",
        validation_status="unvalidated",
        status="draft",
    )

    registry.register_implementation(
        implementation
    )

    selected = registry.select_implementation(
        "experiment_designer"
    )

    assert selected is None


def test_select_implementation_is_deterministic():
    registry = create_registry_with_definition()

    first = ToolImplementation(
        implementation_id=(
            "experiment_designer_v1"
        ),
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    second = ToolImplementation(
        implementation_id=(
            "experiment_designer_v2"
        ),
        tool_id="experiment_designer",
        version="2.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    registry.register_implementation(
        first
    )

    registry.register_implementation(
        second
    )

    selected = registry.select_implementation(
        "experiment_designer"
    )

    assert selected is not None

    assert (
        selected.implementation_id
        == "experiment_designer_v2"
    )
