from storage.database.memory import InMemoryDatabase
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
        implementation_id="experiment_designer_v1",
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    draft = ToolImplementation(
        implementation_id="experiment_designer_v2",
        tool_id="experiment_designer",
        version="2.0.0",
        implementation_type="generated",
        validation_status="unvalidated",
        status="draft",
    )

    failed = ToolImplementation(
        implementation_id="experiment_designer_v3",
        tool_id="experiment_designer",
        version="3.0.0",
        implementation_type="generated",
        validation_status="failed",
        status="inactive",
    )

    registry.register_implementation(usable)
    registry.register_implementation(draft)
    registry.register_implementation(failed)

    result = registry.get_usable_implementations(
        "experiment_designer"
    )

    assert len(result) == 1
    assert (
        result[0].implementation_id
        == "experiment_designer_v1"
    )


def test_select_implementation_returns_usable_implementation():
    registry = create_registry_with_definition()

    implementation = ToolImplementation(
        implementation_id="experiment_designer_v1",
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
        implementation_id="experiment_designer_v1",
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
        implementation_id="experiment_designer_v1",
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    second = ToolImplementation(
        implementation_id="experiment_designer_v2",
        tool_id="experiment_designer",
        version="2.0.0",
        implementation_type="builtin",
        validation_status="validated",
        status="active",
    )

    registry.register_implementation(first)
    registry.register_implementation(second)

    selected = registry.select_implementation(
        "experiment_designer"
    )

    assert selected is not None
    assert (
        selected.implementation_id
        == "experiment_designer_v2"
    )


def test_validate_tool_creates_implementation_and_makes_it_usable():
    database = InMemoryDatabase()

    registry = ToolRegistry(
        database=database
    )

    definition = ToolDefinition(
        tool_id="dataset_download",
        name="dataset_download",
        capability="dataset_download",
        description=(
            "Download datasets from HTTP or HTTPS sources."
        ),
    )

    registry.register_definition(
        definition
    )

    result = registry.validate_tool(
        "dataset_download"
    )

    assert result.valid is True

    implementation = registry.get_implementation(
        "dataset_download_builtin_v1"
    )

    assert implementation is not None
    assert (
        implementation.tool_id
        == "dataset_download"
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

    assert (
        "dataset_download_builtin_v1"
        in definition.implementation_ids
    )

    usable = registry.get_usable_implementations(
        "dataset_download"
    )

    assert len(usable) == 1
    assert (
        usable[0].implementation_id
        == "dataset_download_builtin_v1"
    )

    selected = registry.select_implementation(
        "dataset_download"
    )

    assert selected is not None
    assert (
        selected.implementation_id
        == "dataset_download_builtin_v1"
    )

    persisted = (
        database.get_tool_implementation(
            "dataset_download_builtin_v1"
        )
    )

    assert persisted is not None
    assert (
        persisted.validation_status
        == "validated"
    )
    assert persisted.status == "active"


def test_validate_tool_reuses_existing_implementation():
    database = InMemoryDatabase()

    registry = ToolRegistry(
        database=database
    )

    definition = ToolDefinition(
        tool_id="dataset_download",
        name="dataset_download",
        capability="dataset_download",
        description=(
            "Download datasets from HTTP or HTTPS sources."
        ),
    )

    registry.register_definition(
        definition
    )

    existing = ToolImplementation(
        implementation_id=(
            "dataset_download_custom_v1"
        ),
        tool_id="dataset_download",
        version="2.0.0",
        implementation_type="builtin",
        validation_status="unvalidated",
        status="draft",
    )

    registry.register_implementation(
        existing
    )

    result = registry.validate_tool(
        "dataset_download"
    )

    assert result.valid is True

    assert (
        registry.list_implementations()
        == ["dataset_download_custom_v1"]
    )

    updated = registry.get_implementation(
        "dataset_download_custom_v1"
    )

    assert updated is not None
    assert (
        updated.validation_status
        == "validated"
    )
    assert updated.status == "active"

    persisted = (
        database.get_tool_implementation(
            "dataset_download_custom_v1"
        )
    )

    assert persisted is not None
    assert (
        persisted.validation_status
        == "validated"
    )
    assert persisted.status == "active"
