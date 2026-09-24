from storage.models.research import ToolDefinition
from tools.registry import ToolRegistry


def create_dataset_definition() -> ToolDefinition:
    return ToolDefinition(
        tool_id="dataset_download",
        name="dataset_download",
        capability="dataset_download",
        description=(
            "Download a dataset from an HTTP or HTTPS "
            "URL into AURA's controlled storage directory."
        ),
        purpose=(
            "Allow AURA to acquire datasets required "
            "for research experiments."
        ),
        input_schema={
            "inputs": [
                "source_url",
                "destination_path",
                "checksum",
            ],
        },
        output_schema={
            "outputs": [
                "status",
                "local_path",
                "size_bytes",
                "sha256",
                "checksum_verified",
            ],
        },
        dependencies=[
            "requests",
        ],
        validation_requirements=[
            "Verify HTTP and HTTPS URLs.",
            "Prevent path traversal.",
            "Calculate SHA-256 checksum.",
        ],
        status="draft",
    )


def test_registered_tool_is_not_automatically_validated():
    registry = ToolRegistry()

    assert registry.has(
        "dataset_download"
    )

    assert not registry.has_validated(
        "dataset_download"
    )


def test_valid_tool_changes_status_to_validated():
    registry = ToolRegistry()

    definition = create_dataset_definition()

    registry.register_definition(
        definition
    )

    result = registry.validate_tool(
        "dataset_download"
    )

    assert result.valid is True

    assert definition.status == "validated"

    assert registry.has_validated(
        "dataset_download"
    )

    assert (
        registry.get_validated(
            "dataset_download"
        )
        is registry.get(
            "dataset_download"
        )
    )


def test_invalid_tool_changes_status_to_failed():
    registry = ToolRegistry()

    definition = create_dataset_definition()

    definition.name = "wrong_name"

    registry.register_definition(
        definition
    )

    result = registry.validate_tool(
        "dataset_download"
    )

    assert result.valid is False

    assert definition.status == "failed"

    assert not registry.has_validated(
        "dataset_download"
    )


def test_failed_tool_can_become_validated_after_fix():
    registry = ToolRegistry()

    definition = create_dataset_definition()

    definition.name = "wrong_name"

    registry.register_definition(
        definition
    )

    first_result = registry.validate_tool(
        "dataset_download"
    )

    assert first_result.valid is False

    assert definition.status == "failed"

    assert not registry.has_validated(
        "dataset_download"
    )

    # Fix the persistent definition.

    definition.name = "dataset_download"

    second_result = registry.validate_tool(
        "dataset_download"
    )

    assert second_result.valid is True

    assert definition.status == "validated"

    assert registry.has_validated(
        "dataset_download"
    )


def test_missing_definition_prevents_validation():
    registry = ToolRegistry()

    result_error = None

    try:
        registry.validate_tool(
            "dataset_download"
        )

    except ValueError as error:
        result_error = error

    assert result_error is not None

    assert (
        "No ToolDefinition exists"
        in str(result_error)
    )

    assert not registry.has_validated(
        "dataset_download"
    )
