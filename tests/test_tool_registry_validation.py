from storage.models.research import ToolDefinition
from tools.implementations import DatasetDownloadTool
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
        status="validated",
    )


def test_registered_tool_is_not_automatically_validated():
    registry = ToolRegistry()

    assert registry.has(
        "dataset_download"
    )

    assert not registry.has_validated(
        "dataset_download"
    )


def test_valid_tool_can_be_validated():
    registry = ToolRegistry()

    definition = create_dataset_definition()

    registry.register_definition(
        definition
    )

    result = registry.validate_tool(
        "dataset_download"
    )

    assert result.valid is True

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


def test_invalid_tool_is_not_validated():
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

    assert not registry.has_validated(
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
