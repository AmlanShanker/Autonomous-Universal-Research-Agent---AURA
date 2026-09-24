from storage.models.research import ToolDefinition
from tools.implementations import DatasetDownloadTool
from tools.validator import ToolValidator


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


def test_dataset_download_tool_is_valid():
    tool = DatasetDownloadTool()

    definition = create_dataset_definition()

    validator = ToolValidator()

    result = validator.validate(
        tool=tool,
        definition=definition,
    )

    assert result.valid is True

    assert result.tool_id == "dataset_download"

    assert result.tool_name == "dataset_download"

    assert result.errors == []

    assert len(result.checks) > 0


def test_mismatched_tool_name_is_invalid():
    tool = DatasetDownloadTool()

    definition = create_dataset_definition()

    definition.name = "different_tool"

    validator = ToolValidator()

    result = validator.validate(
        tool=tool,
        definition=definition,
    )

    assert result.valid is False

    assert any(
        "name does not match"
        in error
        for error in result.errors
    )


def test_missing_definition_name_is_invalid():
    tool = DatasetDownloadTool()

    definition = create_dataset_definition()

    definition.name = ""

    validator = ToolValidator()

    result = validator.validate(
        tool=tool,
        definition=definition,
    )

    assert result.valid is False

    assert any(
        "empty name"
        in error
        for error in result.errors
    )


def test_missing_execute_method_is_invalid():
    class InvalidTool:
        @property
        def name(self):
            return "dataset_download"

        @property
        def description(self):
            return "Invalid tool."

    tool = InvalidTool()

    definition = create_dataset_definition()

    validator = ToolValidator()

    result = validator.validate(
        tool=tool,
        definition=definition,
    )

    assert result.valid is False

    assert any(
        "execute method"
        in error
        for error in result.errors
    )
