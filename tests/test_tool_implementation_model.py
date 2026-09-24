from storage.models.research import (
    ToolDefinition,
    ToolImplementation,
)


def test_tool_implementation_defaults():
    implementation = ToolImplementation(
        implementation_id="dataset_download_builtin_v1",
        tool_id="dataset_download",
        implementation_type="builtin",
    )

    assert (
        implementation.implementation_id
        == "dataset_download_builtin_v1"
    )

    assert (
        implementation.tool_id
        == "dataset_download"
    )

    assert implementation.version == "1.0.0"

    assert (
        implementation.implementation_type
        == "builtin"
    )

    assert (
        implementation.validation_status
        == "unvalidated"
    )

    assert implementation.status == "draft"

    assert implementation.dependencies == []

    assert implementation.metadata == {}


def test_tool_implementation_can_describe_generated_source():
    implementation = ToolImplementation(
        implementation_id="experiment_designer_generated_v1",
        tool_id="experiment_designer",
        version="1.0.0",
        implementation_type="generated",
        source_reference=(
            "storage/tool_implementations/"
            "experiment_designer/1.0.0/"
        ),
        entrypoint="main.py:execute",
        dependencies=[
            "numpy",
        ],
        validation_status="validated",
        status="active",
        description=(
            "Generated implementation of an experiment "
            "design capability."
        ),
        metadata={
            "generator": "AURA Tool Builder",
            "generated_by": "tool_builder",
        },
    )

    assert (
        implementation.implementation_type
        == "generated"
    )

    assert (
        implementation.validation_status
        == "validated"
    )

    assert implementation.status == "active"

    assert (
        implementation.entrypoint
        == "main.py:execute"
    )

    assert implementation.dependencies == [
        "numpy"
    ]

    assert (
        implementation.metadata["generator"]
        == "AURA Tool Builder"
    )


def test_tool_definition_defaults_to_no_implementations():
    tool = ToolDefinition(
        tool_id="experiment_designer",
        name="experiment_designer",
        description=(
            "Design standardized research experiments."
        ),
    )

    assert tool.implementation_ids == []


def test_tool_definition_can_reference_implementations():
    tool = ToolDefinition(
        tool_id="experiment_designer",
        name="experiment_designer",
        description=(
            "Design standardized research experiments."
        ),
        implementation_ids=[
            "experiment_designer_builtin_v1",
            "experiment_designer_generated_v1",
        ],
    )

    assert tool.implementation_ids == [
        "experiment_designer_builtin_v1",
        "experiment_designer_generated_v1",
    ]
