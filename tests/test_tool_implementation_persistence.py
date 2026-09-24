from storage.database.memory import InMemoryDatabase
from storage.models.research import ToolImplementation


def test_tool_implementation_persistence():
    database = InMemoryDatabase()

    implementation = ToolImplementation(
        implementation_id="dataset_download_builtin_v1",
        tool_id="dataset_download",
        version="1.0.0",
        implementation_type="builtin",
        source_reference=(
            "tools/implementations.py"
        ),
        entrypoint="DatasetDownloadTool.execute",
        dependencies=[
            "requests",
        ],
        validation_status="validated",
        status="active",
        description=(
            "Built-in dataset download implementation."
        ),
        metadata={
            "source": "AURA built-in implementation",
        },
    )

    database.save_tool_implementation(
        implementation
    )

    retrieved = database.get_tool_implementation(
        "dataset_download_builtin_v1"
    )

    assert retrieved is not None

    assert (
        retrieved.implementation_id
        == implementation.implementation_id
    )

    assert (
        retrieved.tool_id
        == "dataset_download"
    )

    assert (
        retrieved.implementation_type
        == "builtin"
    )

    assert (
        retrieved.validation_status
        == "validated"
    )

    assert retrieved.status == "active"

    assert retrieved.dependencies == [
        "requests"
    ]

    assert (
        retrieved.metadata["source"]
        == "AURA built-in implementation"
    )


def test_missing_tool_implementation_returns_none():
    database = InMemoryDatabase()

    result = database.get_tool_implementation(
        "does_not_exist"
    )

    assert result is None
