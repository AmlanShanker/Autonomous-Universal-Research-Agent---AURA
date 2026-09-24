from config.settings import MONGO_URL

from storage.database.mongo import MongoDatabase

from tools.builder import (
    ToolBuildRequest,
    ToolBuilder,
)

from tools.registry import ToolRegistry


def main():
    # -----------------------------------------
    # Database
    # -----------------------------------------

    database = MongoDatabase(
        connection_string=MONGO_URL,
        database_name="aura",
    )

    # -----------------------------------------
    # Build a tool specification
    # -----------------------------------------

    builder = ToolBuilder()

    request = ToolBuildRequest(
        capability="experiment_design",
        reason=(
            "AURA requires a capability for "
            "designing reproducible experiments."
        ),
    )

    generated_tool = builder.build(
        request
    )

    # -----------------------------------------
    # Convert to persistent definition
    # -----------------------------------------

    definition = (
        builder.to_tool_definition(
            generated_tool
        )
    )

    # -----------------------------------------
    # Save definition
    # -----------------------------------------

    database.save_tool(
        definition
    )

    print(
        "\nTool definition saved."
    )

    # -----------------------------------------
    # Create a NEW registry
    # -----------------------------------------

    registry = ToolRegistry(
        database=database
    )

    print(
        "\n=== EXECUTABLE TOOLS ==="
    )

    for tool_name in registry.list_tools():
        print(
            f"- {tool_name}"
        )

    print(
        "\n=== PERSISTED DEFINITIONS ==="
    )

    for tool_id in registry.list_definitions():
        print(
            f"- {tool_id}"
        )

    # -----------------------------------------
    # Verify persisted definition
    # -----------------------------------------

    retrieved = registry.get_definition(
        definition.tool_id
    )

    if retrieved is None:
        raise RuntimeError(
            "Persisted tool definition "
            "was not loaded by the registry."
        )

    print(
        "\n=== RETRIEVED DEFINITION ==="
    )

    print(
        f"Name: {retrieved.name}"
    )

    print(
        f"Purpose: {retrieved.purpose}"
    )

    print(
        f"Status: {retrieved.status}"
    )

    # -----------------------------------------
    # Verify executable separation
    # -----------------------------------------

    if registry.has(
        definition.name
    ):
        raise RuntimeError(
            "Generated tool definition was "
            "incorrectly registered as executable."
        )

    print(
        "\nGenerated tool is correctly "
        "kept separate from executable tools."
    )

    print(
        "\nTool Registry persistence test passed!"
    )


if __name__ == "__main__":
    main()
