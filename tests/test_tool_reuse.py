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
    # Create a tool definition
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

    definition = (
        builder.to_tool_definition(
            generated_tool
        )
    )

    database.save_tool(
        definition
    )

    print(
        "\nTool definition saved:"
    )

    print(
        f"- {definition.name}"
    )

    # -----------------------------------------
    # Create a fresh registry
    # -----------------------------------------

    registry = ToolRegistry(
        database=database
    )

    # -----------------------------------------
    # Search by capability
    # -----------------------------------------

    capability = "experiment_design"

    found = (
        registry.find_definition_by_capability(
            capability
        )
    )

    print(
        "\n=== CAPABILITY LOOKUP ==="
    )

    print(
        f"Requested capability: "
        f"{capability}"
    )

    if found is None:

        print(
            "No matching tool definition found."
        )

        raise RuntimeError(
            "Tool reuse lookup failed."
        )

    print(
        f"Found tool: {found.name}"
    )

    print(
        f"Status: {found.status}"
    )

    # -----------------------------------------
    # Test unknown capability
    # -----------------------------------------

    unknown = (
        registry.find_definition_by_capability(
            "completely_unknown_capability"
        )
    )

    if unknown is not None:

        raise RuntimeError(
            "Unknown capability incorrectly "
            "matched an existing tool."
        )

    print(
        "\nUnknown capability correctly "
        "returned no match."
    )

    print(
        "\nTool reuse lookup test passed!"
    )


if __name__ == "__main__":
    main()
