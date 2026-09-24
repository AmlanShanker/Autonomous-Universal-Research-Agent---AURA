from agents.llm_client import LLMClient

from tools.builder import (
    ToolBuildRequest,
    ToolBuilder,
)

from storage.database.mongo import MongoDatabase

from config.settings import MONGO_URL


def main():

    # -----------------------------------------
    # Database
    # -----------------------------------------

    database = MongoDatabase(
        connection_string=MONGO_URL,
        database_name="aura",
    )

    # -----------------------------------------
    # Tool Builder
    # -----------------------------------------

    builder = ToolBuilder(
        llm_client=LLMClient(),
    )

    # -----------------------------------------
    # Generate Tool Specification
    # -----------------------------------------

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

    print(
        "\n=== GENERATED TOOL ==="
    )

    print(
        generated_tool.name
    )

    # -----------------------------------------
    # Convert to Persistent Definition
    # -----------------------------------------

    tool_definition = (
        builder.to_tool_definition(
            generated_tool
        )
    )

    print(
        "\n=== TOOL DEFINITION ==="
    )

    print(
        tool_definition.model_dump_json(
            indent=2
        )
    )

    # -----------------------------------------
    # Save to MongoDB
    # -----------------------------------------

    database.save_tool(
        tool_definition
    )

    print(
        "\nTool saved to MongoDB."
    )

    # -----------------------------------------
    # Retrieve from MongoDB
    # -----------------------------------------

    retrieved_tool = database.get_tool(
        tool_definition.tool_id
    )

    if retrieved_tool is None:

        raise RuntimeError(
            "Tool was not found in MongoDB."
        )

    print(
        "\n=== RETRIEVED TOOL ==="
    )

    print(
        retrieved_tool.model_dump_json(
            indent=2
        )
    )

    # -----------------------------------------
    # Verify
    # -----------------------------------------

    assert (
        retrieved_tool.tool_id
        == tool_definition.tool_id
    )

    assert (
        retrieved_tool.name
        == tool_definition.name
    )

    assert (
        retrieved_tool.description
        == tool_definition.description
    )

    assert (
        retrieved_tool.dependencies
        == tool_definition.dependencies
    )

    print(
        "\nTool persistence test passed!"
    )


if __name__ == "__main__":
    main()
