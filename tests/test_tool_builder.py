from tools.builder import (
    ToolBuildRequest,
    ToolBuilder,
)


def main():

    builder = ToolBuilder()

    request = ToolBuildRequest(
        capability="experiment_design",
        reason=(
            "The research task requires an "
            "experimental design capability."
        ),
    )

    tool = builder.build(
        request
    )

    print("=== GENERATED TOOL ===")

    print(
        f"Name: {tool.name}"
    )

    print(
        f"Description: {tool.description}"
    )

    print(
        f"Purpose: {tool.purpose}"
    )

    print(
        f"Inputs: {tool.inputs}"
    )

    print(
        f"Outputs: {tool.outputs}"
    )


if __name__ == "__main__":
    main()
