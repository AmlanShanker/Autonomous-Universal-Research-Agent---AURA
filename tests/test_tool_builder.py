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

    print(
        "\n=== GENERATED TOOL SPECIFICATION ==="
    )

    print(
        f"\nName:\n{tool.name}"
    )

    print(
        f"\nDescription:\n{tool.description}"
    )

    print(
        f"\nPurpose:\n{tool.purpose}"
    )

    print(
        f"\nInputs:\n{tool.inputs}"
    )

    print(
        f"\nOutputs:\n{tool.outputs}"
    )

    print(
        f"\nDependencies:\n{tool.dependencies}"
    )

    print(
        "\nValidation Requirements:"
    )

    for requirement in (
        tool.validation_requirements
    ):

        print(
            f"- {requirement}"
        )


if __name__ == "__main__":
    main()
