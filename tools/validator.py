from dataclasses import dataclass

from storage.models.research import ToolDefinition
from tools.base import ResearchTool


@dataclass
class ToolValidationResult:
    """
    Result of validating an executable tool
    against its persistent ToolDefinition.
    """

    tool_id: str

    tool_name: str

    valid: bool

    checks: list[str]

    errors: list[str]


class ToolValidator:
    """
    Validates executable research tools against
    their persistent ToolDefinition.

    This validator checks the tool's interface and
    declared metadata.

    It does NOT:
        - execute generated code
        - install dependencies
        - modify the host system
        - trust a tool merely because it has a definition
    """

    def validate(
        self,
        tool: ResearchTool,
        definition: ToolDefinition,
    ) -> ToolValidationResult:
        """
        Validate an executable tool against a
        persistent tool definition.
        """

        checks = []
        errors = []

        # -----------------------------------------
        # Tool ID
        # -----------------------------------------

        if not definition.tool_id.strip():

            errors.append(
                "Tool definition has an empty tool_id."
            )

        else:

            checks.append(
                "Tool definition has a valid tool_id."
            )

        # -----------------------------------------
        # Tool Name
        # -----------------------------------------

        if not definition.name.strip():

            errors.append(
                "Tool definition has an empty name."
            )

        else:

            checks.append(
                "Tool definition has a valid name."
            )

        # -----------------------------------------
        # Executable Tool Name
        # -----------------------------------------

        if not tool.name.strip():

            errors.append(
                "Executable tool has an empty name."
            )

        else:

            checks.append(
                "Executable tool has a valid name."
            )

        # -----------------------------------------
        # Name Consistency
        # -----------------------------------------

        if tool.name != definition.name:

            errors.append(
                "Executable tool name does not match "
                "the ToolDefinition name."
            )

        else:

            checks.append(
                "Executable tool name matches "
                "the ToolDefinition name."
            )

        # -----------------------------------------
        # Description
        # -----------------------------------------

        if not tool.description.strip():

            errors.append(
                "Executable tool has an empty description."
            )

        else:

            checks.append(
                "Executable tool has a valid description."
            )

        if not definition.description.strip():

            errors.append(
                "Tool definition has an empty description."
            )

        else:

            checks.append(
                "Tool definition has a valid description."
            )

        # -----------------------------------------
        # Capability Consistency
        # -----------------------------------------

        if definition.capability.strip():

            if (
                definition.capability.strip().lower()
                == tool.name.strip().lower()
            ):

                checks.append(
                    "Tool capability matches the executable "
                    "tool name."
                )

        # -----------------------------------------
        # Execute Method
        # -----------------------------------------

        execute_method = getattr(
            tool,
            "execute",
            None,
        )

        if not callable(execute_method):

            errors.append(
                "Executable tool does not provide "
                "a callable execute method."
            )

        else:

            checks.append(
                "Executable tool provides a callable "
                "execute method."
            )

        # -----------------------------------------
        # Final Result
        # -----------------------------------------

        return ToolValidationResult(
            tool_id=definition.tool_id,
            tool_name=tool.name,
            valid=not errors,
            checks=checks,
            errors=errors,
        )
