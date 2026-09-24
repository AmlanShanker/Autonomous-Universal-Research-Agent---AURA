from storage.database.base import Database
from storage.models.research import (
    ToolDefinition,
    ToolImplementation,
)

from tools.base import ResearchTool
from tools.implementations import DatasetDownloadTool
from tools.literature_search import LiteratureSearchTool
from tools.validator import (
    ToolValidationResult,
    ToolValidator,
)


class ToolRegistry:
    """
    Central registry for all tools available to AURA.

    The registry maintains three separate collections:

    1. Executable tools
       - Actual ResearchTool implementations.

    2. Tool definitions
       - Persistent specifications stored in MongoDB.
       - These describe capabilities that AURA knows about.
       - A ToolDefinition is NOT executable code.

    3. Validated tools
       - Executable implementations that have successfully
         passed validation against their ToolDefinition.
       - Only validated tools are considered executable by
         the research execution layer.

    Tool implementations are tracked separately from
    ToolDefinitions so one capability can have multiple
    implementations and versions.
    """

    def __init__(
        self,
        database: Database | None = None,
    ):
        self.tools: dict[str, ResearchTool] = {}

        self.definitions: dict[
            str,
            ToolDefinition,
        ] = {}

        self.implementations: dict[
            str,
            ToolImplementation,
        ] = {}

        self.validated_tools: dict[
            str,
            ResearchTool,
        ] = {}

        self.database = database

        self.validator = ToolValidator()

        self.register(
            LiteratureSearchTool()
        )

        self.register(
            DatasetDownloadTool()
        )

        if self.database is not None:
            self.load_definitions()
            self.load_implementations()

    def register(
        self,
        tool: ResearchTool,
    ) -> None:
        if tool.name in self.tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self.tools[tool.name] = tool

    def register_definition(
        self,
        definition: ToolDefinition,
    ) -> None:
        if definition.tool_id in self.definitions:
            raise ValueError(
                "Tool definition "
                f"'{definition.tool_id}' "
                "is already registered."
            )

        self.definitions[
            definition.tool_id
        ] = definition

    def register_implementation(
        self,
        implementation: ToolImplementation,
    ) -> None:
        if (
            implementation.implementation_id
            in self.implementations
        ):
            raise ValueError(
                "Tool implementation "
                f"'{implementation.implementation_id}' "
                "is already registered."
            )

        if (
            implementation.tool_id
            not in self.definitions
        ):
            raise ValueError(
                "Cannot register tool implementation "
                f"'{implementation.implementation_id}' "
                "because tool definition "
                f"'{implementation.tool_id}' "
                "does not exist."
            )

        self.implementations[
            implementation.implementation_id
        ] = implementation

        definition = self.definitions[
            implementation.tool_id
        ]

        if (
            implementation.implementation_id
            not in definition.implementation_ids
        ):
            definition.implementation_ids.append(
                implementation.implementation_id
            )

        if self.database is not None:
            self.database.save_tool_implementation(
                implementation
            )

            self.database.save_tool(
                definition
            )

    def load_definitions(
        self,
    ) -> None:
        if self.database is None:
            return

        collection = self.database.tools

        if hasattr(collection, "find"):
            documents = collection.find()

            for document in documents:
                document.pop(
                    "_id",
                    None,
                )

                definition = ToolDefinition(
                    **document
                )

                self.definitions[
                    definition.tool_id
                ] = definition

            return

        if isinstance(collection, dict):
            for definition in collection.values():
                self.definitions[
                    definition.tool_id
                ] = definition

    def load_implementations(
        self,
    ) -> None:
        if self.database is None:
            return

        collection = (
            self.database.tool_implementations
        )

        if hasattr(collection, "find"):
            documents = collection.find()

            for document in documents:
                document.pop(
                    "_id",
                    None,
                )

                implementation = ToolImplementation(
                    **document
                )

                self.implementations[
                    implementation.implementation_id
                ] = implementation

            return

        if isinstance(collection, dict):
            for implementation in collection.values():
                self.implementations[
                    implementation.implementation_id
                ] = implementation

    def validate_tool(
        self,
        tool_name: str,
    ) -> ToolValidationResult:
        tool = self.get(tool_name)

        if tool is None:
            raise ValueError(
                f"Executable tool '{tool_name}' "
                "is not registered."
            )

        definition = self.get_definition(
            tool_name
        )

        if definition is None:
            raise ValueError(
                f"No ToolDefinition exists for "
                f"executable tool '{tool_name}'."
            )

        result = self.validator.validate(
            tool=tool,
            definition=definition,
        )

        if result.valid:
            definition.status = "validated"

            self.validated_tools[
                tool_name
            ] = tool

            implementation = (
                self._find_implementation_for_tool(
                    tool_name
                )
            )

            if implementation is not None:
                implementation.validation_status = (
                    "validated"
                )

                implementation.status = "active"

                if self.database is not None:
                    self.database.save_tool_implementation(
                        implementation
                    )

        else:
            definition.status = "failed"

            self.validated_tools.pop(
                tool_name,
                None,
            )

            implementation = (
                self._find_implementation_for_tool(
                    tool_name
                )
            )

            if implementation is not None:
                implementation.validation_status = (
                    "failed"
                )

                implementation.status = "inactive"

                if self.database is not None:
                    self.database.save_tool_implementation(
                        implementation
                    )

        if self.database is not None:
            self.database.save_tool(
                definition
            )

        return result

    def get(
        self,
        tool_name: str,
    ) -> ResearchTool | None:
        return self.tools.get(
            tool_name
        )

    def get_validated(
        self,
        tool_name: str,
    ) -> ResearchTool | None:
        return self.validated_tools.get(
            tool_name
        )

    def get_definition(
        self,
        tool_id: str,
    ) -> ToolDefinition | None:
        return self.definitions.get(
            tool_id
        )

    def get_implementation(
        self,
        implementation_id: str,
    ) -> ToolImplementation | None:
        return self.implementations.get(
            implementation_id
        )

    def get_implementations_for_tool(
        self,
        tool_id: str,
    ) -> list[ToolImplementation]:
        definition = self.get_definition(
            tool_id
        )

        if definition is None:
            return []

        implementations = []

        for implementation_id in (
            definition.implementation_ids
        ):
            implementation = (
                self.get_implementation(
                    implementation_id
                )
            )

            if implementation is not None:
                implementations.append(
                    implementation
                )

        return implementations

    def get_usable_implementations(
        self,
        tool_id: str,
    ) -> list[ToolImplementation]:
        """
        Return implementations that are currently
        eligible for execution.

        An implementation is usable only when:

        - validation_status == "validated"
        - status == "active"
        """

        implementations = (
            self.get_implementations_for_tool(
                tool_id
            )
        )

        return [
            implementation
            for implementation in implementations
            if (
                implementation.validation_status
                == "validated"
                and implementation.status
                == "active"
            )
        ]

    def select_implementation(
        self,
        tool_id: str,
    ) -> ToolImplementation | None:
        """
        Select one usable implementation.

        Selection is intentionally deterministic for now.

        AURA will later be able to select implementations
        using reliability, performance, cost, dependencies,
        environment compatibility, and historical results.
        """

        usable = (
            self.get_usable_implementations(
                tool_id
            )
        )

        if not usable:
            return None

        usable.sort(
            key=lambda implementation: (
                implementation.version,
                implementation.implementation_id,
            ),
            reverse=True,
        )

        return usable[0]

    def find_definition_by_capability(
        self,
        capability: str,
    ) -> ToolDefinition | None:
        capability = (
            capability.strip().lower()
        )

        if not capability:
            return None

        for definition in (
            self.definitions.values()
        ):
            definition_capability = (
                definition.capability
                .strip()
                .lower()
            )

            if (
                definition_capability
                == capability
            ):
                return definition

        return None

    def has(
        self,
        tool_name: str,
    ) -> bool:
        return tool_name in self.tools

    def has_validated(
        self,
        tool_name: str,
    ) -> bool:
        return (
            tool_name
            in self.validated_tools
        )

    def has_definition(
        self,
        tool_id: str,
    ) -> bool:
        return (
            tool_id
            in self.definitions
        )

    def has_implementation(
        self,
        implementation_id: str,
    ) -> bool:
        return (
            implementation_id
            in self.implementations
        )

    def list_tools(
        self,
    ) -> list[str]:
        return list(
            self.tools.keys()
        )

    def list_validated_tools(
        self,
    ) -> list[str]:
        return list(
            self.validated_tools.keys()
        )

    def list_definitions(
        self,
    ) -> list[str]:
        return list(
            self.definitions.keys()
        )

    def list_implementations(
        self,
    ) -> list[str]:
        return list(
            self.implementations.keys()
        )

    def _find_implementation_for_tool(
        self,
        tool_id: str,
    ) -> ToolImplementation | None:
        definition = self.get_definition(
            tool_id
        )

        if definition is None:
            return None

        for implementation_id in (
            definition.implementation_ids
        ):
            implementation = (
                self.get_implementation(
                    implementation_id
                )
            )

            if implementation is not None:
                if (
                    implementation.status
                    in {
                        "active",
                        "draft",
                    }
                ):
                    return implementation

        return None
