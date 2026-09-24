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

    The registry maintains four separate collections:

    1. Executable tools
       - Actual ResearchTool implementations.

    2. Tool definitions
       - Persistent specifications stored in MongoDB.
       - These describe capabilities known to AURA.
       - A ToolDefinition is NOT executable code.

    3. Tool implementations
       - Persistent metadata describing concrete
         implementations of a ToolDefinition.

    4. Validated tools
       - Executable ResearchTool instances that have
         successfully passed validation.

    Generated implementations are represented as metadata
    only and are not executable by AURA yet.
    """

    def __init__(
        self,
        database: Database | None = None,
    ):
        self.tools: dict[str, ResearchTool] = {}
        self.definitions: dict[str, ToolDefinition] = {}
        self.implementations: dict[str, ToolImplementation] = {}
        self.validated_tools: dict[str, ResearchTool] = {}

        self.database = database
        self.validator = ToolValidator()

        self.register(LiteratureSearchTool())
        self.register(DatasetDownloadTool())

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

        self.definitions[definition.tool_id] = definition

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

        if implementation.tool_id not in self.definitions:
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
            self.database.save_tool(definition)

    def load_definitions(self) -> None:
        if self.database is None:
            return

        collection = self.database.tools

        if hasattr(collection, "find"):
            documents = collection.find()

            for document in documents:
                document.pop("_id", None)

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

    def load_implementations(self) -> None:
        if self.database is None:
            return

        collection = (
            self.database.tool_implementations
        )

        if hasattr(collection, "find"):
            documents = collection.find()

            for document in documents:
                document.pop("_id", None)

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
        """
        Validate an executable tool.

        If a builtin implementation already exists for
        this tool, it is reused.

        Otherwise, AURA creates a new builtin implementation
        record for the registered executable tool.

        Successful validation marks the implementation:

            validation_status = "validated"
            status = "active"

        Failed validation marks it:

            validation_status = "failed"
            status = "inactive"
        """

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
                "No ToolDefinition exists for "
                f"executable tool '{tool_name}'."
            )

        result = self.validator.validate(
            tool=tool,
            definition=definition,
        )

        implementation = (
            self._get_or_create_builtin_implementation(
                tool_name
            )
        )

        if result.valid:
            definition.status = "validated"

            self.validated_tools[
                tool_name
            ] = tool

            implementation.validation_status = (
                "validated"
            )

            implementation.status = "active"
        else:
            definition.status = "failed"

            self.validated_tools.pop(
                tool_name,
                None,
            )

            implementation.validation_status = (
                "failed"
            )

            implementation.status = "inactive"

        if self.database is not None:
            self.database.save_tool(
                definition
            )

            self.database.save_tool_implementation(
                implementation
            )

        return result

    def _get_or_create_builtin_implementation(
        self,
        tool_name: str,
    ) -> ToolImplementation:
        """
        Return the existing builtin implementation for
        a tool, or create one if none exists.

        This prevents repeated validation from creating
        duplicate implementation records.
        """

        existing = (
            self._find_builtin_implementation_for_tool(
                tool_name
            )
        )

        if existing is not None:
            return existing

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
                "No ToolDefinition exists for "
                f"tool '{tool_name}'."
            )

        implementation = ToolImplementation(
            implementation_id=(
                f"{tool_name}_builtin_v1"
            ),
            tool_id=tool_name,
            version="1.0.0",
            implementation_type="builtin",
            source_reference=(
                "registered ResearchTool implementation"
            ),
            entrypoint=(
                f"{tool.__class__.__name__}.execute"
            ),
            dependencies=[],
            validation_status="unvalidated",
            status="draft",
            description=tool.description,
            metadata={
                "class": tool.__class__.__name__,
            },
        )

        self.implementations[
            implementation.implementation_id
        ] = implementation

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

        return implementation

    def _find_builtin_implementation_for_tool(
        self,
        tool_id: str,
    ) -> ToolImplementation | None:
        """
        Find an existing builtin implementation.

        Selection is deterministic based on version and
        implementation ID.
        """

        implementations = (
            self.get_implementations_for_tool(
                tool_id
            )
        )

        builtin = [
            implementation
            for implementation in implementations
            if implementation.implementation_type
            == "builtin"
        ]

        if not builtin:
            return None

        builtin.sort(
            key=lambda implementation: (
                implementation.version,
                implementation.implementation_id,
            ),
            reverse=True,
        )

        return builtin[0]

    def resolve_implementation(
        self,
        implementation: ToolImplementation,
    ) -> ResearchTool | None:
        """
        Resolve a ToolImplementation to an executable
        ResearchTool.

        Only builtin implementations can currently
        be resolved.

        Generated implementations are intentionally
        NOT executable yet because AURA does not
        currently execute generated source code.

        Resolution requires:

        - validation_status == "validated"
        - status == "active"
        - implementation_type == "builtin"
        - corresponding executable tool exists
        - implementation belongs to the tool definition
        """

        if implementation.validation_status != "validated":
            return None

        if implementation.status != "active":
            return None

        if implementation.implementation_type != "builtin":
            return None

        definition = self.get_definition(
            implementation.tool_id
        )

        if definition is None:
            return None

        if (
            implementation.implementation_id
            not in definition.implementation_ids
        ):
            return None

        tool = self.get(
            implementation.tool_id
        )

        if tool is None:
            return None

        return tool

    def get_executable_implementation(
        self,
        tool_id: str,
    ) -> tuple[
        ToolImplementation,
        ResearchTool,
    ] | None:
        """
        Select and resolve the executable implementation
        for a tool.

        Returns:

            (ToolImplementation, ResearchTool)

        when a usable executable implementation exists.

        Returns None when:

        - no usable implementation exists
        - the selected implementation cannot be resolved
        - the selected implementation is generated
        """

        implementation = (
            self.select_implementation(
                tool_id
            )
        )

        if implementation is None:
            return None

        tool = self.resolve_implementation(
            implementation
        )

        if tool is None:
            return None

        return implementation, tool

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
        Return implementations currently eligible
        for execution.

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

        Selection is deterministic for now.

        AURA can later consider:

        - reliability
        - performance
        - cost
        - dependencies
        - environment compatibility
        - historical execution results
        """

        usable = self.get_usable_implementations(
            tool_id
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
        capability = capability.strip().lower()

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

    def list_tools(self) -> list[str]:
        return list(
            self.tools.keys()
        )

    def list_validated_tools(self) -> list[str]:
        return list(
            self.validated_tools.keys()
        )

    def list_definitions(self) -> list[str]:
        return list(
            self.definitions.keys()
        )

    def list_implementations(self) -> list[str]:
        return list(
            self.implementations.keys()
        )
