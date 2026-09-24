from storage.database.base import Database
from storage.models.research import ToolDefinition

from tools.base import ResearchTool
from tools.implementations import DatasetDownloadTool
from tools.literature_search import LiteratureSearchTool


class ToolRegistry:
    """
    Central registry for all tools available to AURA.

    The registry maintains two separate collections:

    1. Executable tools
       - Actual ResearchTool implementations.
       - Safe, registered capabilities that AURA can execute.

    2. Tool definitions
       - Persistent specifications stored in MongoDB.
       - These describe tools that AURA knows about.
       - A ToolDefinition is NOT executable code.
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

        self.database = database

        # -----------------------------------------
        # Register Built-in Executable Tools
        # -----------------------------------------

        self.register(
            LiteratureSearchTool()
        )

        self.register(
            DatasetDownloadTool()
        )

        # -----------------------------------------
        # Load Persisted Tool Definitions
        # -----------------------------------------

        if self.database is not None:
            self.load_definitions()

    def register(
        self,
        tool: ResearchTool,
    ) -> None:
        """
        Register an executable research tool.
        """

        if tool.name in self.tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self.tools[
            tool.name
        ] = tool

    def register_definition(
        self,
        definition: ToolDefinition,
    ) -> None:
        """
        Register a persistent tool definition.

        This does NOT make the tool executable.
        """

        if (
            definition.tool_id
            in self.definitions
        ):
            raise ValueError(
                "Tool definition "
                f"'{definition.tool_id}' "
                "is already registered."
            )

        self.definitions[
            definition.tool_id
        ] = definition

    def load_definitions(self) -> None:
        """
        Load persisted tool definitions from the database.

        Only definitions are loaded.

        No generated code is executed.
        """

        if self.database is None:
            return

        collection = self.database.tools

        for document in collection.find():

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

    def get(
        self,
        tool_name: str,
    ) -> ResearchTool | None:
        """
        Retrieve an executable tool by name.
        """

        return self.tools.get(
            tool_name
        )

    def get_definition(
        self,
        tool_id: str,
    ) -> ToolDefinition | None:
        """
        Retrieve a persistent tool definition.
        """

        return self.definitions.get(
            tool_id
        )

    def find_definition_by_capability(
        self,
        capability: str,
    ) -> ToolDefinition | None:
        """
        Find a persisted tool definition by capability.

        Capability matching is exact and case-insensitive.

        The capability is kept separate from the tool name
        because a generated tool may have a human-friendly
        name such as 'ExperimentDesigner' while the requested
        capability is 'experiment_design'.
        """

        capability = capability.strip().lower()

        if not capability:
            return None

        for definition in self.definitions.values():

            definition_capability = (
                definition.capability.strip().lower()
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
        """
        Check whether an executable tool
        is registered.
        """

        return tool_name in self.tools

    def has_definition(
        self,
        tool_id: str,
    ) -> bool:
        """
        Check whether a persistent tool
        definition is known.
        """

        return tool_id in self.definitions

    def list_tools(
        self,
    ) -> list[str]:
        """
        Return the names of all executable tools.
        """

        return list(
            self.tools.keys()
        )

    def list_definitions(
        self,
    ) -> list[str]:
        """
        Return the IDs of all known tool definitions.
        """

        return list(
            self.definitions.keys()
        )
