from tools.base import ResearchTool
from tools.literature_search import LiteratureSearchTool


class ToolRegistry:
    """
    Central registry for all tools available to AURA.
    """

    def __init__(self):
        self.tools: dict[str, ResearchTool] = {}

        self.register(LiteratureSearchTool())

    def register(self, tool: ResearchTool) -> None:
        """
        Register a research tool.
        """

        if tool.name in self.tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self.tools[tool.name] = tool

    def get(self, tool_name: str) -> ResearchTool | None:
        """
        Retrieve a registered tool by name.
        """

        return self.tools.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """
        Check whether a tool is registered.
        """

        return tool_name in self.tools

    def list_tools(self) -> list[str]:
        """
        Return the names of all registered tools.
        """

        return list(self.tools.keys())
