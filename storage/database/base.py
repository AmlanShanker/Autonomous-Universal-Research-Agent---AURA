from abc import ABC, abstractmethod

from memory.research_memory import ResearchMemoryItem
from storage.models.context import ResearchContext
from storage.models.execution import ExecutionResult
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
    ToolImplementation,
)


class Database(ABC):

    @abstractmethod
    def save_research_plan(self, plan: ResearchPlan) -> None:
        pass

    @abstractmethod
    def save_research_task(self, task: ResearchTask) -> None:
        pass

    @abstractmethod
    def save_research_run(self, run: ResearchRun) -> None:
        pass

    @abstractmethod
    def save_tool(self, tool: ToolDefinition) -> None:
        pass

    @abstractmethod
    def save_tool_implementation(self, implementation: ToolImplementation) -> None:
        pass

    @abstractmethod
    def save_execution_result(self, result: ExecutionResult) -> None:
        pass

    @abstractmethod
    def save_research_memory(self, item: ResearchMemoryItem) -> None:
        pass

    @abstractmethod
    def update_research_memory(self, item: ResearchMemoryItem) -> None:
        pass

    @abstractmethod
    def delete_research_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def list_research_memory(self) -> list[ResearchMemoryItem]:
        pass

    @abstractmethod
    def get_research_task(self, task_id: str) -> ResearchTask | None:
        pass

    @abstractmethod
    def get_research_run(self, run_id: str) -> ResearchRun | None:
        pass

    @abstractmethod
    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        pass

    @abstractmethod
    def get_tool_implementation(
        self, implementation_id: str
    ) -> ToolImplementation | None:
        pass

    @abstractmethod
    def get_execution_result(self, result_id: str) -> ExecutionResult | None:
        pass

    @abstractmethod
    def get_research_memory(self, memory_id: str) -> ResearchMemoryItem | None:
        pass

    @abstractmethod
    def list_research_memory_for_run(
        self, run_id: str
    ) -> list[ResearchMemoryItem]:
        pass

    @abstractmethod
    def search_research_memory(
        self, query: str
    ) -> list[ResearchMemoryItem]:
        pass
