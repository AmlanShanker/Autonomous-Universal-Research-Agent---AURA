from memory.research_memory import ResearchMemoryItem

from storage.database.base import Database
from storage.models.execution import ExecutionResult
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
    ToolImplementation,
)


class InMemoryDatabase(Database):

    def __init__(self):
        self.research_plans: dict[
            str,
            ResearchPlan,
        ] = {}

        self.research_tasks: dict[
            str,
            ResearchTask,
        ] = {}

        self.research_runs: dict[
            str,
            ResearchRun,
        ] = {}

        self.tools: dict[
            str,
            ToolDefinition,
        ] = {}

        self.tool_implementations: dict[
            str,
            ToolImplementation,
        ] = {}

        self.execution_results: dict[
            str,
            ExecutionResult,
        ] = {}

        self.research_memory: dict[
            str,
            ResearchMemoryItem,
        ] = {}

    def save_research_plan(
        self,
        plan: ResearchPlan,
    ) -> None:
        self.research_plans[
            plan.question
        ] = plan

    def save_research_task(
        self,
        task: ResearchTask,
    ) -> None:
        self.research_tasks[
            task.task_id
        ] = task

    def save_research_run(
        self,
        run: ResearchRun,
    ) -> None:
        self.research_runs[
            run.run_id
        ] = run

    def save_tool(
        self,
        tool: ToolDefinition,
    ) -> None:
        self.tools[
            tool.tool_id
        ] = tool

    def save_tool_implementation(
        self,
        implementation: ToolImplementation,
    ) -> None:
        self.tool_implementations[
            implementation.implementation_id
        ] = implementation

    def save_execution_result(
        self,
        result: ExecutionResult,
    ) -> None:
        self.execution_results[
            result.result_id
        ] = result

    def save_research_memory(
        self,
        item: ResearchMemoryItem,
    ) -> None:
        if item.memory_id in self.research_memory:
            raise ValueError(
                "Research memory item "
                f"'{item.memory_id}' already exists."
            )

        self.research_memory[
            item.memory_id
        ] = item

    def get_research_task(
        self,
        task_id: str,
    ) -> ResearchTask | None:
        return self.research_tasks.get(
            task_id
        )

    def get_research_run(
        self,
        run_id: str,
    ) -> ResearchRun | None:
        return self.research_runs.get(
            run_id
        )

    def get_tool(
        self,
        tool_id: str,
    ) -> ToolDefinition | None:
        return self.tools.get(
            tool_id
        )

    def get_tool_implementation(
        self,
        implementation_id: str,
    ) -> ToolImplementation | None:
        return self.tool_implementations.get(
            implementation_id
        )

    def get_execution_result(
        self,
        result_id: str,
    ) -> ExecutionResult | None:
        return self.execution_results.get(
            result_id
        )

    def get_research_memory(
        self,
        memory_id: str,
    ) -> ResearchMemoryItem | None:
        return self.research_memory.get(
            memory_id
        )

    def list_research_memory_for_run(
        self,
        run_id: str,
    ) -> list[ResearchMemoryItem]:
        return [
            item
            for item in self.research_memory.values()
            if item.run_id == run_id
        ]

    def search_research_memory(
        self,
        query: str,
    ) -> list[ResearchMemoryItem]:
        query_terms = {
            term.lower()
            for term in query.split()
            if term.strip()
        }

        if not query_terms:
            return []

        results = []

        for item in self.research_memory.values():
            searchable_text = (
                f"{item.research_question} "
                f"{item.content} "
                f"{' '.join(item.tags)}"
            ).lower()

            if any(
                term in searchable_text
                for term in query_terms
            ):
                results.append(item)

        return results