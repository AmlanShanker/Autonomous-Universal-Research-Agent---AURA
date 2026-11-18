from pymongo import MongoClient

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


class MongoDatabase(Database):
    def __init__(
        self,
        connection_string: str,
        database_name: str,
    ):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]

        self.research_plans = self.db["research_plans"]
        self.research_tasks = self.db["research_tasks"]
        self.research_runs = self.db["research_runs"]
        self.tools = self.db["tools"]
        self.tool_implementations = self.db["tool_implementations"]
        self.execution_results = self.db["execution_results"]
        self.research_memory = self.db["research_memory"]

    def save_research_plan(
        self,
        plan: ResearchPlan,
    ) -> None:
        document = plan.model_dump()

        self.research_plans.replace_one(
            {"question": plan.question},
            document,
            upsert=True,
        )

    def save_research_task(
        self,
        task: ResearchTask,
    ) -> None:
        document = task.model_dump()

        self.research_tasks.replace_one(
            {"task_id": task.task_id},
            document,
            upsert=True,
        )

    def save_research_run(
        self,
        run: ResearchRun,
    ) -> None:
        document = run.model_dump()

        self.research_runs.replace_one(
            {"run_id": run.run_id},
            document,
            upsert=True,
        )

    def save_tool(
        self,
        tool: ToolDefinition,
    ) -> None:
        document = tool.model_dump()

        self.tools.replace_one(
            {"tool_id": tool.tool_id},
            document,
            upsert=True,
        )

    def save_tool_implementation(
        self,
        implementation: ToolImplementation,
    ) -> None:
        document = implementation.model_dump()

        self.tool_implementations.replace_one(
            {
                "implementation_id": implementation.implementation_id
            },
            document,
            upsert=True,
        )

    def save_execution_result(
        self,
        result: ExecutionResult,
    ) -> None:
        document = result.model_dump()

        self.execution_results.replace_one(
            {
                "result_id": result.result_id
            },
            document,
            upsert=True,
        )

    def save_research_memory(
        self,
        item: ResearchMemoryItem,
    ) -> None:
        document = item.model_dump()

        self.research_memory.replace_one(
            {"memory_id": item.memory_id},
            document,
            upsert=True,
        )

    def get_research_task(
        self,
        task_id: str,
    ) -> ResearchTask | None:
        document = self.research_tasks.find_one(
            {"task_id": task_id}
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ResearchTask(**document)

    def get_research_run(
        self,
        run_id: str,
    ) -> ResearchRun | None:
        document = self.research_runs.find_one(
            {"run_id": run_id}
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ResearchRun(**document)

    def get_tool(
        self,
        tool_id: str,
    ) -> ToolDefinition | None:
        document = self.tools.find_one(
            {"tool_id": tool_id}
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ToolDefinition(**document)

    def get_tool_implementation(
        self,
        implementation_id: str,
    ) -> ToolImplementation | None:
        document = self.tool_implementations.find_one(
            {
                "implementation_id":
                implementation_id
            }
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ToolImplementation(**document)

    def get_execution_result(
        self,
        result_id: str,
    ) -> ExecutionResult | None:
        document = self.execution_results.find_one(
            {
                "result_id":
                result_id
            }
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ExecutionResult(**document)

    def get_research_memory(
        self,
        memory_id: str,
    ) -> ResearchMemoryItem | None:
        document = self.research_memory.find_one(
            {"memory_id": memory_id}
        )

        if document is None:
            return None

        document.pop("_id", None)

        return ResearchMemoryItem(**document)

    def list_research_memory_for_run(
        self,
        run_id: str,
    ) -> list[ResearchMemoryItem]:
        documents = self.research_memory.find(
            {"run_id": run_id}
        )

        results = []

        for document in documents:
            document.pop("_id", None)
            results.append(
                ResearchMemoryItem(**document)
            )

        return results

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

        for document in self.research_memory.find():
            document.pop("_id", None)

            item = ResearchMemoryItem(**document)

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

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False