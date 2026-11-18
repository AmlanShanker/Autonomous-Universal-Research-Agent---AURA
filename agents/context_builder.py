from memory.research_memory import ResearchMemoryItem
from storage.models.context import ResearchContextItem
from storage.models.research import ResearchTask


class TaskContextBuilder:
    """
    Build structured context inputs for a research task.

    The builder combines evidence from completed task dependencies
    with reusable findings retrieved from earlier research runs.
    """

    def build(
        self,
        task: ResearchTask,
        context_items: list[ResearchContextItem],
        prior_findings: list[ResearchMemoryItem] | None = None,
    ) -> dict:
        """
        Build the context payload available to a task.

        Dependency evidence is restricted to the task's declared
        dependencies. Prior findings are supplied separately so
        consumers can distinguish current-run evidence from
        historical research memory.
        """

        dependency_ids = set(task.dependencies)

        relevant_items = [
            item
            for item in context_items
            if item.task_id in dependency_ids
        ]

        return {
            "dependencies": [
                {
                    "task_id": item.task_id,
                    "result_id": item.result_id,
                    "tool_id": item.tool_id,
                    "status": item.status,
                    "output": item.output,
                    "metadata": item.metadata,
                }
                for item in relevant_items
            ],
            "prior_findings": [
                {
                    "memory_id": item.memory_id,
                    "source_run_id": item.run_id,
                    "source_task_id": item.task_id,
                    "research_question": item.research_question,
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "tags": item.tags,
                    "metadata": item.metadata,
                    "confidence": item.confidence,
                    "source_result_id": item.source_result_id,
                    "source_tool_id": item.source_tool_id,
                }
                for item in (prior_findings or [])
            ],
        }
