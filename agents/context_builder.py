from storage.models.context import ResearchContextItem
from storage.models.research import ResearchTask


class TaskContextBuilder:
    """
    Build structured context inputs for a research task.

    The builder converts internal ResearchContext items into a
    tool-independent structure that can be passed to downstream
    task execution.
    """

    def build(
        self,
        task: ResearchTask,
        context_items: list[ResearchContextItem],
    ) -> dict:
        """
        Build the context payload available to a task.

        Only evidence produced by the task's dependencies should be
        supplied to the task.
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
            ]
        }