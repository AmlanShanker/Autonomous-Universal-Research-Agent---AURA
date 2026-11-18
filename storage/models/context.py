from pydantic import BaseModel, Field


class ResearchContextItem(BaseModel):
    """
    A piece of evidence produced by a completed research task.

    Context items provide structured information that later tasks
    can consume without directly depending on another task's
    implementation details.
    """

    result_id: str
    task_id: str
    tool_id: str
    status: str
    output: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class ResearchContext(BaseModel):
    """
    Evidence accumulated during a research run.

    The context is intentionally separate from raw ExecutionResult
    persistence. ExecutionResult represents what happened during one
    tool execution; ResearchContext represents the evidence currently
    available to the research process.
    """

    run_id: str
    items: list[ResearchContextItem] = Field(default_factory=list)

    def add_result(
        self,
        result_id: str,
        task_id: str,
        tool_id: str,
        status: str,
        output: dict,
        metadata: dict | None = None,
    ) -> None:
        """
        Add an execution result to the research context.
        """

        self.items.append(
            ResearchContextItem(
                result_id=result_id,
                task_id=task_id,
                tool_id=tool_id,
                status=status,
                output=output,
                metadata=metadata or {},
            )
        )

    def get_results_for_task(
        self,
        task_id: str,
    ) -> list[ResearchContextItem]:
        """
        Return context items produced by a specific task.
        """

        return [
            item
            for item in self.items
            if item.task_id == task_id
        ]

    def get_dependency_context(
        self,
        task: object,
    ) -> list[ResearchContextItem]:
        """
        Return evidence produced by the dependencies of a task.
        """

        dependency_ids = set(task.dependencies)

        return [
            item
            for item in self.items
            if item.task_id in dependency_ids
        ]