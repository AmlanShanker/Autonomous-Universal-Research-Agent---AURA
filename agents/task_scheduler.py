from storage.models.research import ResearchTask


class TaskScheduler:
    """
    Determine which research tasks are ready to execute.

    The scheduler is deterministic. It does not use an LLM.

    A task is ready when:
    - its status is pending
    - all of its dependencies exist
    - all dependencies have completed successfully
    """

    def __init__(
        self,
        tasks: list[ResearchTask],
    ):
        self.tasks = {
            task.task_id: task
            for task in tasks
        }

    def get_ready_tasks(self) -> list[ResearchTask]:
        """
        Return all tasks that are currently ready to execute.
        """

        ready_tasks = []

        for task in self.tasks.values():
            if task.status != "pending":
                continue

            if self._dependencies_completed(task):
                ready_tasks.append(task)

        return ready_tasks

    def _dependencies_completed(
        self,
        task: ResearchTask,
    ) -> bool:
        """
        Check whether every dependency of a task has completed.
        """

        for dependency_id in task.dependencies:
            dependency = self.tasks.get(
                dependency_id
            )

            if dependency is None:
                raise ValueError(
                    "Task "
                    f"'{task.task_id}' depends on unknown "
                    f"task '{dependency_id}'."
                )

            if dependency.status != "completed":
                return False

        return True

    def has_pending_tasks(self) -> bool:
        """
        Return True if at least one task is still pending.
        """

        return any(
            task.status == "pending"
            for task in self.tasks.values()
        )

    def has_ready_tasks(self) -> bool:
        """
        Return True if at least one task is ready.
        """

        return bool(
            self.get_ready_tasks()
        )