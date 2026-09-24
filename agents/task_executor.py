from storage.database.base import Database
from storage.models.research import ResearchTask


class TaskExecutor:
    """
    Executes research tasks while respecting task dependencies
    and persists execution state to the database.
    """

    def __init__(
        self,
        tasks: list[ResearchTask],
        database: Database,
        run_id: str,
    ):
        self.tasks = {task.task_id: task for task in tasks}
        self.database = database
        self.run_id = run_id
        self.completed_tasks: set[str] = set()

    def get_ready_tasks(self) -> list[ResearchTask]:
        """
        Return tasks whose dependencies have all been completed.
        """

        ready_tasks = []

        for task in self.tasks.values():
            if task.status != "pending":
                continue

            dependencies_completed = all(
                dependency in self.completed_tasks
                for dependency in task.dependencies
            )

            if dependencies_completed:
                ready_tasks.append(task)

        return ready_tasks

    def execute_task(self, task: ResearchTask) -> None:
        """
        Execute one task and persist its status.
        """

        print(
            f"Executing: {task.task_id} - "
            f"{task.description}"
        )

        # Mark task as running
        task.status = "running"
        self.database.save_research_task(task)

        try:
            # Real tools will be connected here later.
            print(f"Completed: {task.task_id}")

            # Mark task as completed
            task.status = "completed"
            self.database.save_research_task(task)

            self.completed_tasks.add(task.task_id)

            # Update research run
            run = self.database.get_research_run(self.run_id)

            if run is not None:
                if task.task_id not in run.completed_tasks:
                    run.completed_tasks.append(task.task_id)

                self.database.save_research_run(run)

        except Exception:
            task.status = "failed"
            self.database.save_research_task(task)

            run = self.database.get_research_run(self.run_id)

            if run is not None:
                if task.task_id not in run.failed_tasks:
                    run.failed_tasks.append(task.task_id)

                self.database.save_research_run(run)

            raise

    def run(self) -> None:
        """
        Execute all tasks in dependency order.
        """

        run = self.database.get_research_run(self.run_id)

        if run is not None:
            run.status = "running"
            self.database.save_research_run(run)

        while len(self.completed_tasks) < len(self.tasks):

            ready_tasks = self.get_ready_tasks()

            if not ready_tasks:
                if run is not None:
                    run.status = "failed"
                    self.database.save_research_run(run)

                raise RuntimeError(
                    "No executable tasks found. "
                    "The task graph may contain a dependency cycle."
                )

            for task in ready_tasks:
                self.execute_task(task)

        run = self.database.get_research_run(self.run_id)

        if run is not None:
            run.status = "completed"
            self.database.save_research_run(run)