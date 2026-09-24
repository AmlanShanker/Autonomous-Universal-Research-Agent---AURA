from agents.capability_gap_detector import CapabilityGapDetector

from storage.database.base import Database
from storage.models.research import ResearchTask

from tools.builder import (
    ToolBuildRequest,
    ToolBuilder,
)

from tools.registry import ToolRegistry


class TaskExecutor:
    """
    Executes research tasks while respecting dependencies,
    detecting missing capabilities, and invoking the
    Tool Builder when a required capability is unavailable.
    """

    def __init__(
        self,
        tasks: list[ResearchTask],
        database: Database,
        run_id: str,
        tool_registry: ToolRegistry,
    ):
        self.tasks = {
            task.task_id: task
            for task in tasks
        }

        self.database = database
        self.run_id = run_id
        self.tool_registry = tool_registry

        # -----------------------------------------
        # Capability Gap Detector
        # -----------------------------------------

        self.capability_gap_detector = (
            CapabilityGapDetector(
                tool_registry=tool_registry
            )
        )

        # -----------------------------------------
        # Tool Builder
        # -----------------------------------------

        self.tool_builder = ToolBuilder()

        # -----------------------------------------
        # Runtime State
        # -----------------------------------------

        self.completed_tasks: set[str] = set()

        self.results: dict[str, object] = {}

        self.capability_gaps: dict[
            str,
            list[str],
        ] = {}

        self.generated_tool_specs: dict[
            str,
            object,
        ] = {}

    def get_ready_tasks(
        self,
    ) -> list[ResearchTask]:
        """
        Return tasks whose dependencies have all
        been completed.
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

    def execute_task(
        self,
        task: ResearchTask,
    ) -> bool:
        """
        Execute one research task.

        Returns:
            True if the task completed successfully.
            False if the task is blocked by a
            missing capability.
        """

        print(
            f"Executing: {task.task_id} - "
            f"{task.description}"
        )

        task.status = "running"

        self.database.save_research_task(
            task
        )

        try:

            result = self._execute_task_logic(
                task
            )

            self.results[
                task.task_id
            ] = result

            task.status = "completed"

            self.database.save_research_task(
                task
            )

            self.completed_tasks.add(
                task.task_id
            )

            run = self.database.get_research_run(
                self.run_id
            )

            if run is not None:

                if (
                    task.task_id
                    not in run.completed_tasks
                ):

                    run.completed_tasks.append(
                        task.task_id
                    )

                self.database.save_research_run(
                    run
                )

            print(
                f"Completed: {task.task_id}"
            )

            return True

        except CapabilityGapError as exc:

            task.status = "blocked"

            self.database.save_research_task(
                task
            )

            self.capability_gaps[
                task.task_id
            ] = exc.missing_tools

            print(
                f"Blocked: {task.task_id}"
            )

            print(
                "Missing capabilities:"
            )

            for tool_name in (
                exc.missing_tools
            ):

                print(
                    f"  - {tool_name}"
                )

            return False

        except Exception as exc:

            task.status = "failed"

            self.database.save_research_task(
                task
            )

            run = self.database.get_research_run(
                self.run_id
            )

            if run is not None:

                if (
                    task.task_id
                    not in run.failed_tasks
                ):

                    run.failed_tasks.append(
                        task.task_id
                    )

                run.status = "failed"

                self.database.save_research_run(
                    run
                )

            print(
                f"Failed: {task.task_id} - {exc}"
            )

            raise

    def _execute_task_logic(
        self,
        task: ResearchTask,
    ):
        """
        Execute all tools required by a task.

        Missing tools are detected by the
        Capability Gap Detector.

        The Tool Builder then creates a specification
        describing how the missing capability could
        eventually be implemented.
        """

        # -----------------------------------------
        # No Tools Required
        # -----------------------------------------

        if not task.required_tools:

            print(
                f"No tools required for "
                f"'{task.task_id}'."
            )

            return {
                "status": "no_tools_required",
                "task_id": task.task_id,
            }

        # -----------------------------------------
        # Detect Capability Gaps
        # -----------------------------------------

        gaps = (
            self.capability_gap_detector.detect(
                task_id=task.task_id,
                required_tools=(
                    task.required_tools
                ),
            )
        )

        # -----------------------------------------
        # Build Specifications for Missing Tools
        # -----------------------------------------

        if gaps:

            missing_tools = []

            for gap in gaps:

                print(
                    f"Capability gap detected: "
                    f"{gap.capability}"
                )

                build_request = (
                    ToolBuildRequest(
                        capability=(
                            gap.capability
                        ),
                        reason=gap.reason,
                    )
                )

                generated_tool = (
                    self.tool_builder.build(
                        build_request
                    )
                )

                self.generated_tool_specs[
                    gap.capability
                ] = generated_tool

                print(
                    f"Tool specification generated: "
                    f"{generated_tool.name}"
                )

                print(
                    f"Purpose: "
                    f"{generated_tool.purpose}"
                )

                missing_tools.append(
                    gap.capability
                )

            raise CapabilityGapError(
                missing_tools
            )

        # -----------------------------------------
        # Execute Available Tools
        # -----------------------------------------

        results = {}

        for tool_name in (
            task.required_tools
        ):

            tool = self.tool_registry.get(
                tool_name
            )

            if tool is None:

                raise RuntimeError(
                    f"Tool '{tool_name}' was detected "
                    "as available but could not be "
                    "retrieved from the Tool Registry."
                )

            print(
                f"Using tool: {tool_name}"
            )

            result = self._execute_tool(
                tool_name=tool_name,
                tool=tool,
                task=task,
            )

            results[
                tool_name
            ] = result

        return {
            "status": "completed",
            "task_id": task.task_id,
            "tools": results,
        }

    def _execute_tool(
        self,
        tool_name: str,
        tool,
        task: ResearchTask,
    ):
        """
        Execute one registered tool.

        Currently only the literature search tool
        has an execution adapter.

        More tool adapters will be added as AURA
        gains capabilities.
        """

        if tool_name == "literature_search":

            query = task.description

            return tool.execute(
                query=query,
                max_results=5,
            )

        raise ValueError(
            f"No execution adapter exists for "
            f"registered tool '{tool_name}'."
        )

    def run(self) -> None:
        """
        Execute the research task graph.

        If a required capability is missing,
        AURA detects the gap, asks the Tool Builder
        to generate a specification, and blocks the run.

        The research run is never marked completed
        unless every task actually completes.
        """

        run = self.database.get_research_run(
            self.run_id
        )

        # -----------------------------------------
        # Start Run
        # -----------------------------------------

        if run is not None:

            run.status = "running"

            self.database.save_research_run(
                run
            )

        # -----------------------------------------
        # Execute Task Graph
        # -----------------------------------------

        while len(
            self.completed_tasks
        ) < len(self.tasks):

            ready_tasks = (
                self.get_ready_tasks()
            )

            # -----------------------------------------
            # No Ready Tasks
            # -----------------------------------------

            if not ready_tasks:

                incomplete_tasks = [
                    task
                    for task in (
                        self.tasks.values()
                    )
                    if task.status
                    not in {
                        "completed",
                        "failed",
                        "blocked",
                    }
                ]

                # -----------------------------------------
                # Capability Gap
                # -----------------------------------------

                if self.capability_gaps:

                    if run is not None:

                        run.status = "blocked"

                        self.database.save_research_run(
                            run
                        )

                    self._print_capability_gaps()

                    raise RuntimeError(
                        "Research run is blocked "
                        "because required tools "
                        "are unavailable."
                    )

                # -----------------------------------------
                # Unresolved Dependency / Cycle
                # -----------------------------------------

                if incomplete_tasks:

                    if run is not None:

                        run.status = "failed"

                        self.database.save_research_run(
                            run
                        )

                    raise RuntimeError(
                        "No executable tasks found. "
                        "The task graph may contain "
                        "a dependency cycle or "
                        "unresolved dependency."
                    )

                break

            progress_made = False

            # -----------------------------------------
            # Execute Ready Tasks
            # -----------------------------------------

            for task in ready_tasks:

                completed = (
                    self.execute_task(
                        task
                    )
                )

                if completed:

                    progress_made = True

                else:

                    # A missing capability was found.
                    #
                    # Stop immediately rather than
                    # pretending the research can continue.

                    if self.capability_gaps:

                        if run is not None:

                            run.status = "blocked"

                            self.database.save_research_run(
                                run
                            )

                        self._print_capability_gaps()

                        raise RuntimeError(
                            "Research run is blocked "
                            "because required tools "
                            "are unavailable."
                        )

            # -----------------------------------------
            # No Progress
            # -----------------------------------------

            if not progress_made:

                if run is not None:

                    run.status = "blocked"

                    self.database.save_research_run(
                        run
                    )

                raise RuntimeError(
                    "Research run is blocked by "
                    "missing capabilities."
                )

        # -----------------------------------------
        # Final Run State
        # -----------------------------------------

        run = self.database.get_research_run(
            self.run_id
        )

        if run is not None:

            if len(
                self.completed_tasks
            ) == len(self.tasks):

                run.status = "completed"

                self.database.save_research_run(
                    run
                )

    def _print_capability_gaps(
        self,
    ) -> None:
        """
        Print all capability gaps detected
        during the current research run.
        """

        print(
            "\n=== CAPABILITY GAPS ==="
        )

        for (
            task_id,
            missing_tools,
        ) in self.capability_gaps.items():

            print(
                f"{task_id}:"
            )

            for tool_name in missing_tools:

                print(
                    f"  - {tool_name}"
                )

        if self.generated_tool_specs:

            print(
                "\n=== GENERATED TOOL "
                "SPECIFICATIONS ==="
            )

            for (
                tool_name,
                tool,
            ) in self.generated_tool_specs.items():

                print(
                    f"\nTool: {tool.name}"
                )

                print(
                    f"Description: "
                    f"{tool.description}"
                )

                print(
                    f"Purpose: "
                    f"{tool.purpose}"
                )

                print(
                    f"Inputs: "
                    f"{tool.inputs}"
                )

                print(
                    f"Outputs: "
                    f"{tool.outputs}"
                )


class CapabilityGapError(Exception):
    """
    Raised when a research task requires tools
    that are not currently available.
    """

    def __init__(
        self,
        missing_tools: list[str],
    ):
        self.missing_tools = missing_tools

        message = (
            "Missing required tools: "
            + ", ".join(missing_tools)
        )

        super().__init__(
            message
        )
