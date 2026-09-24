from agents.capability_gap_detector import (
    CapabilityGapDetector,
)

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
    detecting missing capabilities, reusing known tool
    definitions, validating executable tools, selecting
    usable implementations, and invoking the Tool Builder
    when a capability is completely unknown.

    Important distinction:

    - An executable ResearchTool has an implementation.
    - A ToolDefinition is persistent knowledge about a tool.
    - A ToolImplementation describes one implementation.
    - A validated ResearchTool has passed validation against
      its ToolDefinition.
    - Only validated and active implementations may be
      executed by AURA.
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

        self.reused_tool_definitions: dict[
            str,
            object,
        ] = {}

        self.validation_failures: dict[
            str,
            list[str],
        ] = {}

        self.implementation_failures: dict[
            str,
            list[str],
        ] = {}

        self.selected_implementations: dict[
            str,
            str,
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

            False if the task is blocked by a missing
            executable capability, validation failure,
            or unusable implementation.
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
                "Missing executable capabilities:"
            )

            for tool_name in (
                exc.missing_tools
            ):
                print(
                    f"  - {tool_name}"
                )

            return False

        except ToolValidationError as exc:
            task.status = "blocked"

            self.database.save_research_task(
                task
            )

            self.validation_failures[
                task.task_id
            ] = exc.failed_tools

            print(
                f"Blocked: {task.task_id}"
            )

            print(
                "Tool validation failures:"
            )

            for tool_name in (
                exc.failed_tools
            ):
                print(
                    f"  - {tool_name}"
                )

            run = self.database.get_research_run(
                self.run_id
            )

            if run is not None:
                run.status = "blocked"

                self.database.save_research_run(
                    run
                )

            return False

        except ToolImplementationError as exc:
            task.status = "blocked"

            self.database.save_research_task(
                task
            )

            self.implementation_failures[
                task.task_id
            ] = exc.failed_tools

            print(
                f"Blocked: {task.task_id}"
            )

            print(
                "Tool implementation failures:"
            )

            for tool_name in (
                exc.failed_tools
            ):
                print(
                    f"  - {tool_name}"
                )

            run = self.database.get_research_run(
                self.run_id
            )

            if run is not None:
                run.status = "blocked"

                self.database.save_research_run(
                    run
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

        Missing executable tools are first checked against
        persistent tool definitions.

        Existing executable tools are validated before
        implementation selection.

        Execution lifecycle:

            Capability
                 ↓
            ToolDefinition
                 ↓
            ToolImplementation
                 ↓
            Validation
                 ↓
            Implementation Selection
                 ↓
            Executable Tool Resolution
                 ↓
            Execution

        If AURA does not know the capability:

            Capability
                 ↓
            Tool Builder
                 ↓
            Generated specification
                 ↓
            Persistent definition
                 ↓
            Block until implementation exists

        No generated code is executed here.
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
        # Resolve Capability Gaps
        # -----------------------------------------

        if gaps:
            unresolved_capabilities = []

            for gap in gaps:
                capability = gap.capability

                print(
                    f"Capability gap detected: "
                    f"{capability}"
                )

                # -----------------------------------------
                # Check Persistent Tool Knowledge
                # -----------------------------------------

                existing_definition = (
                    self.tool_registry
                    .find_definition_by_capability(
                        capability
                    )
                )

                if existing_definition is not None:
                    self.reused_tool_definitions[
                        capability
                    ] = existing_definition

                    print(
                        f"Known tool definition found: "
                        f"{existing_definition.name}"
                    )

                    print(
                        "The definition is persistent "
                        "knowledge, but no executable "
                        "implementation is registered."
                    )

                    unresolved_capabilities.append(
                        capability
                    )

                    continue

                # -----------------------------------------
                # Build New Tool Specification
                # -----------------------------------------

                print(
                    f"No known definition found "
                    f"for '{capability}'."
                )

                build_request = (
                    ToolBuildRequest(
                        capability=capability,
                        reason=gap.reason,
                    )
                )

                generated_tool = (
                    self.tool_builder.build(
                        build_request
                    )
                )

                # -----------------------------------------
                # Convert to Persistent Definition
                # -----------------------------------------

                tool_definition = (
                    self.tool_builder
                    .to_tool_definition(
                        generated_tool
                    )
                )

                # -----------------------------------------
                # Persist Definition
                # -----------------------------------------

                self.database.save_tool(
                    tool_definition
                )

                # Register the definition in the
                # current registry as knowledge.
                #
                # This does NOT make it executable.

                if not self.tool_registry.has_definition(
                    tool_definition.tool_id
                ):
                    self.tool_registry.register_definition(
                        tool_definition
                    )

                self.generated_tool_specs[
                    capability
                ] = generated_tool

                print(
                    f"Tool specification generated: "
                    f"{generated_tool.name}"
                )

                print(
                    f"Purpose: "
                    f"{generated_tool.purpose}"
                )

                print(
                    "Tool definition persisted "
                    "to MongoDB."
                )

                unresolved_capabilities.append(
                    capability
                )

            # -----------------------------------------
            # Block Until Implementations Exist
            # -----------------------------------------

            raise CapabilityGapError(
                unresolved_capabilities
            )

        # -----------------------------------------
        # Validate Required Tools
        # -----------------------------------------

        validation_failures = []

        for tool_name in task.required_tools:
            if not self.tool_registry.has_validated(
                tool_name
            ):
                print(
                    f"Tool '{tool_name}' has not "
                    "been validated."
                )

                result = (
                    self.tool_registry.validate_tool(
                        tool_name
                    )
                )

                if not result.valid:
                    validation_failures.append(
                        tool_name
                    )

                    print(
                        f"Validation failed for "
                        f"'{tool_name}'."
                    )

                    for error in result.errors:
                        print(
                            f"  - {error}"
                        )

                else:
                    print(
                        f"Tool '{tool_name}' "
                        "validated successfully."
                    )

        if validation_failures:
            raise ToolValidationError(
                validation_failures
            )

        # -----------------------------------------
        # Select Executable Implementations
        # -----------------------------------------

        implementation_failures = []

        selected_tools = {}

        for tool_name in task.required_tools:
            resolved = (
                self.tool_registry
                .get_executable_implementation(
                    tool_name
                )
            )

            if resolved is None:
                implementation_failures.append(
                    tool_name
                )

                print(
                    f"No usable executable "
                    f"implementation found for "
                    f"'{tool_name}'."
                )

                continue

            implementation, tool = resolved

            self.selected_implementations[
                tool_name
            ] = implementation.implementation_id

            selected_tools[
                tool_name
            ] = (
                implementation,
                tool,
            )

            print(
                f"Selected implementation: "
                f"{implementation.implementation_id}"
            )

        if implementation_failures:
            raise ToolImplementationError(
                implementation_failures
            )

        # -----------------------------------------
        # Execute Selected Implementations
        # -----------------------------------------

        results = {}

        for tool_name in task.required_tools:
            implementation, tool = selected_tools[
                tool_name
            ]

            print(
                f"Using implementation "
                f"'{implementation.implementation_id}' "
                f"for tool '{tool_name}'."
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
            "implementations": (
                self.selected_implementations.copy()
            ),
        }

    def _execute_tool(
        self,
        tool_name: str,
        tool,
        task: ResearchTask,
    ):
        """
        Execute one selected executable tool.

        The selected tool has already been resolved
        through its ToolImplementation.

        Tool inputs are supplied through the task's
        structured tool_inputs field.
        """

        # -----------------------------------------
        # Literature Search
        # -----------------------------------------

        if tool_name == "literature_search":
            tool_inputs = task.tool_inputs.get(
                tool_name,
                {},
            )

            query = tool_inputs.get(
                "query",
                task.description,
            )

            max_results = tool_inputs.get(
                "max_results",
                5,
            )

            return tool.execute(
                query=query,
                max_results=max_results,
            )

        # -----------------------------------------
        # Dataset Download
        # -----------------------------------------

        if tool_name == "dataset_download":
            tool_inputs = task.tool_inputs.get(
                tool_name,
                {},
            )

            source_url = tool_inputs.get(
                "source_url"
            )

            destination_path = tool_inputs.get(
                "destination_path"
            )

            checksum = tool_inputs.get(
                "checksum"
            )

            if not source_url:
                raise ValueError(
                    "dataset_download requires "
                    "'source_url' in task.tool_inputs."
                )

            if not destination_path:
                raise ValueError(
                    "dataset_download requires "
                    "'destination_path' in task.tool_inputs."
                )

            return tool.execute(
                source_url=source_url,
                destination_path=destination_path,
                checksum=checksum,
            )

        # -----------------------------------------
        # Unknown Adapter
        # -----------------------------------------

        raise ValueError(
            f"No execution adapter exists for "
            f"registered tool '{tool_name}'."
        )

    def run(
        self,
    ) -> None:
        """
        Execute the research task graph.

        A research run is completed only when every
        task actually completes.

        Missing executable capabilities, failed
        validation, or unavailable implementations
        cause the run to become blocked.
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
                        "because required executable "
                        "capabilities are unavailable."
                    )

                # -----------------------------------------
                # Validation Failure
                # -----------------------------------------

                if self.validation_failures:
                    if run is not None:
                        run.status = "blocked"

                        self.database.save_research_run(
                            run
                        )

                    self._print_validation_failures()

                    raise RuntimeError(
                        "Research run is blocked "
                        "because one or more tools "
                        "failed validation."
                    )

                # -----------------------------------------
                # Implementation Failure
                # -----------------------------------------

                if self.implementation_failures:
                    if run is not None:
                        run.status = "blocked"

                        self.database.save_research_run(
                            run
                        )

                    self._print_implementation_failures()

                    raise RuntimeError(
                        "Research run is blocked "
                        "because one or more required "
                        "tool implementations are "
                        "unavailable."
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
                completed = self.execute_task(
                    task
                )

                if completed:
                    progress_made = True

                else:
                    # Stop immediately when a capability
                    # cannot actually be executed.

                    if self.capability_gaps:
                        if run is not None:
                            run.status = "blocked"

                            self.database.save_research_run(
                                run
                            )

                        self._print_capability_gaps()

                        raise RuntimeError(
                            "Research run is blocked "
                            "because required executable "
                            "capabilities are unavailable."
                        )

                    if self.validation_failures:
                        if run is not None:
                            run.status = "blocked"

                            self.database.save_research_run(
                                run
                            )

                        self._print_validation_failures()

                        raise RuntimeError(
                            "Research run is blocked "
                            "because one or more tools "
                            "failed validation."
                        )

                    if self.implementation_failures:
                        if run is not None:
                            run.status = "blocked"

                            self.database.save_research_run(
                                run
                            )

                        self._print_implementation_failures()

                        raise RuntimeError(
                            "Research run is blocked "
                            "because one or more required "
                            "tool implementations are "
                            "unavailable."
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
                    "Research run is blocked."
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
        Print capability gaps, reused definitions,
        and newly generated specifications.
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

        # -----------------------------------------
        # Known Definitions
        # -----------------------------------------

        if self.reused_tool_definitions:
            print(
                "\n=== REUSED TOOL DEFINITIONS ==="
            )

            for (
                capability,
                definition,
            ) in self.reused_tool_definitions.items():

                print(
                    f"\nCapability: "
                    f"{capability}"
                )

                print(
                    f"Tool: "
                    f"{definition.name}"
                )

                print(
                    f"Status: "
                    f"{definition.status}"
                )

        # -----------------------------------------
        # Generated Specifications
        # -----------------------------------------

        if self.generated_tool_specs:
            print(
                "\n=== GENERATED TOOL "
                "SPECIFICATIONS ==="
            )

            for (
                capability,
                tool,
            ) in self.generated_tool_specs.items():

                print(
                    f"\nCapability: "
                    f"{capability}"
                )

                print(
                    f"Tool: "
                    f"{tool.name}"
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

                print(
                    f"Dependencies: "
                    f"{tool.dependencies}"
                )

    def _print_validation_failures(
        self,
    ) -> None:
        """
        Print tools that failed validation.
        """

        print(
            "\n=== TOOL VALIDATION FAILURES ==="
        )

        for (
            task_id,
            failed_tools,
        ) in self.validation_failures.items():

            print(
                f"{task_id}:"
            )

            for tool_name in failed_tools:
                print(
                    f"  - {tool_name}"
                )

    def _print_implementation_failures(
        self,
    ) -> None:
        """
        Print tools for which no executable
        implementation could be resolved.
        """

        print(
            "\n=== TOOL IMPLEMENTATION FAILURES ==="
        )

        for (
            task_id,
            failed_tools,
        ) in self.implementation_failures.items():

            print(
                f"{task_id}:"
            )

            for tool_name in failed_tools:
                print(
                    f"  - {tool_name}"
                )


class CapabilityGapError(Exception):
    """
    Raised when a research task requires capabilities
    that cannot currently be executed.
    """

    def __init__(
        self,
        missing_tools: list[str],
    ):
        self.missing_tools = missing_tools

        message = (
            "Missing required tools: "
            + ", ".join(
                missing_tools
            )
        )

        super().__init__(
            message
        )


class ToolValidationError(Exception):
    """
    Raised when one or more tools fail validation.
    """

    def __init__(
        self,
        failed_tools: list[str],
    ):
        self.failed_tools = failed_tools

        message = (
            "Tool validation failed: "
            + ", ".join(
                failed_tools
            )
        )

        super().__init__(
            message
        )


class ToolImplementationError(Exception):
    """
    Raised when one or more required tools do not
    have a usable executable implementation.
    """

    def __init__(
        self,
        failed_tools: list[str],
    ):
        self.failed_tools = failed_tools

        message = (
            "Tool implementation unavailable: "
            + ", ".join(
                failed_tools
            )
        )

        super().__init__(
            message
        )
