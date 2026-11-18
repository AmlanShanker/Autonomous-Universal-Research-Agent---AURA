from enum import Enum

from pydantic import BaseModel, Field

from storage.models.experiment import ExperimentSpec


class ResearchPlan(BaseModel):
    question: str = Field(
        ...,
        description="The main research question.",
    )

    objectives: list[str] = Field(
        default_factory=list,
        description="What the research should accomplish.",
    )

    hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypotheses that the research will investigate.",
    )

    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities or tools required to complete the research.",
    )

    experiments: list[str] = Field(
        default_factory=list,
        description="Experiments that should be performed.",
    )

    success_criteria: list[str] = Field(
        default_factory=list,
        description="Conditions used to determine whether the research succeeded.",
    )


class ResearchTask(BaseModel):
    task_id: str
    description: str
    task_type: str

    dependencies: list[str] = Field(
        default_factory=list,
    )

    required_tools: list[str] = Field(
        default_factory=list,
    )

    tool_inputs: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Structured inputs supplied to each tool required by the task. "
            "Keys are tool names."
        ),
    )

    experiments: list[ExperimentSpec] = Field(
        default_factory=list,
        description="Declarative experiments to execute through trusted registered adapters.",
    )

    status: str = "pending"


class ResearchRunStatus(str, Enum):
    """
    Lifecycle states for an AURA research run.
    """

    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ResearchRun(BaseModel):
    """
    Persistent state of one AURA research run.

    A research run represents the lifecycle of an entire
    research workflow, not an individual task execution.
    """

    run_id: str

    research_question: str

    status: ResearchRunStatus = ResearchRunStatus.CREATED

    tasks: list[str] = Field(
        default_factory=list,
        description="Task IDs belonging to this research run.",
    )

    completed_tasks: list[str] = Field(
        default_factory=list,
        description="Task IDs that completed successfully.",
    )

    failed_tasks: list[str] = Field(
        default_factory=list,
        description="Task IDs that failed.",
    )

    findings: list[str] = Field(
        default_factory=list,
        description=(
            "High-level research findings and conclusions. "
            "Raw execution results are stored separately."
        ),
    )

    def transition_to(
        self,
        new_status: ResearchRunStatus,
    ) -> None:
        """
        Transition the research run to another lifecycle state.

        Only explicitly supported state transitions are allowed.
        """

        allowed_transitions = {
            ResearchRunStatus.CREATED: {
                ResearchRunStatus.PLANNED,
                ResearchRunStatus.FAILED,
            },
            ResearchRunStatus.PLANNED: {
                ResearchRunStatus.RUNNING,
                ResearchRunStatus.FAILED,
            },
            ResearchRunStatus.RUNNING: {
                ResearchRunStatus.BLOCKED,
                ResearchRunStatus.FAILED,
                ResearchRunStatus.COMPLETED,
                ResearchRunStatus.CANCELLED,
            },
            ResearchRunStatus.BLOCKED: {
                ResearchRunStatus.RUNNING,
                ResearchRunStatus.FAILED,
            },
            ResearchRunStatus.FAILED: set(),
            ResearchRunStatus.CANCELLED: set(),
            ResearchRunStatus.COMPLETED: set(),
        }

        if new_status == self.status:
            return

        allowed = allowed_transitions.get(
            self.status,
            set(),
        )

        if new_status not in allowed:
            raise ValueError(
                "Invalid research run state transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status


class ToolDefinition(BaseModel):
    """
    Persistent definition of an AURA research tool.

    This model represents the logical capability and
    contract of a tool.

    A ToolDefinition describes a capability.
    It is not executable code.

    A tool can have multiple implementations.
    """

    tool_id: str

    name: str

    capability: str = ""

    description: str

    purpose: str = ""

    version: str = "1.0.0"

    input_schema: dict = Field(
        default_factory=dict,
    )

    output_schema: dict = Field(
        default_factory=dict,
    )

    dependencies: list[str] = Field(
        default_factory=list,
    )

    tests: list[str] = Field(
        default_factory=list,
    )

    validation_requirements: list[str] = Field(
        default_factory=list,
    )

    implementation_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers of implementations that provide "
            "this tool capability."
        ),
    )

    status: str = "draft"

    reliability: float | None = None


class ToolImplementation(BaseModel):
    """
    Persistent metadata describing one implementation
    of a tool.

    This model does NOT contain executable source code.

    It records where an implementation comes from,
    which version it represents, and its lifecycle state.
    """

    implementation_id: str

    tool_id: str

    version: str = "1.0.0"

    implementation_type: str

    source_reference: str = ""

    entrypoint: str = ""

    dependencies: list[str] = Field(
        default_factory=list,
    )

    validation_status: str = "unvalidated"

    status: str = "draft"

    description: str = ""

    metadata: dict = Field(
        default_factory=dict,
    )
