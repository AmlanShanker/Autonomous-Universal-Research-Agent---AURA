from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """
    Persistent record of one research-tool execution.

    This model stores evidence produced by a trusted tool or an
    internal, evidence-bound analysis task during a research run.

    Internal analysis records use an explicit execution_type metadata
    value and an application-owned implementation identifier. The
    record represents execution history, not a research conclusion.
    """

    result_id: str

    run_id: str

    task_id: str

    tool_id: str

    implementation_id: str

    implementation_version: str = "1.0.0"

    status: str = "success"

    output: dict = Field(
        default_factory=dict,
        description=(
            "Structured output produced by the tool."
        ),
    )

    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Additional execution metadata such as "
            "timing, source information, or future "
            "reproducibility information."
        ),
    )
