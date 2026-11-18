from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MetricCriterion(BaseModel):
    """A deterministic acceptance rule for one numeric experiment metric."""

    metric: str = Field(min_length=1)
    operator: Literal["gt", "gte", "lt", "lte", "eq"]
    threshold: float


class ExperimentDefinition(BaseModel):
    """Declarative experiment specification; it contains no executable code."""

    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    method: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[MetricCriterion] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, le=10)
    random_seed: int | None = None

    @field_validator("experiment_id", "run_id", "task_id", "hypothesis", "method", "adapter_name")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment identifiers and descriptions cannot be blank.")
        return value.strip()


class ExperimentResult(BaseModel):
    """Persisted outcome and reproducibility record for one experiment."""

    result_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, bool] = Field(default_factory=dict)
    passed: bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
