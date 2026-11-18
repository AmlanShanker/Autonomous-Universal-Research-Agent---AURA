import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricCriterion(BaseModel):
    """A deterministic acceptance rule for one numeric experiment metric."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    operator: Literal["gt", "gte", "lt", "lte", "eq"]
    threshold: float

    @field_validator("metric")
    @classmethod
    def reject_blank_metric(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Metric name cannot be blank.")
        return value.strip()

    @field_validator("threshold")
    @classmethod
    def require_finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Metric threshold must be finite.")
        return value


class ExperimentSpec(BaseModel):
    """Untrusted, declarative experiment request attached to a research task."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1)
    method: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[MetricCriterion] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, le=10)
    random_seed: int | None = None

    @field_validator("hypothesis", "method", "adapter_name")
    @classmethod
    def reject_blank_spec_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment descriptions and adapter names cannot be blank.")
        return value.strip()

    @field_validator("success_criteria")
    @classmethod
    def require_unique_spec_criteria(cls, value: list[MetricCriterion]) -> list[MetricCriterion]:
        names = [criterion.metric for criterion in value]
        if len(names) != len(set(names)):
            raise ValueError("Experiment success criteria must use unique metric names.")
        return value


class ExperimentDefinition(BaseModel):
    """Persisted experiment identity plus declarative execution specification."""

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

    @field_validator(
        "experiment_id", "run_id", "task_id", "hypothesis", "method", "adapter_name"
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment identifiers and descriptions cannot be blank.")
        return value.strip()

    @field_validator("success_criteria")
    @classmethod
    def require_unique_criteria(cls, value: list[MetricCriterion]) -> list[MetricCriterion]:
        names = [criterion.metric for criterion in value]
        if len(names) != len(set(names)):
            raise ValueError("Experiment success criteria must use unique metric names.")
        return value


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
