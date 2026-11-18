from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ResearchJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchJob(BaseModel):
    """Durable request to execute one persisted research run."""

    job_id: str
    run_id: str
    status: ResearchJobStatus = ResearchJobStatus.QUEUED
    attempts: int = 0
    max_attempts: int = Field(default=3, ge=1, le=5)
    completed_tasks: int = 0
    total_tasks: int = 0
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    idempotency_key: str | None = None
    error: str | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("job_id", "run_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Job and run identifiers cannot be blank.")
        return value

    @field_validator("idempotency_key")
    @classmethod
    def normalize_idempotency_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("Idempotency keys must contain 1 to 200 characters.")
        return normalized

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
