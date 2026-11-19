from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ResearchJobAttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LEASE_EXPIRED = "lease_expired"


class ResearchJobAttempt(BaseModel):
    """Append-only operational record for one worker claim of a research job."""

    attempt_id: str
    job_id: str
    attempt_number: int = Field(ge=1)
    worker_id: str
    status: ResearchJobAttemptStatus = ResearchJobAttemptStatus.RUNNING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = Field(default=None, max_length=2000)

    @field_validator("attempt_id", "job_id", "worker_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Attempt identifiers and worker ID cannot be blank.")
        return value

    def finish(
        self,
        status: ResearchJobAttemptStatus,
        error: str | None = None,
    ) -> None:
        if self.status != ResearchJobAttemptStatus.RUNNING:
            raise ValueError("Only running attempts can be finished.")
        if status == ResearchJobAttemptStatus.RUNNING:
            raise ValueError("A finished attempt cannot remain running.")
        self.status = status
        self.error = (error or None)
        if self.error is not None:
            self.error = self.error[:2000]
        self.finished_at = datetime.now(timezone.utc)
