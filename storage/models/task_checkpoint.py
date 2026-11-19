from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskCheckpoint(BaseModel):
    """Durable snapshot of one task's progress for safe worker restart recovery."""

    checkpoint_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)
    status: Literal["running", "completed", "failed", "blocked"]
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
