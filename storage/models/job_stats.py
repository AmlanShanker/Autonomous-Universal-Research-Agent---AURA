from datetime import datetime
from pydantic import BaseModel, Field


class ResearchJobQueueStats(BaseModel):
    """Point-in-time operational summary of durable research jobs."""

    total: int = Field(ge=0)
    queued: int = Field(ge=0)
    running: int = Field(ge=0)
    cancel_requested: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    expired_leases: int = Field(ge=0)
    active_workers: int = Field(ge=0)
    oldest_queued_at: datetime | None = None
