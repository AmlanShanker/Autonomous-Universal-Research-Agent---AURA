from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ProposalStatus = Literal["proposed", "approved", "rejected", "validated"]


class ImprovementProposal(BaseModel):
    """Evidence-backed, human-governed proposal; never executable code."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1)
    cycle_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    category: str = Field(min_length=1, max_length=80)
    capability: str = Field(min_length=1, max_length=160)
    rationale: str = Field(min_length=1, max_length=2000)
    recommended_action: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    status: ProposalStatus = "proposed"
    approved_by: str | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    validation_experiment_result_id: str | None = None
    validation_passed: bool | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("proposal_id", "cycle_id", "run_id", "category", "capability", "rationale", "recommended_action")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Improvement proposal fields cannot be blank.")
        return value.strip()


class ImprovementCycle(BaseModel):
    """Immutable evidence snapshot and the proposals derived from it."""

    model_config = ConfigDict(extra="forbid")

    cycle_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposal_ids: list[str] = Field(default_factory=list, max_length=500)
    status: Literal["open", "completed"] = "open"
    summary: str = Field(min_length=1, max_length=2000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @field_validator("proposal_ids")
    @classmethod
    def require_unique_proposal_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Improvement cycle proposal IDs must be unique.")
        return value
