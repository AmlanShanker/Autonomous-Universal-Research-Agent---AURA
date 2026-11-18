from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class QualityDimension(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    details: str
    warnings: list[str] = Field(default_factory=list)


class ResearchQualityAssessment(BaseModel):
    run_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    dimensions: dict[str, QualityDimension]
    warnings: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchReport(BaseModel):
    report_id: str
    run_id: str
    research_question: str
    run_status: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    execution_summary: dict[str, int] = Field(default_factory=dict)
    experiment_summary: dict[str, int] = Field(default_factory=dict)
    quality: ResearchQualityAssessment
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
