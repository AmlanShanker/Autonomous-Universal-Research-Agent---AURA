from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from agents.capability_gap_detector import CapabilityGapDetector
from agents.research_quality import ResearchQualityEvaluator
from storage.models.improvement import ImprovementCycle, ImprovementProposal


_NAMESPACE = uuid.UUID("c67b9a2c-57b8-4c90-9d8e-42e3c1e2d4b1")
_QUALITY_FLOOR = 0.8


class AutonomousImprovementEngine:
    """Turns persisted run evidence into proposals; never mutates runtime capabilities."""

    def __init__(self, database, tool_registry, quality_floor: float = _QUALITY_FLOOR):
        if not 0.0 <= quality_floor <= 1.0:
            raise ValueError("quality_floor must be between 0 and 1.")
        self.database = database
        self.tool_registry = tool_registry
        self.quality_floor = quality_floor

    def analyze_run(self, run_id: str) -> ImprovementCycle:
        run = self.database.get_research_run(run_id)
        if run is None:
            raise KeyError(f"Research run '{run_id}' was not found.")
        tasks = [self.database.get_research_task(task_id) for task_id in run.tasks]
        if not tasks or any(task is None for task in tasks):
            raise ValueError("A complete persisted task graph is required for improvement analysis.")

        executions = self.database.list_execution_results_for_run(run_id)
        experiments = self.database.list_experiment_results_for_run(run_id)
        capability_availability = [
            {
                "task_id": task.task_id,
                "tool": tool_name,
                "available": bool(self.tool_registry.has(tool_name)),
            }
            for task in tasks
            for tool_name in task.required_tools
        ]
        evidence = {
            "run": run.model_dump(mode="json"),
            "quality_floor": self.quality_floor,
            "capability_availability": capability_availability,
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "executions": [item.model_dump(mode="json") for item in executions],
            "experiments": [item.model_dump(mode="json") for item in experiments],
        }
        canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        cycle_id = str(uuid.uuid5(_NAMESPACE, f"{run_id}:{digest}"))
        existing = self.database.get_improvement_cycle(cycle_id)
        if existing is not None:
            return existing

        proposals: list[ImprovementProposal] = []
        detector = CapabilityGapDetector(self.tool_registry)
        for task in tasks:
            for gap in detector.detect(task.task_id, task.required_tools):
                proposals.append(self._proposal(
                    cycle_id, run_id, "capability_gap", gap.capability,
                    gap.reason,
                    f"Review the requirement for '{gap.capability}'. If it is essential, design and register a reviewed trusted adapter in a separate change; do not generate or execute code from this proposal.",
                    [f"task:{gap.task_id}"],
                ))

        failed_by_task: dict[str, list[str]] = {}
        for execution in executions:
            if execution.status != "success":
                failed_by_task.setdefault(execution.task_id, []).append(execution.result_id)
        for task_id, result_ids in sorted(failed_by_task.items()):
            proposals.append(self._proposal(
                cycle_id, run_id, "execution_reliability", task_id,
                f"{len(result_ids)} persisted execution result(s) for this task did not succeed.",
                "Inspect the persisted failure diagnostics and make a bounded, reviewed reliability change; preserve retry limits and idempotency guarantees.",
                [f"task:{task_id}", *[f"execution:{item}" for item in sorted(result_ids)]],
            ))

        assessment = ResearchQualityEvaluator().evaluate(run, tasks, executions, experiments)
        for name, dimension in sorted(assessment.dimensions.items()):
            if name == "experiment_reproducibility" and not experiments:
                continue
            if dimension.score < self.quality_floor:
                proposals.append(self._proposal(
                    cycle_id, run_id, "research_quality", name,
                    f"{dimension.details} Observed score {dimension.score:.3f} is below the configured floor {self.quality_floor:.3f}.",
                    f"Investigate the evidence behind '{name}', define a measurable hypothesis, and evaluate any candidate only with an existing trusted experiment adapter.",
                    [f"quality_dimension:{name}"],
                ))

        for proposal in proposals:
            self.database.save_improvement_proposal(proposal)
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            run_id=run_id,
            evidence_sha256=digest,
            proposal_ids=[item.proposal_id for item in proposals],
            summary=f"Detected {len(proposals)} improvement proposal(s) from persisted research evidence.",
            status="open" if proposals else "completed",
            completed_at=None if proposals else datetime.now(timezone.utc),
        )
        self.database.save_improvement_cycle(cycle)
        return cycle

    @staticmethod
    def _proposal(cycle_id, run_id, category, capability, rationale, action, evidence_refs):
        identity = f"{category}:{capability}:" + json.dumps(sorted(evidence_refs))
        proposal_id = str(uuid.uuid5(_NAMESPACE, f"{cycle_id}:{identity}"))
        return ImprovementProposal(
            proposal_id=proposal_id,
            cycle_id=cycle_id,
            run_id=run_id,
            category=category,
            capability=capability,
            rationale=rationale,
            recommended_action=action,
            evidence_refs=evidence_refs,
        )

    def approve(self, proposal_id: str, reviewer: str) -> ImprovementProposal:
        proposal = self._get_proposal(proposal_id)
        if proposal.status != "proposed":
            raise ValueError("Only proposed improvements can be approved.")
        proposal.status = "approved"
        proposal.approved_by = self._reviewer(reviewer)
        proposal.updated_at = datetime.now(timezone.utc)
        self.database.save_improvement_proposal(proposal)
        return proposal

    def reject(self, proposal_id: str, reviewer: str, reason: str) -> ImprovementProposal:
        proposal = self._get_proposal(proposal_id)
        if proposal.status != "proposed":
            raise ValueError("Only proposed improvements can be rejected.")
        proposal.status = "rejected"
        proposal.rejected_by = self._reviewer(reviewer)
        proposal.rejection_reason = reason.strip()
        if not proposal.rejection_reason:
            raise ValueError("A rejection reason is required.")
        proposal.updated_at = datetime.now(timezone.utc)
        self.database.save_improvement_proposal(proposal)
        self._refresh_cycle(proposal.cycle_id)
        return proposal

    def validate(self, proposal_id: str, experiment_result_id: str) -> ImprovementProposal:
        proposal = self._get_proposal(proposal_id)
        if proposal.status != "approved":
            raise ValueError("Only approved improvements can be validated.")
        result = self.database.get_experiment_result(experiment_result_id)
        if result is None:
            raise KeyError(f"Experiment result '{experiment_result_id}' was not found.")
        if result.run_id != proposal.run_id:
            raise ValueError("Validation result must belong to the proposal's research run.")
        proposal.status = "validated"
        proposal.validation_experiment_result_id = result.result_id
        proposal.validation_passed = result.status == "completed" and result.passed is True
        proposal.updated_at = datetime.now(timezone.utc)
        self.database.save_improvement_proposal(proposal)
        self._refresh_cycle(proposal.cycle_id)
        return proposal

    def _get_proposal(self, proposal_id: str) -> ImprovementProposal:
        proposal = self.database.get_improvement_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Improvement proposal '{proposal_id}' was not found.")
        return proposal

    @staticmethod
    def _reviewer(value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 120:
            raise ValueError("Reviewer identity must contain 1 to 120 characters.")
        return normalized

    def _refresh_cycle(self, cycle_id: str) -> None:
        cycle = self.database.get_improvement_cycle(cycle_id)
        if cycle is None:
            return
        proposals = [self.database.get_improvement_proposal(item) for item in cycle.proposal_ids]
        if proposals and all(item is not None and item.status in {"rejected", "validated"} for item in proposals):
            cycle.status = "completed"
            cycle.completed_at = datetime.now(timezone.utc)
            self.database.save_improvement_cycle(cycle)
