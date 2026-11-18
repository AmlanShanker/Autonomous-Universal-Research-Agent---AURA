import uuid

from agents.research_quality import ResearchQualityEvaluator
from storage.database.base import Database
from storage.models.experiment import ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.report import ResearchReport
from storage.models.research import ResearchPlan, ResearchRun, ResearchTask


class ResearchReportBuilder:
    """Build an auditable report from supplied persisted research artifacts."""

    def __init__(self, quality_evaluator: ResearchQualityEvaluator | None = None):
        self.quality_evaluator = quality_evaluator or ResearchQualityEvaluator()

    def build_and_save(
        self,
        database: Database,
        run: ResearchRun,
        plan: ResearchPlan,
        tasks: list[ResearchTask],
        execution_results: list[ExecutionResult],
        experiment_results: list[ExperimentResult],
    ) -> ResearchReport:
        """Build a report and persist it through the database contract."""
        report = self.build(
            run, plan, tasks, execution_results, experiment_results
        )
        database.save_research_report(report)
        return report

    def build(
        self,
        run: ResearchRun,
        plan: ResearchPlan,
        tasks: list[ResearchTask],
        execution_results: list[ExecutionResult],
        experiment_results: list[ExperimentResult],
    ) -> ResearchReport:
        executions = [r for r in execution_results if r.run_id == run.run_id]
        experiments = [r for r in experiment_results if r.run_id == run.run_id]
        quality = self.quality_evaluator.evaluate(
            run, tasks, executions, experiments
        )
        execution_counts = {
            "total": len(executions),
            "successful": sum(r.status == "success" for r in executions),
            "failed": sum(r.status != "success" for r in executions),
        }
        experiment_counts = {
            "total": len(experiments),
            "completed": sum(r.status == "completed" for r in experiments),
            "failed": sum(r.status == "failed" for r in experiments),
            "passed": sum(r.passed is True for r in experiments),
        }
        findings = [finding.strip() for finding in run.findings if finding.strip()]
        if findings:
            summary = f"Research produced {len(findings)} recorded finding(s) for: {plan.question}"
        else:
            summary = (
                f"No final findings have been recorded for: {plan.question}. "
                "This report summarizes execution evidence and quality checks only."
            )
        return ResearchReport(
            report_id=str(uuid.uuid4()),
            run_id=run.run_id,
            research_question=plan.question,
            run_status=run.status.value if hasattr(run.status, "value") else str(run.status),
            summary=summary,
            findings=findings,
            execution_summary=execution_counts,
            experiment_summary=experiment_counts,
            quality=quality,
            metadata={
                "task_count": len(tasks),
                "completed_task_count": len(set(run.completed_tasks)),
                "failed_task_count": len(set(run.failed_tasks)),
                "success_criteria": list(plan.success_criteria),
            },
        )
