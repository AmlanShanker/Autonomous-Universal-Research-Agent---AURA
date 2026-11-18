from collections import Counter

from storage.models.experiment import ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.research import ResearchRun, ResearchTask
from storage.models.report import QualityDimension, ResearchQualityAssessment


class ResearchQualityEvaluator:
    """Deterministic, transparent quality checks over persisted run evidence."""

    def evaluate(
        self,
        run: ResearchRun,
        tasks: list[ResearchTask],
        execution_results: list[ExecutionResult],
        experiment_results: list[ExperimentResult],
    ) -> ResearchQualityAssessment:
        warnings: list[str] = []
        task_ids = {task.task_id for task in tasks}
        execution_by_task = Counter(
            result.task_id for result in execution_results if result.run_id == run.run_id
        )
        experiment_by_task = Counter(
            result.task_id for result in experiment_results if result.run_id == run.run_id
        )

        valid_execution_task_ids = {
            result.task_id
            for result in execution_results
            if result.run_id == run.run_id
            and result.status == "success"
            and not (
                result.metadata.get("execution_type") == "task_analysis"
                and result.output.get("analysis_status") != "completed"
            )
        }
        covered_task_ids = (
            task_ids.intersection(valid_execution_task_ids)
            | task_ids.intersection(experiment_by_task)
        )
        covered = len(covered_task_ids)
        coverage = covered / len(task_ids) if task_ids else 0.0
        if task_ids and covered < len(task_ids):
            missing = sorted(task_ids - covered_task_ids)
            warnings.append("Tasks without substantive persisted evidence: " + ", ".join(missing))

        relevant_executions = [r for r in execution_results if r.run_id == run.run_id]
        tool_task_ids = {task.task_id for task in tasks if task.required_tools}
        successful_tool_tasks = {
            r.task_id for r in relevant_executions
            if r.status == "success" and r.task_id in tool_task_ids
        }
        execution_score = (
            len(successful_tool_tasks) / len(tool_task_ids) if tool_task_ids else 1.0
        )
        successful_executions = sum(r.status == "success" for r in relevant_executions)
        if any(r.status != "success" for r in relevant_executions):
            warnings.append("One or more tool executions did not succeed.")

        relevant_experiments = [r for r in experiment_results if r.run_id == run.run_id]
        completed_experiments = [r for r in relevant_experiments if r.status == "completed"]
        reproducible = sum(
            bool(r.metadata.get("random_seeds"))
            and all(seed is not None for seed in r.metadata.get("random_seeds", []))
            for r in completed_experiments
        )
        reproducibility_score = reproducible / len(relevant_experiments) if relevant_experiments else 0.0
        if not relevant_experiments:
            warnings.append("Experiment reproducibility was not assessed because no experiments were executed.")
        if any(r.status == "failed" for r in relevant_experiments):
            warnings.append("One or more experiments failed.")
        if completed_experiments and reproducible < len(completed_experiments):
            warnings.append("Some completed experiments do not record random-seed provenance.")

        planned = len(tasks)
        completed = len(set(run.completed_tasks).intersection(task_ids))
        completion_score = completed / planned if planned else 0.0
        if run.failed_tasks:
            warnings.append("The research run records failed tasks.")

        dimensions = {
            "evidence_coverage": QualityDimension(
                score=coverage,
                details=f"{covered} of {len(task_ids)} tasks have persisted evidence.",
            ),
            "execution_reliability": QualityDimension(
                score=execution_score,
                details=f"{len(successful_tool_tasks)} of {len(tool_task_ids)} tasks requiring tools have a successful execution.",
            ),
            "experiment_reproducibility": QualityDimension(
                score=reproducibility_score,
                details=f"{reproducible} of {len(relevant_experiments)} experiment results include seed provenance.",
            ),
            "task_completion": QualityDimension(
                score=completion_score,
                details=f"{completed} of {planned} planned tasks are marked completed.",
            ),
        }
        # A dimension with no applicable experiment data is reported as
        # unassessed (score 0) but excluded from the aggregate.
        applicable_dimensions = [
            dimension for name, dimension in dimensions.items()
            if name != "experiment_reproducibility" or bool(relevant_experiments)
        ]
        overall = (
            sum(item.score for item in applicable_dimensions) / len(applicable_dimensions)
            if applicable_dimensions else 0.0
        )
        return ResearchQualityAssessment(
            run_id=run.run_id,
            overall_score=overall,
            dimensions=dimensions,
            warnings=warnings,
        )
