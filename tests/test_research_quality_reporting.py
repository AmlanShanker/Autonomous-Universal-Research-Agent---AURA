from storage.database.memory import InMemoryDatabase
from storage.models.experiment import ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.research import ResearchPlan, ResearchRun, ResearchRunStatus, ResearchTask
from agents.research_quality import ResearchQualityEvaluator
from agents.research_reporter import ResearchReportBuilder


def artifacts():
    run = ResearchRun(
        run_id="run_quality",
        research_question="Does retrieval improve answer quality?",
        status=ResearchRunStatus.COMPLETED,
        tasks=["task_search", "task_eval"],
        completed_tasks=["task_search", "task_eval"],
        findings=["The labeled benchmark achieved full recall."],
    )
    plan = ResearchPlan(
        question=run.research_question,
        objectives=["Evaluate retrieval quality"],
        success_criteria=["Recall@k >= 0.8"],
    )
    tasks = [
        ResearchTask(task_id="task_search", description="Search evidence", task_type="search", required_tools=["search"]),
        ResearchTask(task_id="task_eval", description="Evaluate evidence", task_type="experiment"),
    ]
    executions = [
        ExecutionResult(
            result_id="exec_1", run_id=run.run_id, task_id="task_search",
            tool_id="search", implementation_id="search_v1", status="success",
            output={"sources": ["source-a"]},
        )
    ]
    experiments = [
        ExperimentResult(
            result_id="exp_result_1", experiment_id="exp_1", run_id=run.run_id,
            task_id="task_eval", status="completed", metrics={"recall_at_k": 1.0},
            passed=True, metadata={"random_seeds": [11, 12]},
        )
    ]
    return run, plan, tasks, executions, experiments


def test_quality_assessment_scores_coverage_execution_reproducibility_and_completion():
    run, _, tasks, executions, experiments = artifacts()

    assessment = ResearchQualityEvaluator().evaluate(run, tasks, executions, experiments)

    assert assessment.overall_score == 1.0
    assert assessment.dimensions["evidence_coverage"].score == 1.0
    assert assessment.dimensions["execution_reliability"].score == 1.0
    assert assessment.dimensions["experiment_reproducibility"].score == 1.0
    assert assessment.dimensions["task_completion"].score == 1.0
    assert assessment.warnings == []


def test_quality_assessment_reports_missing_evidence_and_seed_provenance():
    run, _, tasks, executions, experiments = artifacts()
    run.completed_tasks = ["task_search"]
    experiments[0].metadata = {}

    assessment = ResearchQualityEvaluator().evaluate(run, tasks, executions, experiments)

    assert assessment.dimensions["evidence_coverage"].score == 1.0
    assert assessment.dimensions["experiment_reproducibility"].score == 0.0
    assert assessment.dimensions["task_completion"].score == 0.5
    assert any("random-seed provenance" in warning for warning in assessment.warnings)


def test_report_builder_summarizes_findings_and_evidence_without_inventing_conclusions():
    run, plan, tasks, executions, experiments = artifacts()

    report = ResearchReportBuilder().build(run, plan, tasks, executions, experiments)

    assert report.research_question == plan.question
    assert report.findings == run.findings
    assert report.execution_summary == {"total": 1, "successful": 1, "failed": 0}
    assert report.experiment_summary == {"total": 1, "completed": 1, "failed": 0, "passed": 1}
    assert "1 recorded finding" in report.summary


def test_report_without_findings_explicitly_avoids_claiming_a_conclusion():
    run, plan, tasks, executions, experiments = artifacts()
    run.findings = []

    report = ResearchReportBuilder().build(run, plan, tasks, executions, experiments)

    assert report.findings == []
    assert "No final findings have been recorded" in report.summary


def test_research_report_persistence_round_trip_in_memory_database():
    run, plan, tasks, executions, experiments = artifacts()
    report = ResearchReportBuilder().build(run, plan, tasks, executions, experiments)
    database = InMemoryDatabase()

    database.save_research_report(report)

    assert database.get_research_report(report.report_id) == report
    assert database.get_research_report("missing") is None



def test_quality_assessment_does_not_treat_missing_seed_as_reproducible():
    run, _, tasks, executions, experiments = artifacts()
    experiments[0].metadata = {"random_seeds": [None, None]}

    assessment = ResearchQualityEvaluator().evaluate(run, tasks, executions, experiments)

    assert assessment.dimensions["experiment_reproducibility"].score == 0.0


def test_quality_assessment_penalizes_tool_tasks_without_successful_execution():
    run, _, tasks, _, experiments = artifacts()

    assessment = ResearchQualityEvaluator().evaluate(run, tasks, [], experiments)

    assert assessment.dimensions["execution_reliability"].score == 0.0
