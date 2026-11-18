import json

from agents.research_quality import ResearchQualityEvaluator
from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.execution import ExecutionResult
from storage.models.research import ResearchRun, ResearchRunStatus, ResearchTask
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return self.response


def test_dependent_analysis_persists_output_and_adds_final_findings():
    database = InMemoryDatabase()
    run = ResearchRun(
        run_id="run-analysis",
        research_question="Review memory retrieval methods.",
        status=ResearchRunStatus.PLANNED,
        tasks=["task_search", "task_synthesis"],
        completed_tasks=["task_search"],
    )
    database.save_research_run(run)

    search_task = ResearchTask(
        task_id="task_search",
        description="Search arXiv.",
        task_type="literature_search",
        status="completed",
    )
    synthesis_task = ResearchTask(
        task_id="task_synthesis",
        description="Synthesize the supplied evidence.",
        task_type="report_generation",
        dependencies=["task_search"],
    )
    database.save_research_task(search_task)
    database.save_research_task(synthesis_task)

    source = ExecutionResult(
        result_id="search-result-1",
        run_id=run.run_id,
        task_id="task_search",
        tool_id="literature_search",
        implementation_id="literature_search_v1",
        output={
            "source": "arXiv",
            "results": [
                {
                    "title": "Memory for autonomous agents",
                    "abstract": "The paper evaluates retrieval quality.",
                    "url": "https://arxiv.org/abs/0000.00000",
                }
            ],
        },
        metadata={"execution_type": "tool"},
    )
    database.save_execution_result(source)

    analysis = {
        "summary": "One supplied paper discusses retrieval quality.",
        "findings": ["The supplied paper evaluates retrieval quality."],
        "evidence": [
            {
                "source_result_id": "search-result-1",
                "claim": "The abstract explicitly mentions retrieval quality.",
            }
        ],
        "artifacts": {"unique_papers": [{"title": "Memory for autonomous agents"}]},
        "limitations": ["Only one abstract was supplied."],
        "recommendations": [],
        "metrics": [],
        "uncertainty": [],
    }
    llm = FakeLLM(json.dumps(analysis))
    executor = TaskExecutor(
        tasks=[search_task, synthesis_task],
        database=database,
        run_id=run.run_id,
        tool_registry=ToolRegistry(database),
        llm_client=llm,
    )

    executor.run()

    persisted = database.list_execution_results_for_run(run.run_id)
    task_result = next(item for item in persisted if item.task_id == "task_synthesis")
    completed_run = database.get_research_run(run.run_id)

    assert task_result.metadata["execution_type"] == "task_analysis"
    assert task_result.output["analysis_status"] == "completed"
    assert task_result.output["dependency_result_ids"] == ["search-result-1"]
    assert "Memory for autonomous agents" in llm.prompt
    assert completed_run.findings == analysis["findings"]
    assert completed_run.status == ResearchRunStatus.COMPLETED


def test_tool_free_task_without_dependency_evidence_is_not_counted_as_coverage():
    database = InMemoryDatabase()
    run = ResearchRun(
        run_id="run-empty-analysis",
        research_question="No evidence available",
        status=ResearchRunStatus.PLANNED,
        tasks=["task_empty"],
        completed_tasks=["task_empty"],
    )
    task = ResearchTask(
        task_id="task_empty",
        description="Analyze evidence",
        task_type="analysis",
        status="completed",
    )
    database.save_research_run(run)
    database.save_research_task(task)

    from storage.models.execution import ExecutionResult

    database.save_execution_result(
        ExecutionResult(
            result_id="empty-analysis-result",
            run_id=run.run_id,
            task_id=task.task_id,
            tool_id="internal_task_analysis",
            implementation_id="aura.internal.llm-analysis",
            output={"analysis_status": "insufficient_evidence"},
            metadata={"execution_type": "task_analysis"},
        )
    )

    assessment = ResearchQualityEvaluator().evaluate(
        run, [task], database.list_execution_results_for_run(run.run_id), []
    )

    assert assessment.dimensions["evidence_coverage"].score == 0.0
    assert "task_empty" in assessment.warnings[0]
