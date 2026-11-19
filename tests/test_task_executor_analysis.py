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
        if isinstance(self.response, list):
            return self.response.pop(0)
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


def test_analysis_prompt_includes_all_records_in_normal_literature_search_evidence():
    database = InMemoryDatabase()
    run = ResearchRun(
        run_id="run-bounded-analysis",
        research_question="Review retrieval methods.",
        status=ResearchRunStatus.PLANNED,
        tasks=["search", "synthesis"],
        completed_tasks=["search"],
    )
    database.save_research_run(run)
    search_task = ResearchTask(
        task_id="search",
        description="Search literature.",
        task_type="literature_search",
        status="completed",
    )
    synthesis_task = ResearchTask(
        task_id="synthesis",
        description="Synthesize the evidence.",
        task_type="synthesis",
        dependencies=["search"],
    )
    database.save_research_task(search_task)
    database.save_research_task(synthesis_task)
    database.save_execution_result(ExecutionResult(
        result_id="large-search-result",
        run_id=run.run_id,
        task_id="search",
        tool_id="literature_search",
        implementation_id="literature_search_v1",
        output={
            "query": "agent memory",
            "source": "arXiv",
            "count": 25,
            "results": [
                {
                    "title": f"Paper {index}",
                    "publication_year": "2026",
                    "authors": ["Author"],
                    "abstract": ("A" * 5000) + f" unique-tail-{index}",
                    "url": f"https://arxiv.org/abs/{index}",
                }
                for index in range(25)
            ],
        },
        metadata={"execution_type": "tool"},
    ))
    analysis = {
        "summary": "Bounded evidence was analyzed.",
        "findings": ["The retrieved set contains literature records."],
        "evidence": [{"source_result_id": "large-search-result", "claim": "The search result contains 25 records."}],
        "artifacts": {},
        "limitations": ["Record fields were compacted to fit the analysis context."],
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
    assert llm.prompt is not None
    assert "Paper 0" in llm.prompt
    assert "Paper 24" in llm.prompt
    assert '"records_included": 25' in llm.prompt
    assert '"records_omitted": 0' in llm.prompt
    assert "large-search-result" in llm.prompt
    stored = database.list_execution_results_for_run(run.run_id)
    original = next(item for item in stored if item.result_id == "large-search-result")
    assert len(original.output["results"]) == 25
    assert len(original.output["results"][0]["abstract"]) == 5014



def test_dependent_analysis_repairs_invalid_evidence_reference_once():
    database = InMemoryDatabase()
    run = ResearchRun(
        run_id="run-analysis-repair",
        research_question="Review retrieval evidence.",
        status=ResearchRunStatus.PLANNED,
        tasks=["search", "screen"],
        completed_tasks=["search"],
    )
    database.save_research_run(run)
    search_task = ResearchTask(
        task_id="search",
        description="Search literature.",
        task_type="literature_search",
        status="completed",
    )
    screen_task = ResearchTask(
        task_id="screen",
        description="Screen supplied literature.",
        task_type="screening",
        dependencies=["search"],
    )
    database.save_research_task(search_task)
    database.save_research_task(screen_task)
    database.save_execution_result(ExecutionResult(
        result_id="persisted-search-result",
        run_id=run.run_id,
        task_id="search",
        tool_id="literature_search",
        implementation_id="literature_search_v1",
        output={"source": "arXiv", "results": [{"title": "Evidence paper", "abstract": "RAG evaluation."}]},
        metadata={"execution_type": "tool"},
    ))
    invalid = {
        "summary": "Screened the supplied paper.",
        "findings": ["The supplied paper evaluates RAG."],
        "evidence": [{"source_result_id": "invented-result-id", "claim": "The abstract evaluates RAG."}],
        "artifacts": {"included": ["Evidence paper"]},
        "limitations": ["Only one record was supplied."],
        "recommendations": [],
        "metrics": [],
        "uncertainty": [],
    }
    corrected = {
        **invalid,
        "evidence": [{"source_result_id": "persisted-search-result", "claim": "The abstract evaluates RAG."}],
    }
    llm = FakeLLM([json.dumps(invalid), json.dumps(corrected)])
    executor = TaskExecutor(
        tasks=[search_task, screen_task],
        database=database,
        run_id=run.run_id,
        tool_registry=ToolRegistry(database),
        llm_client=llm,
    )

    executor.run()

    stored = database.list_execution_results_for_run(run.run_id)
    analysis_result = next(item for item in stored if item.task_id == "screen")
    assert analysis_result.output["analysis_status"] == "completed"
    assert analysis_result.output["analysis"]["evidence"][0]["source_result_id"] == "persisted-search-result"
    assert "ONLY permitted source_result_id values" in llm.prompt
    assert executor.tasks["screen"].status == "completed"
