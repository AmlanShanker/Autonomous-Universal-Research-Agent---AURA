from agents.task_executor import TaskExecutor
from memory.research_memory import ResearchMemoryItem
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchRun, ResearchTask
from tools.registry import ToolRegistry


class FakeMemoryRetriever:
    def __init__(self, findings):
        self.findings = findings
        self.calls = []

    def retrieve(
        self,
        query,
        limit=5,
        exclude_run_id=None,
    ):
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "exclude_run_id": exclude_run_id,
            }
        )
        return self.findings


def create_memory(
    memory_id="memory_previous_001",
    run_id="run_previous",
):
    return ResearchMemoryItem(
        memory_id=memory_id,
        run_id=run_id,
        task_id="task_previous",
        research_question="How does RAG chunking affect retrieval?",
        content="Smaller chunks improved recall in the evaluated dataset.",
        memory_type="finding",
        tags=["rag", "chunking", "retrieval"],
        metadata={"source": "previous experiment"},
    )


def test_executor_injects_prior_findings_into_task_context():
    database = InMemoryDatabase()
    database.save_research_run(
        ResearchRun(
            run_id="run_current",
            research_question="Investigate RAG retrieval quality.",
        )
    )

    finding = create_memory()
    retriever = FakeMemoryRetriever([finding])

    task = ResearchTask(
        task_id="task_analysis",
        description="Analyze chunk-size trade-offs.",
        task_type="analysis",
    )

    executor = TaskExecutor(
        database=database,
        tasks=[task],
        run_id="run_current",
        tool_registry=ToolRegistry(),
        memory_retriever=retriever,
    )

    context = executor.build_task_context(task)

    assert retriever.calls == [
        {
            "query": (
                "Investigate RAG retrieval quality. "
                "Analyze chunk-size trade-offs."
            ),
            "limit": 5,
            "exclude_run_id": "run_current",
        }
    ]

    assert context["prior_findings"] == [
        {
            "memory_id": "memory_previous_001",
            "source_run_id": "run_previous",
            "source_task_id": "task_previous",
            "research_question": (
                "How does RAG chunking affect retrieval?"
            ),
            "content": (
                "Smaller chunks improved recall in the evaluated dataset."
            ),
            "memory_type": "finding",
            "tags": ["rag", "chunking", "retrieval"],
            "metadata": {"source": "previous experiment"},
        }
    ]


def test_executor_caches_prior_findings_per_task():
    database = InMemoryDatabase()
    database.save_research_run(
        ResearchRun(
            run_id="run_current",
            research_question="Investigate RAG retrieval quality.",
        )
    )

    retriever = FakeMemoryRetriever([create_memory()])
    task = ResearchTask(
        task_id="task_analysis",
        description="Analyze chunk-size trade-offs.",
        task_type="analysis",
    )

    executor = TaskExecutor(
        database=database,
        tasks=[task],
        run_id="run_current",
        tool_registry=ToolRegistry(),
        memory_retriever=retriever,
    )

    first = executor.build_task_context(task)
    second = executor.build_task_context(task)

    assert first == second
    assert len(retriever.calls) == 1


def test_executor_returns_empty_prior_findings_without_retriever():
    database = InMemoryDatabase()
    task = ResearchTask(
        task_id="task_analysis",
        description="Analyze the topic.",
        task_type="analysis",
    )

    executor = TaskExecutor(
        database=database,
        tasks=[task],
        run_id="run_current",
        tool_registry=ToolRegistry(),
    )

    context = executor.build_task_context(task)

    assert context["prior_findings"] == []
