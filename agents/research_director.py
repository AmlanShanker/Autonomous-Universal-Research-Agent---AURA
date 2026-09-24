from storage.database.base import Database
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
)


class ResearchDirector:
    """
    Coordinates a research request from planning to execution.

    The director does not perform the research itself.
    It decides what needs to happen and coordinates other components.
    """

    def __init__(self, database: Database):
        self.database = database

    def create_plan(self, question: str) -> ResearchPlan:
        """
        Create an initial research plan for a question.

        This is intentionally rule-based for now.
        An LLM planner will replace this logic later.
        """

        plan = ResearchPlan(
            question=question,
            objectives=[
                "Understand the research question",
                "Gather relevant evidence",
                "Evaluate the evidence",
                "Produce a reproducible conclusion",
            ],
            hypotheses=[],
            required_capabilities=[
                "literature_search",
                "evidence_analysis",
            ],
            experiments=[],
            success_criteria=[
                "Relevant evidence is collected",
                "Evidence is evaluated",
                "Conclusion is reproducible",
            ],
        )

        self.database.save_research_plan(plan)

        return plan

    def start_run(self, run_id: str, question: str) -> ResearchRun:
        """
        Create and persist a new research run.
        """

        run = ResearchRun(
            run_id=run_id,
            research_question=question,
            status="planned",
        )

        self.database.save_research_run(run)

        return run

    def create_tasks(self, plan: ResearchPlan) -> list[ResearchTask]:
        """
        Convert a research plan into an executable task graph.
        """

        tasks = [
            ResearchTask(
                task_id="task_001",
                description="Search and collect relevant research literature",
                task_type="literature_search",
                required_tools=["literature_search"],
            ),

            ResearchTask(
                task_id="task_002",
                description="Extract and organize evidence from the collected literature",
                task_type="evidence_collection",
                dependencies=["task_001"],
            ),

            ResearchTask(
                task_id="task_003",
                description="Analyze the collected evidence",
                task_type="evidence_analysis",
                dependencies=["task_002"],
                required_tools=["evidence_analyzer"],
            ),

            ResearchTask(
                task_id="task_004",
                description="Evaluate the strength and limitations of the evidence",
                task_type="evaluation",
                dependencies=["task_003"],
            ),

            ResearchTask(
                task_id="task_005",
                description="Produce a reproducible research conclusion",
                task_type="conclusion",
                dependencies=["task_004"],
            ),
        ]

        for task in tasks:
            self.database.save_research_task(task)

        return tasks