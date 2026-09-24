import json

from storage.database.base import Database
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
)


class ResearchDirector:
    """
    Coordinates a research request from planning to execution.

    The director uses an LLM to generate the research plan,
    then converts that plan into executable research tasks.
    """

    def __init__(
        self,
        database: Database,
        llm_client,
    ):
        self.database = database
        self.llm_client = llm_client

    def create_plan(self, question: str) -> ResearchPlan:
        """
        Generate a research plan using the LLM.
        """

        prompt = f"""
You are the Research Director of an autonomous research system called AURA.

Create a structured research plan for this question:

{question}

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "question": "the original research question",
    "objectives": [
        "objective 1",
        "objective 2",
        "objective 3"
    ],
    "hypotheses": [
        "hypothesis 1",
        "hypothesis 2"
    ],
    "required_capabilities": [
        "capability 1",
        "capability 2"
    ],
    "experiments": [
        "experiment 1",
        "experiment 2"
    ],
    "success_criteria": [
        "criterion 1",
        "criterion 2"
    ]
}}

Requirements:

- Keep the original question unchanged.
- Create concrete research objectives.
- Include testable hypotheses when appropriate.
- Identify capabilities or tools AURA will need.
- Propose experiments when useful.
- Define measurable success criteria.
- Do not include markdown.
- Do not include explanations outside the JSON.
"""

        response = self.llm_client.generate(prompt)

        try:
            plan_data = self._parse_json(response)
            plan = ResearchPlan(**plan_data)

        except Exception as error:
            raise RuntimeError(
                f"Failed to create a valid research plan from LLM response: {error}"
            ) from error

        self.database.save_research_plan(plan)

        return plan

    def _parse_json(self, response: str) -> dict:
        """
        Parse JSON returned by the LLM.

        Also handles JSON wrapped in markdown code fences.
        """

        response = response.strip()

        if response.startswith("```"):
            lines = response.splitlines()

            if lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response = "\n".join(lines).strip()

        return json.loads(response)

    def start_run(
        self,
        run_id: str,
        question: str,
    ) -> ResearchRun:
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

    def create_tasks(
        self,
        plan: ResearchPlan,
    ) -> list[ResearchTask]:
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
