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

    The director uses an LLM to:
    1. Create a research plan.
    2. Create a dynamic research task graph.

    The director does not perform the research itself.
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
You are the Research Director of an autonomous research system
called AURA.

Create a structured research plan for this question:

{question}

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "question": "the original research question",
    "objectives": [
        "objective 1",
        "objective 2"
    ],
    "hypotheses": [
        "hypothesis 1"
    ],
    "required_capabilities": [
        "capability 1"
    ],
    "experiments": [
        "experiment 1"
    ],
    "success_criteria": [
        "criterion 1"
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

    def create_tasks(
        self,
        plan: ResearchPlan,
    ) -> list[ResearchTask]:
        """
        Generate an executable research task graph using the LLM.
        """

        prompt = f"""
You are the task-planning component of an autonomous research
system called AURA.

Create an executable research task graph for this research plan.

Research question:
{plan.question}

Objectives:
{json.dumps(plan.objectives, indent=2)}

Hypotheses:
{json.dumps(plan.hypotheses, indent=2)}

Required capabilities:
{json.dumps(plan.required_capabilities, indent=2)}

Experiments:
{json.dumps(plan.experiments, indent=2)}

Success criteria:
{json.dumps(plan.success_criteria, indent=2)}

Return ONLY valid JSON.

Return exactly this structure:

{{
    "tasks": [
        {{
            "task_id": "task_001",
            "description": "clear description of the task",
            "task_type": "literature_search",
            "dependencies": [],
            "required_tools": ["literature_search"]
        }}
    ]
}}

Rules:

- Create only tasks necessary for the research.
- Tasks must form a directed acyclic graph.
- Every dependency must reference another task_id.
- task_id values must be unique.
- Start task numbering at task_001.
- Use descriptive task types.
- Include literature research when relevant.
- Include experiments when the research requires experiments.
- Include analysis and evaluation when appropriate.
- Include a final conclusion task.
- Do not include markdown.
- Do not include explanations outside the JSON.
"""

        response = self.llm_client.generate(prompt)

        try:
            data = self._parse_json(response)

            raw_tasks = data["tasks"]

            tasks = [
                ResearchTask(**task)
                for task in raw_tasks
            ]

        except Exception as error:
            raise RuntimeError(
                f"Failed to create valid research tasks from LLM response: {error}"
            ) from error

        self._validate_task_graph(tasks)

        for task in tasks:
            self.database.save_research_task(task)

        return tasks

    def _parse_json(self, response: str) -> dict:
        """
        Parse JSON returned by the LLM.

        Handles JSON wrapped in markdown code fences.
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

    def _validate_task_graph(
        self,
        tasks: list[ResearchTask],
    ) -> None:
        """
        Validate the research task graph before execution.
        """

        if not tasks:
            raise ValueError(
                "Research task graph cannot be empty."
            )

        task_ids = {
            task.task_id
            for task in tasks
        }

        if len(task_ids) != len(tasks):
            raise ValueError(
                "Research task IDs must be unique."
            )

        for task in tasks:
            for dependency in task.dependencies:
                if dependency not in task_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on "
                        f"unknown task {dependency}."
                    )

        graph = {
            task.task_id: task.dependencies
            for task in tasks
        }

        visiting = set()
        visited = set()

        def visit(task_id: str):
            if task_id in visiting:
                raise ValueError(
                    "Research task graph contains a dependency cycle."
                )

            if task_id in visited:
                return

            visiting.add(task_id)

            for dependency in graph[task_id]:
                visit(dependency)

            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)

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
