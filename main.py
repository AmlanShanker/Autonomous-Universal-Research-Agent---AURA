from agents.llm_client import LLMClient
from agents.research_director import ResearchDirector
from agents.task_executor import TaskExecutor

from storage.database.memory import InMemoryDatabase

from tools.registry import ToolRegistry


def main():
    # -----------------------------
    # Database
    # -----------------------------

    database = InMemoryDatabase()

    # -----------------------------
    # LLM
    # -----------------------------

    llm_client = LLMClient()

    # -----------------------------
    # Research Director
    # -----------------------------

    director = ResearchDirector(
        database=database,
        llm_client=llm_client,
    )

    # -----------------------------
    # Research Question
    # -----------------------------

    question = "Does chunk size affect RAG performance?"

    # -----------------------------
    # Create Research Plan
    # -----------------------------

    plan = director.create_plan(
        question
    )

    # -----------------------------
    # Create Research Run
    # -----------------------------

    run = director.start_run(
        run_id="run_001",
        question=question,
    )

    # -----------------------------
    # Create Research Tasks
    # -----------------------------

    tasks = director.create_tasks(
        plan
    )

    # -----------------------------
    # Tool Registry
    # -----------------------------

    tool_registry = ToolRegistry()

    print("=== AURA RESEARCH DIRECTOR ===")

    print("\nResearch Question:")
    print(plan.question)

    print("\nObjectives:")
    for objective in plan.objectives:
        print(f"- {objective}")

    print("\nHypotheses:")
    for hypothesis in plan.hypotheses:
        print(f"- {hypothesis}")

    print("\nRequired Capabilities:")
    for capability in plan.required_capabilities:
        print(f"- {capability}")

    print("\nExperiments:")
    for experiment in plan.experiments:
        print(f"- {experiment}")

    print("\nSuccess Criteria:")
    for criterion in plan.success_criteria:
        print(f"- {criterion}")

    print("\nResearch Run:")
    print(f"ID: {run.run_id}")
    print(f"Status: {run.status}")

    # -----------------------------
    # Registered Tools
    # -----------------------------

    print("\nRegistered Tools:")

    for tool_name in tool_registry.list_tools():
        print(f"- {tool_name}")

    # -----------------------------
    # Research Tasks
    # -----------------------------

    print("\nResearch Tasks:")

    for task in tasks:

        print(
            f"\n[{task.task_id}] "
            f"{task.task_type}"
        )

        print(
            f"  {task.description}"
        )

        print(
            f"  Dependencies: "
            f"{task.dependencies}"
        )

        print(
            f"  Required Tools: "
            f"{task.required_tools}"
        )

    # -----------------------------
    # Database State
    # -----------------------------

    print("\nDatabase State:")

    print(
        f"Plans: "
        f"{len(database.research_plans)}"
    )

    print(
        f"Runs: "
        f"{len(database.research_runs)}"
    )

    print(
        f"Tasks: "
        f"{len(database.research_tasks)}"
    )

    # -----------------------------
    # Task Execution
    # -----------------------------

    print("\n=== TASK EXECUTION ===")

    executor = TaskExecutor(
        tasks=tasks,
        database=database,
        run_id=run.run_id,
        tool_registry=tool_registry,
    )

    executor.run()

    # -----------------------------
    # Final Run State
    # -----------------------------

    print("\n=== FINAL RUN STATE ===")

    final_run = database.get_research_run(
        run.run_id
    )

    if final_run is not None:

        print(
            f"Run ID: "
            f"{final_run.run_id}"
        )

        print(
            f"Status: "
            f"{final_run.status}"
        )

        print(
            f"Completed Tasks: "
            f"{final_run.completed_tasks}"
        )

        print(
            f"Failed Tasks: "
            f"{final_run.failed_tasks}"
        )

    print("\n=== AURA RUN COMPLETE ===")


if __name__ == "__main__":
    main()
