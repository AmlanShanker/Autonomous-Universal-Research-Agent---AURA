from agents.experiment_adapters import build_default_experiment_adapters
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
    # Trusted Experiment Adapters
    # -----------------------------

    experiment_adapters = build_default_experiment_adapters()

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
        available_experiment_adapters=list(experiment_adapters),
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
    # Trusted Experiment Adapters
    # -----------------------------

    print("\nTrusted Experiment Adapters:")
    for adapter_name in experiment_adapters:
        print(f"- {adapter_name}")

    # -----------------------------
    # Research Tasks
    # -----------------------------

    print("\nResearch Tasks:")
    for task in tasks:
        print(
            f"\n[{task.task_id}] "
            f"{task.task_type}"
        )
        print(f"  {task.description}")
        print(f"  Dependencies: {task.dependencies}")
        print(f"  Required Tools: {task.required_tools}")
        for experiment in task.experiments:
            print(f"  Experiment adapter: {experiment.adapter_name}")

    # -----------------------------
    # Database State
    # -----------------------------

    print("\nDatabase State:")
    print(f"Plans: {len(database.research_plans)}")
    print(f"Runs: {len(database.research_runs)}")
    print(f"Tasks: {len(database.research_tasks)}")

    # -----------------------------
    # Task Execution
    # -----------------------------

    print("\n=== TASK EXECUTION ===")
    executor = TaskExecutor(
        tasks=tasks,
        database=database,
        run_id=run.run_id,
        tool_registry=tool_registry,
        experiment_adapters=experiment_adapters,
        llm_client=llm_client,
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
        print(f"Run ID: {final_run.run_id}")
        print(f"Status: {final_run.status}")
        print(f"Completed Tasks: {final_run.completed_tasks}")
        print(f"Failed Tasks: {final_run.failed_tasks}")

    print("\n=== AURA RUN COMPLETE ===")


if __name__ == "__main__":
    main()
