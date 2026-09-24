from agents.llm_client import LLMClient
from agents.research_director import ResearchDirector
from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase


def main():
    database = InMemoryDatabase()

    llm_client = LLMClient()

    director = ResearchDirector(
        database=database,
        llm_client=llm_client,
    )

    question = "Does chunk size affect RAG performance?"

    # Create research plan using the LLM
    plan = director.create_plan(question)

    # Create research run
    run = director.start_run(
        run_id="run_001",
        question=question,
    )

    # Convert plan into executable tasks
    tasks = director.create_tasks(plan)

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

    print("\nResearch Tasks:")

    for task in tasks:
        print(f"\n[{task.task_id}] {task.task_type}")
        print(f"  {task.description}")
        print(f"  Dependencies: {task.dependencies}")
        print(f"  Required Tools: {task.required_tools}")

    print("\nDatabase State:")
    print(f"Plans: {len(database.research_plans)}")
    print(f"Runs: {len(database.research_runs)}")
    print(f"Tasks: {len(database.research_tasks)}")

    print("\n=== TASK EXECUTION ===")

    executor = TaskExecutor(
        tasks=tasks,
        database=database,
        run_id=run.run_id,
    )

    executor.run()

    print("\n=== FINAL RUN STATE ===")

    final_run = database.get_research_run(run.run_id)

    if final_run is not None:
        print(f"Run ID: {final_run.run_id}")
        print(f"Status: {final_run.status}")
        print(f"Completed Tasks: {final_run.completed_tasks}")
        print(f"Failed Tasks: {final_run.failed_tasks}")

    print("\nAll research tasks completed.")


if __name__ == "__main__":
    main()
