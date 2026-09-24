from storage.database.memory import InMemoryDatabase
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
)


def main():
    database = InMemoryDatabase()

    plan = ResearchPlan(
        question="Does chunk size affect RAG performance?",
        objectives=[
            "Compare different chunk sizes",
            "Measure retrieval performance"
        ],
        hypotheses=[
            "Moderate chunk sizes may improve retrieval accuracy"
        ],
        required_capabilities=[
            "dataset_loader",
            "embedding_model",
            "retrieval_evaluator"
        ],
        experiments=[
            "Evaluate 256-token chunks",
            "Evaluate 512-token chunks",
            "Evaluate 1024-token chunks"
        ],
        success_criteria=[
            "Experiments complete successfully",
            "Results are reproducible"
        ]
    )

    task = ResearchTask(
        task_id="task_001",
        description="Load the research dataset",
        task_type="data_loading",
        required_tools=["dataset_loader"]
    )

    run = ResearchRun(
        run_id="run_001",
        research_question=plan.question,
        tasks=[task.task_id]
    )

    tool = ToolDefinition(
        tool_id="tool_001",
        name="dataset_loader",
        description="Loads and validates a research dataset.",
        input_schema={
            "path": "string"
        },
        output_schema={
            "dataset": "object"
        },
        tests=[
            "valid_dataset",
            "missing_file",
            "invalid_format"
        ]
    )

    database.save_research_plan(plan)
    database.save_research_task(task)
    database.save_research_run(run)
    database.save_tool(tool)

    print("=== DATABASE TEST ===")
    print(f"Research plans: {len(database.research_plans)}")
    print(f"Research tasks: {len(database.research_tasks)}")
    print(f"Research runs: {len(database.research_runs)}")
    print(f"Tools: {len(database.tools)}")


if __name__ == "__main__":
    main()