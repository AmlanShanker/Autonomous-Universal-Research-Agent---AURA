from storage.models.research import (
    ResearchPlan,
    ResearchTask,
    ResearchRun,
    ToolDefinition,
)


def main():
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
        dependencies=[],
        tests=[
            "valid_dataset",
            "missing_file",
            "invalid_format"
        ]
    )

    print("=== RESEARCH PLAN ===")
    print(plan.model_dump_json(indent=2))

    print("\n=== RESEARCH TASK ===")
    print(task.model_dump_json(indent=2))

    print("\n=== RESEARCH RUN ===")
    print(run.model_dump_json(indent=2))

    print("\n=== TOOL DEFINITION ===")
    print(tool.model_dump_json(indent=2))


if __name__ == "__main__":
    main()