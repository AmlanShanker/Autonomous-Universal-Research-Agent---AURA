import json

import pytest
from pydantic import ValidationError

from agents.research_director import ResearchDirector
from agents.task_executor import TaskExecutor
from storage.database.memory import InMemoryDatabase
from storage.models.experiment import ExperimentSpec
from storage.models.research import ResearchPlan, ResearchTask


def valid_experiment():
    return {
        "hypothesis": "The revised retriever improves recall.",
        "method": "Run the retrieval benchmark with fixed parameters.",
        "adapter_name": "retrieval_benchmark",
        "parameters": {"chunk_size": 256},
        "success_criteria": [
            {"metric": "recall", "operator": "gte", "threshold": 0.8}
        ],
        "repetitions": 2,
        "random_seed": 17,
    }


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return json.dumps(self.response)


def test_experiment_spec_rejects_code_and_unknown_fields():
    payload = valid_experiment()
    payload["source_code"] = "print('unsafe')"

    with pytest.raises(ValidationError):
        ExperimentSpec(**payload)


def test_research_director_accepts_only_registered_experiment_adapters():
    plan = ResearchPlan(
        question="Does chunk size affect retrieval?",
        objectives=["Compare retrieval quality."],
        success_criteria=["Measure recall."],
    )
    task_payload = {
        "tasks": [{
            "task_id": "task_experiment",
            "description": "Run the retrieval benchmark.",
            "task_type": "experiment",
            "experiments": [valid_experiment()],
        }]
    }
    database = InMemoryDatabase()
    director = ResearchDirector(
        database=database,
        llm_client=FakeLLM(task_payload),
        available_experiment_adapters=["retrieval_benchmark"],
    )

    tasks = director.create_tasks(plan)

    assert len(tasks) == 1
    assert tasks[0].experiments[0].adapter_name == "retrieval_benchmark"
    assert database.get_research_task("task_experiment") == tasks[0]


def test_research_director_rejects_unregistered_adapter_before_persistence():
    plan = ResearchPlan(
        question="Does chunk size affect retrieval?",
        objectives=["Compare retrieval quality."],
        success_criteria=["Measure recall."],
    )
    experiment = valid_experiment()
    experiment["adapter_name"] = "model_generated_adapter"
    database = InMemoryDatabase()
    director = ResearchDirector(
        database=database,
        llm_client=FakeLLM({
            "tasks": [{
                "task_id": "task_experiment",
                "description": "Run an experiment.",
                "task_type": "experiment",
                "experiments": [experiment],
            }]
        }),
        available_experiment_adapters=["retrieval_benchmark"],
        max_generation_attempts=1,
    )

    with pytest.raises(RuntimeError, match="unavailable experiment adapter"):
        director.create_tasks(plan)

    assert database.research_tasks == {}


class FakeToolRegistry:
    pass


def test_task_executor_runs_trusted_experiments_and_exposes_results_to_dependents():
    database = InMemoryDatabase()
    calls = []

    def benchmark(*, parameters, random_seed):
        calls.append((parameters, random_seed))
        return {
            "metrics": {"recall": 0.85},
            "outputs": {"evaluated_documents": 100},
        }

    experiment_task = ResearchTask(
        task_id="task_experiment",
        description="Run the retrieval benchmark.",
        task_type="experiment",
        experiments=[valid_experiment()],
    )
    conclusion_task = ResearchTask(
        task_id="task_conclusion",
        description="Interpret the benchmark result.",
        task_type="conclusion",
        dependencies=["task_experiment"],
    )
    executor = TaskExecutor(
        tasks=[experiment_task, conclusion_task],
        database=database,
        run_id="run_001",
        tool_registry=FakeToolRegistry(),
        experiment_adapters={"retrieval_benchmark": benchmark},
    )

    assert executor.execute_task(experiment_task) is True

    stored_experiments = database.list_experiments_for_run("run_001")
    assert len(stored_experiments) == 1
    assert stored_experiments[0].task_id == "task_experiment"
    assert calls == [({"chunk_size": 256}, 17), ({"chunk_size": 256}, 18)]

    dependency_context = executor.build_task_context(conclusion_task)["dependencies"]
    assert len(dependency_context) == 1
    assert dependency_context[0]["tool_id"] == "experiment:retrieval_benchmark"
    assert dependency_context[0]["status"] == "completed"
    assert dependency_context[0]["output"]["metrics"] == {"recall": 0.85}
    assert dependency_context[0]["output"]["passed"] is True


def test_task_executor_does_not_run_unregistered_experiment_code():
    database = InMemoryDatabase()
    calls = []

    def should_not_run(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Unregistered adapter must not execute.")

    task = ResearchTask(
        task_id="task_experiment",
        description="Run the retrieval benchmark.",
        task_type="experiment",
        experiments=[valid_experiment()],
    )
    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id="run_001",
        tool_registry=FakeToolRegistry(),
        experiment_adapters={},
    )

    assert executor.execute_task(task) is True
    results = list(database.experiment_results.values())
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "No trusted experiment adapter" in results[0].error
    assert calls == []
