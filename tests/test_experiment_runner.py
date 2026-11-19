import math

from agents.experiment_runner import ExperimentRunner
from storage.database.memory import InMemoryDatabase
from storage.models.experiment import ExperimentDefinition, MetricCriterion


def make_experiment(**overrides):
    values = {
        "experiment_id": "experiment_001",
        "run_id": "run_001",
        "task_id": "task_001",
        "hypothesis": "The revised method improves recall.",
        "method": "Run a controlled retrieval benchmark.",
        "adapter_name": "retrieval_benchmark",
        "parameters": {"chunk_size": 256},
        "success_criteria": [
            MetricCriterion(metric="recall", operator="gte", threshold=0.8),
            MetricCriterion(metric="latency_ms", operator="lt", threshold=100),
        ],
        "repetitions": 2,
        "random_seed": 41,
    }
    values.update(overrides)
    return ExperimentDefinition(**values)


def test_runner_executes_bounded_repetitions_and_aggregates_metrics():
    database = InMemoryDatabase()
    calls = []

    def adapter(*, parameters, random_seed):
        calls.append((parameters, random_seed))
        return {
            "metrics": {"recall": 0.8 + (random_seed - 41) * 0.1, "latency_ms": 80},
            "outputs": {"sample_count": 20},
        }

    result = ExperimentRunner(database, {"retrieval_benchmark": adapter}).run(
        make_experiment()
    )

    assert result.status == "completed"
    assert result.metrics == {"latency_ms": 80.0, "recall": 0.85}
    assert result.evaluation == {"recall": True, "latency_ms": True}
    assert result.passed is True
    assert result.metadata["random_seeds"] == [41, 42]
    assert [seed for _, seed in calls] == [41, 42]
    assert all(params == {"chunk_size": 256} for params, _ in calls)
    assert database.get_experiment("experiment_001") is not None
    assert database.get_experiment_result(result.result_id) == result


def test_runner_persists_failed_result_for_unknown_adapter():
    database = InMemoryDatabase()
    result = ExperimentRunner(database, {}).run(make_experiment())

    assert result.status == "failed"
    assert result.passed is False
    assert "No trusted experiment adapter" in result.error
    assert database.get_experiment_result(result.result_id) == result


def test_runner_marks_missing_metrics_as_failed_evaluation():
    database = InMemoryDatabase()
    adapter = lambda **kwargs: {"metrics": {"recall": 0.9}, "outputs": {}}

    result = ExperimentRunner(database, {"retrieval_benchmark": adapter}).run(
        make_experiment(repetitions=1)
    )

    assert result.status == "completed"
    assert result.evaluation == {"recall": True, "latency_ms": False}
    assert result.passed is False


def test_runner_rejects_non_finite_metrics_and_persists_failure():
    database = InMemoryDatabase()
    adapter = lambda **kwargs: {"metrics": {"recall": math.nan}, "outputs": {}}

    result = ExperimentRunner(database, {"retrieval_benchmark": adapter}).run(
        make_experiment(repetitions=1)
    )

    assert result.status == "failed"
    assert "finite" in result.error


def test_runner_requires_consistent_metric_names_across_repetitions():
    database = InMemoryDatabase()
    count = 0

    def adapter(**kwargs):
        nonlocal count
        count += 1
        metrics = {"recall": 0.9} if count == 1 else {"precision": 0.8}
        return {"metrics": metrics, "outputs": {}}

    result = ExperimentRunner(database, {"retrieval_benchmark": adapter}).run(
        make_experiment()
    )

    assert result.status == "failed"
    assert "same metric names" in result.error


def test_experiment_repetition_count_is_bounded():
    try:
        make_experiment(repetitions=11)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected repetitions above the maximum to be rejected.")


def test_experiment_text_fields_cannot_be_blank():
    try:
        make_experiment(hypothesis="  ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected blank hypothesis to be rejected.")


def test_experiment_rejects_non_finite_criterion_threshold():
    from pydantic import ValidationError

    try:
        make_experiment(
            success_criteria=[
                MetricCriterion(metric="recall", operator="gte", threshold=float("nan"))
            ]
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected a non-finite threshold to be rejected.")


def test_experiment_rejects_duplicate_criterion_metrics():
    from pydantic import ValidationError

    try:
        make_experiment(
            success_criteria=[
                MetricCriterion(metric="recall", operator="gte", threshold=0.8),
                MetricCriterion(metric="recall", operator="lt", threshold=0.95),
            ]
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected duplicate criterion metrics to be rejected.")



def test_repeated_experiment_execution_reuses_completed_result():
    database = InMemoryDatabase()
    calls = []

    def adapter(*, parameters, random_seed):
        calls.append(random_seed)
        return {"metrics": {"recall": 0.9, "latency_ms": 50}, "outputs": {"count": 10}}

    runner = ExperimentRunner(database, {"retrieval_benchmark": adapter})
    experiment = make_experiment()
    first = runner.run(experiment)
    second = runner.run(experiment)

    assert first.result_id == second.result_id
    assert second.status == "completed"
    assert len(calls) == experiment.repetitions



def test_changed_experiment_specification_does_not_reuse_stale_result():
    database = InMemoryDatabase()
    calls = []

    def adapter(*, parameters, random_seed):
        calls.append(parameters["chunk_size"])
        return {"metrics": {"recall": 0.9, "latency_ms": 50}, "outputs": {}}

    runner = ExperimentRunner(database, {"retrieval_benchmark": adapter})
    first = runner.run(make_experiment())
    changed = runner.run(make_experiment(parameters={"chunk_size": 512}))

    assert first.result_id == changed.result_id
    assert changed.status == "completed"
    assert calls == [256, 256, 512, 512]
