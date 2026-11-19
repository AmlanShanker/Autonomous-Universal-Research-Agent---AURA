import hashlib

import pytest
from pydantic import ValidationError

from agents.experiment_runner import ExperimentRunner
from storage.database.memory import InMemoryDatabase
from storage.models.experiment import (
    ExperimentArtifact,
    ExperimentDefinition,
    ExperimentEnvironment,
    ExperimentParameterSnapshot,
    MetricCriterion,
)


def experiment(**overrides):
    values = {
        "experiment_id": "repro_experiment",
        "run_id": "repro_run",
        "task_id": "repro_task",
        "hypothesis": "The adapter produces stable results.",
        "method": "Run the trusted adapter with a fixed seed.",
        "adapter_name": "test_adapter",
        "parameters": {"batch_size": 16},
        "success_criteria": [
            MetricCriterion(metric="score", operator="gte", threshold=0.8)
        ],
        "repetitions": 1,
        "random_seed": 7,
    }
    values.update(overrides)
    return ExperimentDefinition(**values)


def test_runner_persists_immutable_parameter_snapshot_and_increments_changed_version():
    database = InMemoryDatabase()
    adapter = lambda **kwargs: {"metrics": {"score": 0.9}, "outputs": {}}
    runner = ExperimentRunner(database, {"test_adapter": adapter})

    first = runner.run(experiment())
    changed = runner.run(experiment(parameters={"batch_size": 32}))

    first_snapshot = database.get_experiment_parameter_snapshot(first.parameter_snapshot_id)
    changed_snapshot = database.get_experiment_parameter_snapshot(changed.parameter_snapshot_id)
    assert first_snapshot.version == 1
    assert first_snapshot.parameters == {"batch_size": 16}
    assert changed_snapshot.version == 2
    assert changed_snapshot.parameters == {"batch_size": 32}
    assert first_snapshot.snapshot_id != changed_snapshot.snapshot_id
    assert first.result_id != changed.result_id


def test_explicit_parameter_version_is_preserved_and_snapshot_is_immutable():
    database = InMemoryDatabase()
    runner = ExperimentRunner(
        database,
        {"test_adapter": lambda **kwargs: {"metrics": {"score": 0.9}, "outputs": {}}},
    )
    result = runner.run(experiment(parameter_version=4))
    snapshot = database.get_experiment_parameter_snapshot(result.parameter_snapshot_id)

    assert snapshot.version == 4
    with pytest.raises(ValueError, match="immutable"):
        database.save_experiment_parameter_snapshot(
            ExperimentParameterSnapshot(
                snapshot_id=snapshot.snapshot_id,
                experiment_id=snapshot.experiment_id,
                version=snapshot.version,
                parameters={"batch_size": 99},
                parameters_sha256=hashlib.sha256(b'{"batch_size":99}').hexdigest(),
            )
        )


def test_runner_captures_environment_and_persists_text_artifacts():
    database = InMemoryDatabase()
    adapter = lambda **kwargs: {
        "metrics": {"score": 0.9},
        "outputs": {"rows": 3},
        "artifacts": [
            {"name": "summary.txt", "content": "three rows evaluated"},
        ],
    }
    definition = experiment(environment=ExperimentEnvironment(
        adapter_version="2.1.0",
        code_revision="abc123",
        package_versions={"numpy": "2.0.0"},
    ))

    result = ExperimentRunner(database, {"test_adapter": adapter}).run(definition)
    artifacts = database.list_experiment_artifacts(result.result_id)

    assert result.status == "completed"
    assert result.parameter_snapshot_id is not None
    assert result.environment.python_version
    assert result.environment.implementation
    assert result.environment.platform
    assert result.environment.adapter_version == "2.1.0"
    assert result.environment.package_versions == {"numpy": "2.0.0"}
    assert len(artifacts) == 1
    assert artifacts[0].artifact_id in result.artifact_ids
    assert artifacts[0].content == "three rows evaluated"
    assert artifacts[0].sha256 == hashlib.sha256(b"three rows evaluated").hexdigest()
    assert database.get_experiment_artifact(artifacts[0].artifact_id) == artifacts[0]


def test_runner_rejects_oversized_artifact_and_persists_failure():
    database = InMemoryDatabase()
    adapter = lambda **kwargs: {
        "metrics": {"score": 0.9},
        "outputs": {},
        "artifacts": [{"name": "large.txt", "content": "x" * 1_000_001}],
    }

    result = ExperimentRunner(database, {"test_adapter": adapter}).run(experiment())

    assert result.status == "failed"
    assert "1 MB" in result.error
    assert result.artifact_ids == []


def test_artifact_model_rejects_incorrect_digest():
    with pytest.raises(ValidationError, match="digest"):
        ExperimentArtifact(
            artifact_id="artifact_1",
            experiment_id="experiment_1",
            result_id="result_1",
            name="notes.txt",
            content="actual",
            sha256="0" * 64,
        )
