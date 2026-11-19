import hashlib
import json
import math
import platform
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
import sys

from storage.database.base import Database
from storage.models.experiment import (
    ExperimentArtifact,
    ExperimentDefinition,
    ExperimentEnvironment,
    ExperimentParameterSnapshot,
    ExperimentResult,
)


ExperimentAdapter = Callable[..., Mapping[str, Any]]
_MAX_ARTIFACT_BYTES = 1_000_000


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ExperimentRunner:
    """
    Runs experiments only through explicitly injected, trusted adapters.

    Parameter snapshots are immutable and versioned. Runtime details and
    bounded text artifacts are persisted with each result. The runner never
    imports generated modules, executes source strings, or installs dependencies.
    """

    def __init__(
        self,
        database: Database,
        adapters: Mapping[str, ExperimentAdapter],
    ):
        self.database = database
        self.adapters = dict(adapters)

    def _environment_snapshot(self, experiment: ExperimentDefinition) -> ExperimentEnvironment:
        declared = experiment.environment
        return declared.model_copy(update={
            "runtime_python_version": sys.version.split()[0],
            "runtime_implementation": platform.python_implementation(),
            "runtime_platform": platform.platform(),
        })

    def run(self, experiment: ExperimentDefinition) -> ExperimentResult:
        """Persist versioned inputs, execute bounded repetitions, and retain artifacts."""
        started_at = datetime.now(timezone.utc)
        previous_definition = self.database.get_experiment(experiment.experiment_id)

        # Parameter changes create a new immutable version rather than replacing
        # the historical snapshot. Explicitly higher versions are always honored.
        if (
            previous_definition is not None
            and previous_definition.parameters != experiment.parameters
            and experiment.parameter_version <= previous_definition.parameter_version
        ):
            experiment = experiment.model_copy(update={
                "parameter_version": previous_definition.parameter_version + 1
            })

        parameters_json = _canonical_json(experiment.parameters)
        parameters_sha256 = _sha256(parameters_json)
        snapshot_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"aura:experiment-parameters:{experiment.experiment_id}:{experiment.parameter_version}",
        ))
        existing_snapshot = self.database.get_experiment_parameter_snapshot(snapshot_id)
        if existing_snapshot is not None:
            if existing_snapshot.parameters_sha256 != parameters_sha256:
                raise ValueError(
                    "Parameter version is immutable; increment parameter_version to change inputs."
                )
            snapshot = existing_snapshot
        else:
            snapshot = ExperimentParameterSnapshot(
                snapshot_id=snapshot_id,
                experiment_id=experiment.experiment_id,
                version=experiment.parameter_version,
                parameters=deepcopy(experiment.parameters),
                parameters_sha256=parameters_sha256,
            )
            self.database.save_experiment_parameter_snapshot(snapshot)

        environment = self._environment_snapshot(experiment)
        spec_fingerprint = _sha256(_canonical_json({
            "experiment": experiment.model_dump(mode="json"),
            "environment": environment.model_dump(mode="json"),
            "parameters_sha256": parameters_sha256,
        }))
        stable_result_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"aura:experiment-result:{experiment.experiment_id}:{spec_fingerprint}",
        ))

        self.database.save_experiment(experiment)
        existing_result = self.database.get_experiment_result(stable_result_id)
        if existing_result is not None and existing_result.status == "completed":
            return existing_result

        adapter = self.adapters.get(experiment.adapter_name)
        if adapter is None:
            return self._save_failure(
                experiment, started_at,
                f"No trusted experiment adapter registered as '{experiment.adapter_name}'.",
                stable_result_id, snapshot.snapshot_id, environment,
            )

        repetition_metrics: list[dict[str, float]] = []
        repetition_outputs: list[dict[str, Any]] = []

        try:
            repetition_artifacts: list[Mapping[str, Any]] = []
            for index in range(experiment.repetitions):
                seed = (
                    None
                    if experiment.random_seed is None
                    else experiment.random_seed + index
                )
                raw = adapter(
                    parameters=deepcopy(experiment.parameters),
                    random_seed=seed,
                )
                if not isinstance(raw, Mapping):
                    raise ValueError("Experiment adapter must return a mapping.")
                metrics = raw.get("metrics")
                outputs = raw.get("outputs", {})
                artifacts = raw.get("artifacts", [])
                if not isinstance(metrics, Mapping) or not isinstance(outputs, Mapping):
                    raise ValueError("Adapter result must contain mapping-valued metrics and outputs.")
                if not isinstance(artifacts, list):
                    raise ValueError("Adapter artifacts must be a list.")

                normalized_metrics: dict[str, float] = {}
                for name, value in metrics.items():
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("Metric names must be non-empty strings.")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"Metric '{name}' must be numeric.")
                    numeric_value = float(value)
                    if not math.isfinite(numeric_value):
                        raise ValueError(f"Metric '{name}' must be finite.")
                    normalized_metrics[name] = numeric_value

                for artifact in artifacts:
                    if (
                        not isinstance(artifact, Mapping)
                        or set(artifact) - {"name", "content", "media_type"}
                        or not isinstance(artifact.get("name"), str)
                        or not artifact.get("name", "").strip()
                        or not isinstance(artifact.get("content"), str)
                    ):
                        raise ValueError(
                            "Each artifact must contain a non-empty name and text content."
                        )
                    if len(artifact["content"].encode("utf-8")) > _MAX_ARTIFACT_BYTES:
                        raise ValueError("Experiment artifacts cannot exceed 1 MB.")
                    repetition_artifacts.append(artifact)

                repetition_metrics.append(normalized_metrics)
                repetition_outputs.append(dict(outputs))

            if not repetition_metrics:
                raise ValueError("Experiment must produce at least one repetition.")
            metric_names = set(repetition_metrics[0])
            if any(set(item) != metric_names for item in repetition_metrics[1:]):
                raise ValueError("Every repetition must return the same metric names.")

            aggregate = {
                name: float(
                    sum(
                        (Decimal(str(item[name])) for item in repetition_metrics),
                        start=Decimal("0"),
                    ) / Decimal(len(repetition_metrics))
                )
                for name in sorted(metric_names)
            }
            evaluation = self._evaluate(experiment, aggregate)
            passed = all(evaluation.values())

            artifact_ids: list[str] = []
            for raw_artifact in repetition_artifacts:
                content = raw_artifact["content"]
                digest = _sha256(content)
                name = raw_artifact["name"].strip()
                artifact_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"aura:experiment-artifact:{stable_result_id}:{name}:{digest}",
                ))
                if artifact_id in artifact_ids:
                    continue
                artifact = ExperimentArtifact(
                    artifact_id=artifact_id,
                    experiment_id=experiment.experiment_id,
                    result_id=stable_result_id,
                    name=name,
                    media_type=raw_artifact.get("media_type", "text/plain"),
                    content=content,
                    sha256=digest,
                )
                self.database.save_experiment_artifact(artifact)
                if artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)

            result = ExperimentResult(
                result_id=stable_result_id,
                experiment_id=experiment.experiment_id,
                run_id=experiment.run_id,
                task_id=experiment.task_id,
                status="completed",
                parameter_snapshot_id=snapshot.snapshot_id,
                environment=environment,
                artifact_ids=artifact_ids,
                metrics=aggregate,
                outputs={"repetitions": repetition_outputs},
                evaluation=evaluation,
                passed=passed,
                metadata={
                    "adapter_name": experiment.adapter_name,
                    "parameter_version": snapshot.version,
                    "parameters_sha256": snapshot.parameters_sha256,
                    "environment_sha256": _sha256(_canonical_json(environment.model_dump(mode="json"))),
                    "repetitions": experiment.repetitions,
                    "random_seeds": [
                        None if experiment.random_seed is None else experiment.random_seed + i
                        for i in range(experiment.repetitions)
                    ],
                    "aggregation": "arithmetic_mean",
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self.database.save_experiment_result(result)
            return result
        except Exception as error:
            return self._save_failure(
                experiment, started_at, str(error),
                stable_result_id, snapshot.snapshot_id, environment,
            )

    def _evaluate(
        self,
        experiment: ExperimentDefinition,
        metrics: Mapping[str, float],
    ) -> dict[str, bool]:
        outcomes: dict[str, bool] = {}
        for criterion in experiment.success_criteria:
            if criterion.metric not in metrics:
                outcomes[criterion.metric] = False
                continue
            value = metrics[criterion.metric]
            threshold = criterion.threshold
            outcomes[criterion.metric] = {
                "gt": value > threshold,
                "gte": value >= threshold,
                "lt": value < threshold,
                "lte": value <= threshold,
                "eq": value == threshold,
            }[criterion.operator]
        return outcomes

    def _save_failure(
        self,
        experiment: ExperimentDefinition,
        started_at: datetime,
        error: str,
        result_id: str,
        snapshot_id: str,
        environment: ExperimentEnvironment,
    ) -> ExperimentResult:
        result = ExperimentResult(
            result_id=result_id,
            experiment_id=experiment.experiment_id,
            run_id=experiment.run_id,
            task_id=experiment.task_id,
            status="failed",
            parameter_snapshot_id=snapshot_id,
            environment=environment,
            passed=False,
            error=error,
            metadata={
                "adapter_name": experiment.adapter_name,
                "parameter_version": experiment.parameter_version,
            },
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self.database.save_experiment_result(result)
        return result
