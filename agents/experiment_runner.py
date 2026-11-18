import math
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from storage.database.base import Database
from storage.models.experiment import ExperimentDefinition, ExperimentResult


ExperimentAdapter = Callable[..., Mapping[str, Any]]


class ExperimentRunner:
    """
    Runs experiments only through explicitly injected, trusted adapters.

    Model-generated experiment specifications remain declarative. This
    runner never imports generated modules, executes source strings, or
    installs dependencies.
    """

    def __init__(
        self,
        database: Database,
        adapters: Mapping[str, ExperimentAdapter],
    ):
        self.database = database
        self.adapters = dict(adapters)

    def run(self, experiment: ExperimentDefinition) -> ExperimentResult:
        """Persist a specification, execute bounded repetitions, and evaluate metrics."""
        started_at = datetime.now(timezone.utc)
        self.database.save_experiment(experiment)
        adapter = self.adapters.get(experiment.adapter_name)

        if adapter is None:
            return self._save_failure(
                experiment,
                started_at,
                f"No trusted experiment adapter registered as '{experiment.adapter_name}'.",
            )

        repetition_metrics: list[dict[str, float]] = []
        repetition_outputs: list[dict[str, Any]] = []

        try:
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
                if not isinstance(metrics, Mapping) or not isinstance(outputs, Mapping):
                    raise ValueError("Adapter result must contain mapping-valued metrics and outputs.")

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

                repetition_metrics.append(normalized_metrics)
                repetition_outputs.append(dict(outputs))

            metric_names = set(repetition_metrics[0])
            if any(set(item) != metric_names for item in repetition_metrics[1:]):
                raise ValueError("Every repetition must return the same metric names.")

            # Use decimal representations of the adapter's numeric values
            # to avoid binary floating-point artifacts in reported means.
            aggregate = {
                name: float(
                    sum(
                        (Decimal(str(item[name])) for item in repetition_metrics),
                        start=Decimal("0"),
                    )
                    / Decimal(len(repetition_metrics))
                )
                for name in sorted(metric_names)
            }
            evaluation = self._evaluate(experiment, aggregate)
            passed = all(evaluation.values())

            result = ExperimentResult(
                result_id=str(uuid.uuid4()),
                experiment_id=experiment.experiment_id,
                run_id=experiment.run_id,
                task_id=experiment.task_id,
                status="completed",
                metrics=aggregate,
                outputs={"repetitions": repetition_outputs},
                evaluation=evaluation,
                passed=passed,
                metadata={
                    "adapter_name": experiment.adapter_name,
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
            return self._save_failure(experiment, started_at, str(error))

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
    ) -> ExperimentResult:
        result = ExperimentResult(
            result_id=str(uuid.uuid4()),
            experiment_id=experiment.experiment_id,
            run_id=experiment.run_id,
            task_id=experiment.task_id,
            status="failed",
            passed=False,
            error=error,
            metadata={"adapter_name": experiment.adapter_name},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self.database.save_experiment_result(result)
        return result
