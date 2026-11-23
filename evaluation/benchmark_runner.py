from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from agents.experiment_adapters import retrieval_benchmark_adapter


DEFAULT_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
_REQUIRED_DATASET_FIELDS = {
    "benchmark_id", "version", "description", "k",
    "minimum_metrics", "memories", "queries",
}
_SUPPORTED_METRICS = {
    "precision_at_k", "recall_at_k", "mean_reciprocal_rank", "ndcg_at_k",
}


def evaluate_dataset(path: str | Path) -> dict[str, Any]:
    """Run the production memory retriever against one versioned JSON dataset."""
    dataset_path = Path(path)
    raw = dataset_path.read_bytes()
    try:
        dataset = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid benchmark JSON at {dataset_path}: {error}") from error

    if not isinstance(dataset, dict) or set(dataset) != _REQUIRED_DATASET_FIELDS:
        raise ValueError(
            "Benchmark dataset must contain exactly: "
            + ", ".join(sorted(_REQUIRED_DATASET_FIELDS))
        )
    if not isinstance(dataset["benchmark_id"], str) or not dataset["benchmark_id"].strip():
        raise ValueError("benchmark_id must be a non-empty string.")
    if not isinstance(dataset["version"], str) or not dataset["version"].strip():
        raise ValueError("version must be a non-empty string.")
    if not isinstance(dataset["minimum_metrics"], dict):
        raise ValueError("minimum_metrics must be an object.")
    unknown_metrics = set(dataset["minimum_metrics"]) - _SUPPORTED_METRICS
    if unknown_metrics:
        raise ValueError("Unsupported metric threshold(s): " + ", ".join(sorted(unknown_metrics)))
    for name, threshold in dataset["minimum_metrics"].items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {name} must be a number in [0, 1].")

    result = retrieval_benchmark_adapter(
        parameters={
            "memories": dataset["memories"],
            "queries": dataset["queries"],
            "limit": dataset["k"],
        },
        random_seed=None,
    )
    metrics = result["metrics"]
    threshold_results = {
        name: {
            "minimum": float(minimum),
            "actual": float(metrics[name]),
            "passed": metrics[name] >= minimum,
        }
        for name, minimum in dataset["minimum_metrics"].items()
    }
    return {
        "benchmark_id": dataset["benchmark_id"],
        "version": dataset["version"],
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        "metrics": metrics,
        "thresholds": threshold_results,
        "passed": all(item["passed"] for item in threshold_results.values()),
        "outputs": result["outputs"],
    }


def run_benchmarks(benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR) -> dict[str, Any]:
    """Evaluate every versioned JSON benchmark in a directory in stable order."""
    paths = sorted(Path(benchmark_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"No benchmark JSON datasets found in {benchmark_dir}.")
    results = [evaluate_dataset(path) for path in paths]
    return {
        "suite": "aura-research-evaluation",
        "dataset_count": len(results),
        "passed": all(item["passed"] for item in results),
        "benchmarks": results,
    }
