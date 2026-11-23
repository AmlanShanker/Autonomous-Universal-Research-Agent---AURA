import json

import pytest

from evaluation.benchmark_runner import DEFAULT_BENCHMARK_DIR, evaluate_dataset, run_benchmarks


def test_all_versioned_research_benchmarks_pass_their_declared_thresholds():
    result = run_benchmarks()

    assert result["suite"] == "aura-research-evaluation"
    assert result["dataset_count"] >= 2
    assert result["passed"] is True
    assert all(item["passed"] for item in result["benchmarks"])
    assert all(
        {"precision_at_k", "recall_at_k", "mean_reciprocal_rank", "ndcg_at_k"}
        <= set(item["metrics"])
        for item in result["benchmarks"]
    )


def test_benchmark_output_is_reproducible_for_the_same_dataset():
    dataset = DEFAULT_BENCHMARK_DIR / "retrieval_core_v1.json"

    first = evaluate_dataset(dataset)
    second = evaluate_dataset(dataset)

    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["metrics"] == second["metrics"]
    assert first["outputs"] == second["outputs"]


def test_benchmark_dataset_rejects_unknown_top_level_fields(tmp_path):
    dataset = {
        "benchmark_id": "invalid",
        "version": "1",
        "description": "invalid",
        "k": 1,
        "minimum_metrics": {},
        "memories": [],
        "queries": [],
        "execute": "not allowed",
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        evaluate_dataset(path)


def test_benchmark_suite_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No benchmark JSON datasets"):
        run_benchmarks(tmp_path)
