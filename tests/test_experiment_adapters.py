import pytest

from agents.experiment_adapters import (
    build_default_experiment_adapters,
    retrieval_benchmark_adapter,
)


def corpus():
    return [
        {
            "memory_id": "memory_rag",
            "run_id": "run_old",
            "task_id": "task_rag",
            "research_question": "How does RAG retrieval work?",
            "content": "Dense retrieval ranks relevant RAG documents.",
            "tags": ["rag", "retrieval", "dense"],
        },
        {
            "memory_id": "memory_sql",
            "run_id": "run_old",
            "task_id": "task_sql",
            "research_question": "How do SQL indexes work?",
            "content": "Database indexes improve query planning.",
            "tags": ["database", "sql"],
        },
        {
            "memory_id": "memory_vector",
            "run_id": "run_old",
            "task_id": "task_vector",
            "research_question": "How do vector databases work?",
            "content": "Vector search retrieves semantically similar documents.",
            "tags": ["vector", "retrieval"],
        },
    ]


def test_retrieval_benchmark_measures_real_retriever_and_returns_metrics():
    result = retrieval_benchmark_adapter(
        parameters={
            "memories": corpus(),
            "queries": [
                {
                    "query": "RAG retrieval",
                    "relevant_memory_ids": ["memory_rag"],
                },
                {
                    "query": "SQL database indexes",
                    "relevant_memory_ids": ["memory_sql"],
                },
            ],
            "limit": 2,
        },
        random_seed=42,
    )

    assert result["metrics"] == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
    }
    assert result["outputs"]["query_count"] == 2
    assert result["outputs"]["queries"][0]["retrieved_memory_ids"][0] == "memory_rag"


def test_default_adapter_registry_exposes_only_trusted_retrieval_benchmark():
    adapters = build_default_experiment_adapters()

    assert set(adapters) == {"retrieval_benchmark"}
    assert adapters["retrieval_benchmark"] is retrieval_benchmark_adapter


@pytest.mark.parametrize(
    "parameters",
    [
        {"memories": [], "queries": [{"query": "rag", "relevant_memory_ids": ["x"]}]},
        {"memories": corpus(), "queries": []},
        {
            "memories": corpus(),
            "queries": [{"query": "rag", "relevant_memory_ids": ["not-present"]}],
        },
        {
            "memories": corpus(),
            "queries": [{"query": "rag", "relevant_memory_ids": ["memory_rag"]}],
            "source_code": "print('unsafe')",
        },
    ],
)
def test_retrieval_benchmark_rejects_invalid_or_untrusted_inputs(parameters):
    with pytest.raises(ValueError):
        retrieval_benchmark_adapter(parameters=parameters, random_seed=None)


def test_retrieval_benchmark_rejects_duplicate_memory_ids():
    memories = corpus()
    memories.append(dict(memories[0]))

    with pytest.raises(ValueError, match="Duplicate memory_id"):
        retrieval_benchmark_adapter(
            parameters={
                "memories": memories,
                "queries": [{"query": "RAG", "relevant_memory_ids": ["memory_rag"]}],
            },
            random_seed=None,
        )
