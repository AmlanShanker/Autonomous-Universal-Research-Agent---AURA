from collections.abc import Mapping
import math
from typing import Any

from memory.research_memory import ResearchMemoryItem
from memory.research_memory_retriever import ResearchMemoryRetriever
from storage.database.memory import InMemoryDatabase


_MAX_MEMORIES = 500
_MAX_QUERIES = 100
_ALLOWED_MEMORY_FIELDS = {
    "memory_id", "run_id", "task_id", "research_question", "content",
    "memory_type", "tags", "metadata", "confidence", "source_result_id",
    "source_tool_id", "embedding", "created_at", "updated_at", "expires_at",
    "archived",
}


def retrieval_benchmark_adapter(
    *,
    parameters: Mapping[str, Any],
    random_seed: int | None,
) -> dict[str, Any]:
    """
    Evaluate AURA's production research-memory retriever against labeled queries.

    The benchmark corpus and relevance labels are data, never executable code.
    A fresh in-memory database isolates each adapter invocation.
    """
    del random_seed  # Retrieval is deterministic and does not sample randomly.

    if not isinstance(parameters, Mapping):
        raise ValueError("Retrieval benchmark parameters must be a mapping.")
    unexpected = set(parameters) - {"memories", "queries", "limit"}
    if unexpected:
        raise ValueError(
            "Unsupported retrieval benchmark parameter(s): "
            + ", ".join(sorted(str(item) for item in unexpected))
        )

    memories = parameters.get("memories")
    queries = parameters.get("queries")
    limit = parameters.get("limit", 5)

    if not isinstance(memories, list) or not 1 <= len(memories) <= _MAX_MEMORIES:
        raise ValueError(f"memories must contain between 1 and {_MAX_MEMORIES} items.")
    if not isinstance(queries, list) or not 1 <= len(queries) <= _MAX_QUERIES:
        raise ValueError(f"queries must contain between 1 and {_MAX_QUERIES} cases.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer between 1 and 50.")

    database = InMemoryDatabase()
    known_ids: set[str] = set()
    for raw_memory in memories:
        if not isinstance(raw_memory, Mapping):
            raise ValueError("Every memory corpus item must be an object.")
        unknown_fields = set(raw_memory) - _ALLOWED_MEMORY_FIELDS
        if unknown_fields:
            raise ValueError(
                "Unsupported memory field(s): "
                + ", ".join(sorted(str(item) for item in unknown_fields))
            )
        item = ResearchMemoryItem.model_validate(dict(raw_memory))
        if item.memory_id in known_ids:
            raise ValueError(f"Duplicate memory_id '{item.memory_id}'.")
        known_ids.add(item.memory_id)
        database.save_research_memory(item)

    retriever = ResearchMemoryRetriever(database)
    query_results = []
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []
    ndcg_scores = []

    for index, case in enumerate(queries):
        if not isinstance(case, Mapping) or set(case) != {"query", "relevant_memory_ids"}:
            raise ValueError(
                f"Query case {index} must contain exactly query and relevant_memory_ids."
            )
        query = case["query"]
        relevant_ids = case["relevant_memory_ids"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query case {index} has an empty query.")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(item, str) or not item for item in relevant_ids)
        ):
            raise ValueError(f"Query case {index} must define relevant memory IDs.")
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"Query case {index} contains duplicate relevant IDs.")
        unknown_relevant = set(relevant_ids) - known_ids
        if unknown_relevant:
            raise ValueError(
                f"Query case {index} references unknown memory IDs: "
                + ", ".join(sorted(unknown_relevant))
            )

        ranked = retriever.retrieve(query, limit=limit)
        ranked_ids = [item.memory_id for item in ranked]
        relevant_set = set(relevant_ids)
        hits = [memory_id for memory_id in ranked_ids if memory_id in relevant_set]
        denominator = min(limit, len(memories))
        precision = len(hits) / denominator
        recall = len(hits) / len(relevant_set)
        reciprocal_rank = (
            1.0 / (ranked_ids.index(hits[0]) + 1) if hits else 0.0
        )
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, memory_id in enumerate(ranked_ids)
            if memory_id in relevant_set
        )
        ideal_hits = min(limit, len(relevant_set))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(ideal_hits)
        )
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        ndcg_scores.append(ndcg)
        query_results.append({
            "query": query,
            "retrieved_memory_ids": ranked_ids,
            "relevant_memory_ids": list(relevant_ids),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
        })

    count = len(query_results)
    return {
        "metrics": {
            "precision_at_k": sum(precision_scores) / count,
            "recall_at_k": sum(recall_scores) / count,
            "mean_reciprocal_rank": sum(reciprocal_ranks) / count,
            "ndcg_at_k": sum(ndcg_scores) / count,
        },
        "outputs": {
            "k": limit,
            "corpus_size": len(memories),
            "query_count": count,
            "queries": query_results,
        },
    }


def build_default_experiment_adapters():
    """Return only application-owned, explicitly trusted experiment adapters."""
    return {
        "retrieval_benchmark": retrieval_benchmark_adapter,
    }



def describe_default_experiment_adapters() -> dict[str, dict[str, Any]]:
    """Return the declarative input/output contracts for trusted adapters."""
    return {
        "retrieval_benchmark": {
            "purpose": (
                "Benchmark AURA's existing research-memory retriever. "
                "This does not implement BM25, FAISS, dense embeddings, or external indexes."
            ),
            "parameters": {
                "memories": (
                    "Required list of 1-500 research-memory objects. Each object must "
                    "include memory_id, research_question, and content; optional supported "
                    "ResearchMemoryItem fields may also be supplied."
                ),
                "queries": (
                    "Required list of 1-100 objects. Each object must contain exactly "
                    "query (non-empty string) and relevant_memory_ids (non-empty list of "
                    "unique memory IDs present in memories)."
                ),
                "limit": "Optional integer from 1 to 50; defaults to 5.",
            },
            "metrics": [
                "precision_at_k",
                "recall_at_k",
                "mean_reciprocal_rank",
                "ndcg_at_k",
            ],
            "output": (
                "Per-query retrieved IDs and relevance labels, aggregate precision, "
                "recall, mean reciprocal rank, and normalized discounted cumulative gain."
            ),
        }
    }
