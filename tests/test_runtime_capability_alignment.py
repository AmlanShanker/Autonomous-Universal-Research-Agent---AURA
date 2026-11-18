import json

import pytest

from agents.research_director import ResearchDirector
from storage.database.memory import InMemoryDatabase
from tools.registry import build_default_tool_registry


class UnusedLLM:
    def generate(self, prompt: str) -> str:
        raise AssertionError("LLM generation is not needed in this test.")


def test_application_registry_bootstraps_definitions_for_builtin_tools():
    database = InMemoryDatabase()

    registry = build_default_tool_registry(database)

    assert set(registry.list_tools()) == {
        "literature_search",
        "dataset_download",
    }
    assert set(registry.list_definitions()) == {
        "literature_search",
        "dataset_download",
    }

    definition = registry.get_definition("literature_search")
    assert definition is not None
    assert definition.status == "draft"

    validation = registry.validate_tool("literature_search")

    assert validation.valid is True
    assert registry.has_validated("literature_search")
    assert registry.get_executable_implementation("literature_search") is not None
    assert database.get_tool("literature_search") is not None


def test_research_director_rejects_unregistered_executable_tool_names():
    director = ResearchDirector(
        database=InMemoryDatabase(),
        llm_client=UnusedLLM(),
        available_tools=["literature_search", "dataset_download"],
        available_experiment_adapters=["retrieval_benchmark"],
    )
    response = json.dumps({
        "tasks": [{
            "task_id": "task_001",
            "description": "Run a retrieval benchmark.",
            "task_type": "benchmark",
            "dependencies": [],
            "required_tools": ["retrieval_benchmark"],
            "experiments": [],
        }]
    })

    with pytest.raises(ValueError, match="unavailable executable tool"):
        director._parse_and_validate_tasks(response)


def test_research_director_accepts_registered_tools_and_trusted_adapters_separately():
    director = ResearchDirector(
        database=InMemoryDatabase(),
        llm_client=UnusedLLM(),
        available_tools=["literature_search", "dataset_download"],
        available_experiment_adapters=["retrieval_benchmark"],
    )
    response = json.dumps({
        "tasks": [{
            "task_id": "task_001",
            "description": "Benchmark the existing research-memory retriever.",
            "task_type": "benchmark",
            "dependencies": [],
            "required_tools": [],
            "experiments": [{
                "hypothesis": "The retriever returns relevant memories.",
                "method": "Run the trusted retrieval benchmark adapter.",
                "adapter_name": "retrieval_benchmark",
                "parameters": {
                    "memories": [{
                        "memory_id": "m1",
                        "research_question": "What is hybrid retrieval?",
                        "content": "Hybrid retrieval combines lexical and semantic signals.",
                    }],
                    "queries": [{
                        "query": "hybrid retrieval",
                        "relevant_memory_ids": ["m1"],
                    }],
                    "limit": 1,
                },
                "success_criteria": [{
                    "metric": "mean_reciprocal_rank",
                    "operator": "gte",
                    "threshold": 1.0,
                }],
                "repetitions": 1,
                "random_seed": 42,
            }],
        }]
    })

    tasks = director._parse_and_validate_tasks(response)

    assert tasks[0].required_tools == []
    assert tasks[0].experiments[0].adapter_name == "retrieval_benchmark"
