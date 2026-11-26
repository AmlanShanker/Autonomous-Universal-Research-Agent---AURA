import json

import pytest

from agents.research_director import ResearchDirector
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchPlan, ResearchTask


QUESTION = "Does chunk size affect RAG performance?"


def valid_plan():
    return {
        "question": QUESTION,
        "objectives": ["Compare retrieval quality across chunk sizes."],
        "hypotheses": ["Smaller chunks improve retrieval precision."],
        "required_capabilities": ["literature_search"],
        "experiments": ["Evaluate fixed chunk sizes on a benchmark."],
        "success_criteria": ["Report precision and recall for each size."],
    }


def valid_tasks():
    return {
        "tasks": [
            {
                "task_id": "task_001",
                "description": "Review relevant RAG literature.",
                "task_type": "literature_search",
                "dependencies": [],
                "required_tools": ["literature_search"],
            },
            {
                "task_id": "task_002",
                "description": "Summarize the evidence and conclude.",
                "task_type": "conclusion",
                "dependencies": ["task_001"],
                "required_tools": [],
            },
        ]
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_director(responses, attempts=3):
    llm = FakeLLM(responses)
    database = InMemoryDatabase()
    return ResearchDirector(
        database=database,
        llm_client=llm,
        max_generation_attempts=attempts,
    ), database, llm


def test_json_parser_accepts_fenced_json_and_surrounding_prose():
    director, _, _ = make_director([])
    parsed = director._parse_json(
        'Here is the result:\n```json\n{"ok": true}\n```'
    )
    assert parsed == {"ok": True}


def test_create_plan_retries_invalid_output_then_persists_valid_plan():
    director, database, llm = make_director([
        '{"question": "wrong"}',
        json.dumps(valid_plan()),
    ])

    plan = director.create_plan(QUESTION)

    assert plan.question == QUESTION
    assert len(llm.prompts) == 2
    assert "previous response was invalid" in llm.prompts[1]
    assert len(database.research_plans) == 1


def test_create_plan_rejects_missing_success_criteria():
    payload = valid_plan()
    payload["success_criteria"] = []
    director, database, _ = make_director([json.dumps(payload)], attempts=1)

    with pytest.raises(RuntimeError, match="success criterion"):
        director.create_plan(QUESTION)

    assert database.research_plans == {}


def test_create_plan_does_not_retry_beyond_configured_attempts():
    director, database, llm = make_director(["not json"] * 4, attempts=2)

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        director.create_plan(QUESTION)

    assert len(llm.prompts) == 2
    assert database.research_plans == {}


def test_create_tasks_retries_invalid_graph_then_persists_valid_tasks():
    invalid = valid_tasks()
    invalid["tasks"][0]["dependencies"] = ["missing"]
    director, database, llm = make_director([
        json.dumps(invalid),
        json.dumps(valid_tasks()),
    ])
    plan = ResearchPlan(**valid_plan())

    tasks = director.create_tasks(plan)

    assert [task.task_id for task in tasks] == ["task_001", "task_002"]
    assert len(llm.prompts) == 2
    assert len(database.research_tasks) == 2


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda tasks: setattr(tasks[0], "task_id", ""), "cannot be empty"),
        (lambda tasks: setattr(tasks[0], "description", " "), "description"),
        (lambda tasks: setattr(tasks[0], "dependencies", ["task_001"]), "itself"),
        (lambda tasks: setattr(tasks[0], "dependencies", ["missing"]), "unknown task"),
        (lambda tasks: setattr(tasks[0], "dependencies", ["task_002"]), "cycle"),
        (lambda tasks: setattr(tasks[0], "dependencies", ["task_001", "task_001"]), "duplicate"),
    ],
)
def test_task_graph_validation_rejects_invalid_graphs(mutate, message):
    director, _, _ = make_director([])
    tasks = [ResearchTask(**item) for item in valid_tasks()["tasks"]]
    mutate(tasks)

    with pytest.raises(ValueError, match=message):
        director._validate_task_graph(tasks)


def test_task_graph_validation_rejects_duplicate_task_ids():
    director, _, _ = make_director([])
    tasks = [ResearchTask(**item) for item in valid_tasks()["tasks"]]
    tasks[1].task_id = tasks[0].task_id

    with pytest.raises(ValueError, match="unique"):
        director._validate_task_graph(tasks)


def test_empty_research_question_is_rejected_before_calling_llm():
    director, _, llm = make_director([])

    with pytest.raises(ValueError, match="cannot be empty"):
        director.create_plan("   ")

    assert llm.prompts == []


def test_create_tasks_rejects_placeholder_dataset_url_and_regenerates_without_download():
    invalid = valid_tasks()
    invalid["tasks"][0]["required_tools"] = ["dataset_download"]
    invalid["tasks"][0]["tool_inputs"] = {
        "dataset_download": {
            "source_url": "https://example.com/datasets/enterprise_ai_cost_benchmarks.json",
            "destination_path": "enterprise_ai_cost_benchmarks.json",
        }
    }
    director, database, llm = make_director([
        json.dumps(invalid),
        json.dumps(valid_tasks()),
    ])

    tasks = director.create_tasks(ResearchPlan(**valid_plan()))

    assert [task.task_id for task in tasks] == ["task_001", "task_002"]
    assert len(llm.prompts) == 2
    assert "invalid or placeholder dataset URL" in llm.prompts[1]
    assert all("dataset_download" not in task.required_tools for task in tasks)
    assert len(database.research_tasks) == 2


@pytest.mark.parametrize(
    "source_url",
    [
        "https://example.com/data.json",
        "https://example.org/data.json",
        "https://datasets.example.net/data.json",
        "file:///tmp/data.json",
        "not-a-url",
    ],
)
def test_task_graph_validation_rejects_unverified_dataset_download_urls(source_url):
    director, _, _ = make_director([])
    payload = valid_tasks()
    payload["tasks"][0]["required_tools"] = ["dataset_download"]
    payload["tasks"][0]["tool_inputs"] = {
        "dataset_download": {
            "source_url": source_url,
            "destination_path": "dataset.json",
        }
    }
    tasks = [ResearchTask(**item) for item in payload["tasks"]]

    with pytest.raises(ValueError, match="invalid or placeholder dataset URL"):
        director._validate_task_graph(tasks)
