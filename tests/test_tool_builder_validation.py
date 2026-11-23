import json

import pytest

from tools.builder import GeneratedTool, ToolBuildRequest, ToolBuilder


def valid_payload():
    return {
        "name": "dataset_analyzer",
        "capability": "dataset_analysis",
        "description": "Analyze a research dataset.",
        "purpose": "Extract reproducible dataset statistics.",
        "inputs": ["dataset_path"],
        "outputs": ["summary"],
        "dependencies": ["pandas"],
        "validation_requirements": ["Test on a fixed fixture dataset."],
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def make_builder(responses, attempts=3):
    llm = FakeLLM(responses)
    return ToolBuilder(llm_client=llm, max_generation_attempts=attempts), llm


def test_build_accepts_fenced_and_prose_wrapped_json():
    builder, _ = make_builder([
        "Specification follows:\n```json\n"
        + json.dumps(valid_payload())
        + "\n```"
    ])
    tool = builder.build(ToolBuildRequest("dataset_analysis", "Need dataset metrics."))
    assert tool.name == "dataset_analyzer"


def test_build_retries_invalid_output_and_then_succeeds():
    builder, llm = make_builder(["not json", json.dumps(valid_payload())])
    tool = builder.build(ToolBuildRequest("dataset_analysis", "Need metrics."))
    assert tool.name == "dataset_analyzer"
    assert len(llm.prompts) == 2
    assert "previous response was invalid" in llm.prompts[1]


def test_build_stops_after_configured_attempts():
    builder, llm = make_builder(["bad"] * 4, attempts=2)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        builder.build(ToolBuildRequest("dataset_analysis", "Need metrics."))
    assert len(llm.prompts) == 2


@pytest.mark.parametrize("field,value,message", [
    ("name", "Bad Name", "snake_case"),
    ("inputs", [], "at least one input"),
    ("outputs", [" "], "non-empty strings"),
    ("dependencies", "pandas", "must be a list"),
    ("validation_requirements", [], "validation requirements"),
    ("inputs", ["dataset_path", "dataset_path"], "duplicates"),
])
def test_build_rejects_invalid_specification_fields(field, value, message):
    payload = valid_payload()
    payload[field] = value
    builder, _ = make_builder([json.dumps(payload)], attempts=1)
    with pytest.raises(RuntimeError, match=message):
        builder.build(ToolBuildRequest("dataset_analysis", "Need metrics."))


def test_build_rejects_capability_mismatch():
    payload = valid_payload()
    payload["capability"] = "other"
    builder, _ = make_builder([json.dumps(payload)], attempts=1)
    with pytest.raises(RuntimeError, match="does not match"):
        builder.build(ToolBuildRequest("dataset_analysis", "Need metrics."))


def test_unknown_fields_are_rejected_including_executable_code():
    payload = valid_payload()
    payload["source_code"] = "import os; os.system('...')"
    builder, _ = make_builder([json.dumps(payload)], attempts=1)
    with pytest.raises(RuntimeError, match="Unexpected"):
        builder.build(ToolBuildRequest("dataset_analysis", "Need metrics."))


def test_conversion_produces_draft_metadata_and_does_not_execute_or_install():
    builder, _ = make_builder([])
    tool = GeneratedTool(**valid_payload())
    definition = builder.to_tool_definition(tool)
    assert definition.status == "draft"
    assert definition.tool_id == "dataset_analyzer"
    assert definition.dependencies == ["pandas"]
    assert definition.input_schema == {"inputs": ["dataset_path"]}


def test_empty_request_reason_is_rejected_before_llm_call():
    builder, llm = make_builder([])
    with pytest.raises(ValueError, match="Reason cannot be empty"):
        builder.build(ToolBuildRequest("dataset_analysis", " "))
    assert llm.prompts == []


def test_builder_defers_llm_client_creation_until_build(monkeypatch):
    calls = []

    class LazyFakeLLM:
        def generate(self, prompt):
            return json.dumps(valid_payload())

    def make_llm_client():
        calls.append(True)
        return LazyFakeLLM()

    monkeypatch.setattr("tools.builder.LLMClient", make_llm_client)
    builder = ToolBuilder()

    assert calls == []
    tool = builder.build(ToolBuildRequest("dataset_analysis", "Need dataset metrics."))

    assert tool.name == "dataset_analyzer"
    assert calls == [True]
