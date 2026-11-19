import pytest

from tools.execution_sandbox import ToolExecutionPolicy, ToolExecutionSandbox


class FakeTool:
    name = "literature_search"

    def __init__(self, output=None):
        self.output = output if output is not None else {"ok": True}
        self.calls = 0

    def execute(self, **inputs):
        self.calls += 1
        return self.output


def test_execution_sandbox_runs_allowlisted_builtin_and_reports_bounded_usage():
    sandbox = ToolExecutionSandbox()
    tool = FakeTool()

    output, metadata = sandbox.execute(tool, query="retrieval", max_results=3)

    assert output == {"ok": True}
    assert metadata["sandbox_policy"] == "trusted_builtin_v1"
    assert metadata["input_bytes"] > 0
    assert metadata["output_bytes"] > 0
    assert metadata["elapsed_ms"] >= 0


def test_execution_sandbox_rejects_unregistered_or_generated_tools():
    sandbox = ToolExecutionSandbox()

    with pytest.raises(PermissionError, match="not permitted"):
        sandbox.execute(
            type("GeneratedTool", (), {"name": "generated_tool", "execute": lambda self: {}})()
        )


def test_execution_sandbox_rejects_oversized_inputs_before_execution():
    policy = ToolExecutionPolicy(max_input_bytes=8)
    sandbox = ToolExecutionSandbox(policy)
    tool = FakeTool()

    with pytest.raises(ValueError, match="inputs exceed"):
        sandbox.execute(tool, query="this is too large")

    assert tool.calls == 0


def test_execution_sandbox_rejects_oversized_outputs():
    policy = ToolExecutionPolicy(max_output_bytes=8)
    sandbox = ToolExecutionSandbox(policy)
    tool = FakeTool({"payload": "this is too large"})

    with pytest.raises(ValueError, match="output exceeds"):
        sandbox.execute(tool)


def test_execution_sandbox_rejects_non_json_and_non_finite_values():
    sandbox = ToolExecutionSandbox()

    with pytest.raises(ValueError, match="finite JSON"):
        sandbox.execute(FakeTool(), query=float("inf"))


def test_execution_sandbox_rejects_invalid_resource_policies():
    with pytest.raises(ValueError, match="must be positive"):
        ToolExecutionPolicy(max_concurrent_executions=0)


def test_execution_sandbox_rejects_when_concurrency_capacity_is_exhausted():
    policy = ToolExecutionPolicy(
        max_concurrent_executions=1,
        admission_timeout_seconds=0.01,
    )
    sandbox = ToolExecutionSandbox(policy)
    sandbox._slots.acquire()
    try:
        with pytest.raises(TimeoutError, match="capacity is exhausted"):
            sandbox.execute(FakeTool())
    finally:
        sandbox._slots.release()


def test_execution_sandbox_rejects_non_object_output():
    sandbox = ToolExecutionSandbox()
    tool = FakeTool(["not", "an", "object"])

    with pytest.raises(ValueError, match="JSON object"):
        sandbox.execute(tool)
