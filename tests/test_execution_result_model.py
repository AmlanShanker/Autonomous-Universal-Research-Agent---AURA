from storage.models.execution import ExecutionResult


def test_execution_result_defaults():
    result = ExecutionResult(
        result_id="result_001",
        run_id="run_001",
        task_id="task_001",
        tool_id="literature_search",
        implementation_id="literature_search_builtin_v1",
    )

    assert result.result_id == "result_001"
    assert result.run_id == "run_001"
    assert result.task_id == "task_001"
    assert result.tool_id == "literature_search"
    assert result.implementation_id == "literature_search_builtin_v1"
    assert result.implementation_version == "1.0.0"
    assert result.status == "success"
    assert result.output == {}
    assert result.metadata == {}


def test_execution_result_can_store_tool_output():
    result = ExecutionResult(
        result_id="result_002",
        run_id="run_001",
        task_id="task_001",
        tool_id="literature_search",
        implementation_id="literature_search_builtin_v1",
        implementation_version="1.0.0",
        status="success",
        output={
            "query": "transformer efficiency",
            "count": 2,
            "results": [
                {
                    "title": "Example Paper",
                    "url": "https://example.com/paper",
                }
            ],
        },
        metadata={
            "source": "arXiv",
        },
    )

    assert result.status == "success"
    assert result.output["query"] == "transformer efficiency"
    assert result.output["count"] == 2
    assert result.output["results"][0]["title"] == "Example Paper"
    assert result.metadata["source"] == "arXiv"


def test_execution_result_can_record_failure():
    result = ExecutionResult(
        result_id="result_003",
        run_id="run_001",
        task_id="task_002",
        tool_id="dataset_download",
        implementation_id="dataset_download_builtin_v1",
        status="failed",
        output={
            "error": "Download failed.",
        },
        metadata={
            "attempt": 1,
        },
    )

    assert result.status == "failed"
    assert result.output["error"] == "Download failed."
    assert result.metadata["attempt"] == 1
