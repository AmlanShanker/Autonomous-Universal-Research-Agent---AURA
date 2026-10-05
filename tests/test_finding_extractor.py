from storage.models.execution import ExecutionResult
from memory.finding_extractor import ResearchFindingExtractor


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def create_execution_result() -> ExecutionResult:
    return ExecutionResult(
        result_id="result_001",
        run_id="run_001",
        task_id="task_a",
        tool_id="literature_search",
        implementation_id="literature_search_builtin",
        output={
            "query": "retrieval performance",
            "results": [
                {
                    "title": "Dense Retrieval Study",
                    "publication_year": 2024,
                }
            ],
            "count": 1,
        },
    )


def test_extractor_returns_structured_findings():
    llm = FakeLLM(
        """
        {
            "findings": [
                {
                    "content": "Dense retrieval research includes a 2024 study.",
                    "memory_type": "finding",
                    "tags": ["retrieval", "dense"],
                    "metadata": {
                        "source": "literature_search"
                    }
                }
            ]
        }
        """
    )

    extractor = ResearchFindingExtractor(llm)

    findings = extractor.extract(
        research_question="Investigate retrieval performance.",
        execution_result=create_execution_result(),
    )

    assert len(findings) == 1
    assert findings[0]["content"] == (
        "Dense retrieval research includes a 2024 study."
    )
    assert findings[0]["memory_type"] == "finding"
    assert findings[0]["tags"] == [
        "retrieval",
        "dense",
    ]
    assert findings[0]["metadata"]["source"] == (
        "literature_search"
    )


def test_extractor_accepts_empty_findings():
    llm = FakeLLM(
        """
        {
            "findings": []
        }
        """
    )

    extractor = ResearchFindingExtractor(llm)

    findings = extractor.extract(
        research_question="Investigate retrieval performance.",
        execution_result=create_execution_result(),
    )

    assert findings == []


def test_extractor_rejects_invalid_json():
    llm = FakeLLM("not valid json")

    extractor = ResearchFindingExtractor(llm)

    try:
        extractor.extract(
            research_question="Investigate retrieval performance.",
            execution_result=create_execution_result(),
        )
        assert False, (
            "Expected invalid JSON to raise ValueError."
        )
    except ValueError as error:
        assert "invalid JSON" in str(error)


def test_extractor_rejects_missing_findings():
    llm = FakeLLM(
        """
        {
            "result": []
        }
        """
    )

    extractor = ResearchFindingExtractor(llm)

    try:
        extractor.extract(
            research_question="Investigate retrieval performance.",
            execution_result=create_execution_result(),
        )
        assert False, (
            "Expected missing findings to raise ValueError."
        )
    except ValueError as error:
        assert "contain" in str(error)
        assert "findings" in str(error)
        assert "list" in str(error)


def test_extractor_rejects_empty_finding_content():
    llm = FakeLLM(
        """
        {
            "findings": [
                {
                    "content": "",
                    "memory_type": "finding",
                    "tags": [],
                    "metadata": {}
                }
            ]
        }
        """
    )

    extractor = ResearchFindingExtractor(llm)

    try:
        extractor.extract(
            research_question="Investigate retrieval performance.",
            execution_result=create_execution_result(),
        )
        assert False, (
            "Expected empty finding content to raise ValueError."
        )
    except ValueError as error:
        assert "non-empty string content" in str(error)