import pytest

from tools.literature_search import LiteratureSearchTool


def test_literature_search_rejects_oversized_query_before_network(monkeypatch):
    called = False

    def unexpected_request(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr("tools.literature_search.requests.get", unexpected_request)
    tool = LiteratureSearchTool()

    with pytest.raises(ValueError, match="exceeds"):
        tool.execute("q" * (tool.MAX_QUERY_LENGTH + 1))

    assert called is False


@pytest.mark.parametrize("max_results", [0, 26, True, "5"])
def test_literature_search_rejects_invalid_result_limits_before_network(
    monkeypatch, max_results
):
    monkeypatch.setattr(
        "tools.literature_search.requests.get",
        lambda *args, **kwargs: pytest.fail("network must not be called"),
    )

    with pytest.raises(ValueError, match="max_results"):
        LiteratureSearchTool().execute("agent safety", max_results=max_results)
