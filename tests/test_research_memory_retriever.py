from memory.research_memory import ResearchMemoryItem
from memory.research_memory_retriever import ResearchMemoryRetriever


class FakeDatabase:
    def __init__(self, items):
        self.items = items
        self.queries = []

    def search_research_memory(self, query):
        self.queries.append(query)
        return list(self.items)


def create_item(
    memory_id,
    *,
    run_id="run_old",
    question="",
    content="",
    tags=None,
):
    return ResearchMemoryItem(
        memory_id=memory_id,
        run_id=run_id,
        task_id=f"task_{memory_id}",
        research_question=question,
        content=content,
        tags=tags or [],
    )


def test_retriever_ranks_tags_above_content_and_question():
    database = FakeDatabase(
        [
            create_item(
                "question-match",
                question="RAG retrieval methods",
            ),
            create_item(
                "content-match",
                content="RAG retrieval methods",
            ),
            create_item(
                "tag-match",
                tags=["RAG", "retrieval"],
            ),
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    results = retriever.retrieve("RAG retrieval")

    assert [
        item.memory_id
        for item in results
    ] == [
        "tag-match",
        "content-match",
        "question-match",
    ]


def test_retriever_applies_result_limit():
    database = FakeDatabase(
        [
            create_item("memory_03", tags=["retrieval"]),
            create_item("memory_01", tags=["retrieval"]),
            create_item("memory_02", tags=["retrieval"]),
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    results = retriever.retrieve(
        "retrieval",
        limit=2,
    )

    assert [
        item.memory_id
        for item in results
    ] == [
        "memory_01",
        "memory_02",
    ]


def test_retriever_can_exclude_current_research_run():
    database = FakeDatabase(
        [
            create_item(
                "current",
                run_id="run_current",
                tags=["retrieval"],
            ),
            create_item(
                "previous",
                run_id="run_previous",
                tags=["retrieval"],
            ),
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    results = retriever.retrieve(
        "retrieval",
        exclude_run_id="run_current",
    )

    assert [
        item.memory_id
        for item in results
    ] == ["previous"]


def test_retriever_returns_empty_for_blank_or_stopword_query():
    database = FakeDatabase(
        [
            create_item(
                "memory_01",
                tags=["retrieval"],
            )
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("how does the") == []
    assert database.queries == []


def test_retriever_returns_empty_when_no_candidates_match():
    database = FakeDatabase(
        [
            create_item(
                "memory_01",
                content="Database indexing improves query latency.",
                tags=["database"],
            )
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    assert retriever.retrieve("quantum computing") == []


def test_retriever_uses_deterministic_memory_id_tiebreaker():
    database = FakeDatabase(
        [
            create_item("memory_b", tags=["retrieval"]),
            create_item("memory_a", tags=["retrieval"]),
        ]
    )

    retriever = ResearchMemoryRetriever(database)

    results = retriever.retrieve("retrieval")

    assert [
        item.memory_id
        for item in results
    ] == [
        "memory_a",
        "memory_b",
    ]


def test_retriever_rejects_non_positive_or_non_integer_limits():
    database = FakeDatabase([])
    retriever = ResearchMemoryRetriever(database)

    for invalid_limit in (0, -1, 1.5, True):
        try:
            retriever.retrieve(
                "retrieval",
                limit=invalid_limit,
            )
            assert False, "Expected invalid limit to raise ValueError."
        except ValueError as error:
            assert "positive integer" in str(error)


def test_retriever_rejects_non_string_queries():
    retriever = ResearchMemoryRetriever(FakeDatabase([]))

    try:
        retriever.retrieve(None)
        assert False, "Expected a non-string query to raise TypeError."
    except TypeError as error:
        assert "must be a string" in str(error)
