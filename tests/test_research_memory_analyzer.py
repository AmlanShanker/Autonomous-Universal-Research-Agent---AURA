from memory.research_memory import ResearchMemoryItem
from memory.research_memory_analyzer import ResearchMemoryAnalyzer


class FakeDatabase:
    def __init__(self, items):
        self.items = items
        self.queries = []

    def search_research_memory(self, query):
        self.queries.append(query)
        return list(self.items)


def create_item(memory_id, content):
    return ResearchMemoryItem(
        memory_id=memory_id,
        run_id=f"run_{memory_id}",
        task_id=f"task_{memory_id}",
        research_question="Evaluate dense retrieval.",
        content=content,
    )


def test_analyzer_detects_normalized_duplicate_content():
    existing = create_item(
        "stored",
        "Dense retrieval improves recall.",
    )
    candidate = create_item(
        "candidate",
        "DENSE retrieval improves recall",
    )
    analyzer = ResearchMemoryAnalyzer(FakeDatabase([existing]))

    result = analyzer.analyze(candidate)

    assert [match.memory_id for match in result.duplicates] == ["stored"]
    assert result.duplicates[0].similarity == 1.0
    assert result.potential_conflicts == []


def test_analyzer_detects_negation_as_potential_conflict():
    existing = create_item(
        "stored",
        "Dense retrieval improves recall.",
    )
    candidate = create_item(
        "candidate",
        "Dense retrieval does not improve recall.",
    )
    analyzer = ResearchMemoryAnalyzer(FakeDatabase([existing]))

    result = analyzer.analyze(candidate)

    assert result.duplicates == []
    assert [match.memory_id for match in result.potential_conflicts] == ["stored"]
    assert result.potential_conflicts[0].similarity >= 0.5


def test_analyzer_detects_opposing_directional_claims():
    existing = create_item(
        "stored",
        "The treatment increases model accuracy.",
    )
    candidate = create_item(
        "candidate",
        "The treatment decreases model accuracy.",
    )
    analyzer = ResearchMemoryAnalyzer(
        FakeDatabase([existing]),
        conflict_threshold=0.4,
    )

    result = analyzer.analyze(candidate)

    assert [match.memory_id for match in result.potential_conflicts] == ["stored"]


def test_analyzer_excludes_candidate_and_duplicate_database_rows():
    candidate = create_item("candidate", "Retrieval improves recall.")
    existing = create_item("stored", "Retrieval improves recall.")
    analyzer = ResearchMemoryAnalyzer(
        FakeDatabase([candidate, existing, existing])
    )

    result = analyzer.analyze(candidate)

    assert [match.memory_id for match in result.duplicates] == ["stored"]


def test_analyzer_returns_empty_result_for_empty_content():
    candidate = create_item("candidate", "   ")
    database = FakeDatabase([])
    analyzer = ResearchMemoryAnalyzer(database)

    result = analyzer.analyze(candidate)

    assert result.duplicates == []
    assert result.potential_conflicts == []
    assert database.queries == []


def test_analyzer_validates_similarity_thresholds():
    for kwargs in (
        {"duplicate_threshold": 0},
        {"duplicate_threshold": 1.1},
        {"conflict_threshold": True},
        {"conflict_threshold": "0.5"},
    ):
        try:
            ResearchMemoryAnalyzer(FakeDatabase([]), **kwargs)
            assert False, "Expected invalid threshold to raise ValueError."
        except ValueError:
            pass


def test_analyzer_orders_matches_deterministically():
    candidate = create_item("candidate", "Dense retrieval improves recall.")
    items = [
        create_item("stored_b", "Dense retrieval improves recall."),
        create_item("stored_a", "Dense retrieval improves recall."),
    ]
    analyzer = ResearchMemoryAnalyzer(FakeDatabase(items))

    result = analyzer.analyze(candidate)

    assert [match.memory_id for match in result.duplicates] == [
        "stored_a",
        "stored_b",
    ]
