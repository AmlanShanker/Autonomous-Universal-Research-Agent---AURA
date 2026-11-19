from agents.evidence_quality import deduplicate_sources, normalize_source
from storage.database.memory import InMemoryDatabase


def test_source_provenance_and_credibility_score_are_explainable():
    source = normalize_source({
        "title": "A study", "authors": ["A. Author"], "abstract": "Evidence.",
        "url": "https://arxiv.org/abs/2401.01234", "publication_year": "2024",
    }, query="retrieval", provider="arxiv")
    assert source.canonical_id == "arxiv:2401.01234"
    assert source.source_provider == "arxiv"
    assert source.retrieved_query == "retrieval"
    assert source.credibility_score == 0.65
    assert source.credibility_reasons


def test_deduplication_prefers_richer_record_for_same_doi():
    sparse = normalize_source({"title": "Paper", "url": "https://doi.org/10.1234/ABC"})
    rich = normalize_source({
        "title": "Paper", "authors": ["Author"], "abstract": "Full abstract",
        "url": "https://publisher.example/paper", "doi": "10.1234/abc",
        "publication_year": 2025,
    })
    unique = deduplicate_sources([sparse, rich])
    assert len(unique) == 1
    assert unique[0].authors == ["Author"]
    assert unique[0].canonical_id == "doi:10.1234/abc"


def test_sources_persist_and_lookup_by_canonical_identity():
    db = InMemoryDatabase()
    source = normalize_source({"title": "Paper", "url": "https://arxiv.org/abs/2401.01234"})
    db.save_research_source(source)
    assert db.get_research_source(source.source_id) == source
    assert db.get_research_source_by_canonical_id(source.canonical_id) == source
    assert db.list_research_sources() == [source]
