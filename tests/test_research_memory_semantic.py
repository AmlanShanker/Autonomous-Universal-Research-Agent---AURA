from memory.research_memory import ResearchMemoryItem
from memory.research_memory_retriever import ResearchMemoryRetriever
from memory.research_memory_writer import ResearchMemoryWriter
from storage.database.memory import InMemoryDatabase


class FakeEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vectors[text]


def make_item(memory_id, question, content, embedding=None, **kwargs):
    return ResearchMemoryItem(
        memory_id=memory_id,
        run_id=f"run_{memory_id}",
        task_id=f"task_{memory_id}",
        research_question=question,
        content=content,
        embedding=embedding,
        **kwargs,
    )


def test_semantic_retrieval_finds_conceptually_similar_finding():
    query = "Does physical activity benefit heart function?"
    memory_text = "Regular exercise improves cardiovascular health."
    provider = FakeEmbeddingProvider({
        query: [1.0, 0.0],
        memory_text: [0.98, 0.02],
    })
    database = InMemoryDatabase()
    item = make_item(
        "exercise",
        "Effects of exercise",
        "Regular exercise improves cardiovascular health.",
        embedding=[0.98, 0.02],
    )
    database.save_research_memory(item)

    results = ResearchMemoryRetriever(
        database, embedding_provider=provider
    ).retrieve(query)

    assert [result.memory_id for result in results] == ["exercise"]


def test_semantic_retrieval_can_find_memory_without_lexical_overlap():
    query = "How do oceans absorb atmospheric carbon?"
    provider = FakeEmbeddingProvider({query: [1.0, 0.0]})
    database = InMemoryDatabase()
    database.save_research_memory(make_item(
        "marine-carbon",
        "Blue carbon ecosystems",
        "Seagrass meadows and mangroves capture and store carbon.",
        embedding=[0.99, 0.01],
    ))

    results = ResearchMemoryRetriever(
        database, embedding_provider=provider
    ).retrieve(query)

    assert [item.memory_id for item in results] == ["marine-carbon"]


def test_hybrid_retrieval_combines_semantic_and_keyword_signals():
    query = "retrieval recall"
    provider = FakeEmbeddingProvider({
        query: [1.0, 0.0],
        "Weak lexical candidate": [0.0, 1.0],
    })
    database = InMemoryDatabase()
    database.save_research_memory(make_item(
        "semantic",
        "Weak lexical candidate",
        "Unrelated wording",
        embedding=[1.0, 0.0],
    ))
    database.save_research_memory(make_item(
        "keyword",
        "retrieval recall",
        "retrieval recall",
        embedding=[0.0, 1.0],
        tags=["retrieval", "recall"],
    ))

    results = ResearchMemoryRetriever(
        database,
        embedding_provider=provider,
        semantic_weight=0.5,
        quality_weight=0.0,
    ).retrieve(query)

    assert [item.memory_id for item in results] == ["keyword", "semantic"]


def test_retrieval_scores_are_normalized_between_zero_and_one():
    database = InMemoryDatabase()
    database.save_research_memory(make_item(
        "match", "retrieval methods", "retrieval improves recall",
        tags=["retrieval"],
    ))

    scored = ResearchMemoryRetriever(database).retrieve_scored("retrieval")

    assert len(scored) == 1
    assert 0.0 <= scored[0][1] <= 1.0


def test_quality_ranking_prefers_confident_and_provenanced_finding():
    database = InMemoryDatabase()
    database.save_research_memory(make_item(
        "weak", "retrieval", "retrieval finding",
        confidence=0.2,
    ))
    database.save_research_memory(make_item(
        "strong", "retrieval", "retrieval finding",
        confidence=0.95,
        source_result_id="result_1",
        source_tool_id="search",
    ))

    results = ResearchMemoryRetriever(
        database, quality_weight=0.5
    ).retrieve("retrieval")

    assert [item.memory_id for item in results] == ["strong", "weak"]


def test_writer_persists_provider_generated_embedding():
    provider = FakeEmbeddingProvider({
        "A research question\nA reusable finding\nscience": [0.2, 0.8]
    })
    database = InMemoryDatabase()
    item = ResearchMemoryWriter(
        database, embedding_provider=provider
    ).write_finding(
        run_id="run_1",
        task_id="task_1",
        research_question="A research question",
        content="A reusable finding",
        tags=["science"],
    )

    stored = database.get_research_memory(item.memory_id)
    assert stored.embedding == [0.2, 0.8]


def test_semantic_retrieval_rejects_embedding_dimension_mismatch():
    query = "semantic query"
    provider = FakeEmbeddingProvider({query: [1.0, 0.0]})
    database = InMemoryDatabase()
    database.save_research_memory(make_item(
        "bad-dimension", "question", "finding", embedding=[1.0, 0.0, 0.0]
    ))

    try:
        ResearchMemoryRetriever(
            database, embedding_provider=provider
        ).retrieve(query)
        assert False, "Expected embedding dimension mismatch."
    except ValueError as error:
        assert "dimensions do not match" in str(error)


def test_retriever_validates_weight_configuration():
    database = InMemoryDatabase()
    for kwargs in (
        {"semantic_weight": -0.1},
        {"semantic_weight": 1.1},
        {"quality_weight": True},
    ):
        try:
            ResearchMemoryRetriever(database, **kwargs)
            assert False, "Expected invalid weight to be rejected."
        except ValueError:
            pass
