from pydantic import BaseModel, Field


class ResearchMemoryItem(BaseModel):
    """
    A persistent piece of knowledge produced by a research run.

    Research memory stores reusable findings rather than raw
    execution details.
    """

    memory_id: str
    run_id: str
    task_id: str

    research_question: str

    content: str

    memory_type: str = "finding"

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict = Field(
        default_factory=dict
    )


class ResearchMemory:
    """
    Persistent-memory abstraction for research knowledge.

    The initial implementation keeps memory in process.
    Persistence will be added through the database layer
    after the model and behavior are verified.
    """

    def __init__(self):
        self.items: dict[
            str,
            ResearchMemoryItem,
        ] = {}

    def add(
        self,
        item: ResearchMemoryItem,
    ) -> None:
        """
        Store a research-memory item.
        """

        if item.memory_id in self.items:
            raise ValueError(
                "Research memory item "
                f"'{item.memory_id}' already exists."
            )

        self.items[
            item.memory_id
        ] = item

    def get(
        self,
        memory_id: str,
    ) -> ResearchMemoryItem | None:
        """
        Retrieve one memory item.
        """

        return self.items.get(
            memory_id
        )

    def list_for_run(
        self,
        run_id: str,
    ) -> list[ResearchMemoryItem]:
        """
        Return memory created by a research run.
        """

        return [
            item
            for item in self.items.values()
            if item.run_id == run_id
        ]

    def search(
        self,
        query: str,
    ) -> list[ResearchMemoryItem]:
        """
        Perform a simple keyword search.

        This is intentionally deterministic.

        Semantic/vector retrieval will be introduced later.
        """

        query_terms = {
            term.lower()
            for term in query.split()
            if term.strip()
        }

        if not query_terms:
            return []

        results = []

        for item in self.items.values():
            searchable_text = (
                f"{item.research_question} "
                f"{item.content} "
                f"{' '.join(item.tags)}"
            ).lower()

            if any(
                term in searchable_text
                for term in query_terms
            ):
                results.append(item)

        return results