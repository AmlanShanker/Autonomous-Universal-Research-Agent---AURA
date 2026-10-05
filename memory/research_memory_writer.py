import uuid

from memory.research_memory import ResearchMemoryItem
from storage.database.base import Database


class ResearchMemoryWriter:
    """
    Persists reusable research findings into the research-memory layer.

    This component deliberately stores findings rather than raw
    ExecutionResult objects. Raw execution data remains in the
    execution-results persistence layer.
    """

    def __init__(self, database: Database):
        self.database = database

    def write_finding(
        self,
        run_id: str,
        task_id: str,
        research_question: str,
        content: str,
        memory_type: str = "finding",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ResearchMemoryItem:
        """
        Create and persist one reusable research-memory item.
        """

        if not content.strip():
            raise ValueError(
                "Research memory content cannot be empty."
            )

        item = ResearchMemoryItem(
            memory_id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            research_question=research_question,
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.database.save_research_memory(item)

        return item