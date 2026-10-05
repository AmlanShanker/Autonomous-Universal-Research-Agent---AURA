from agents.research_director import ResearchDirector
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchRunStatus


class FakeLLM:
    def generate(self, prompt: str) -> str:
        raise AssertionError(
            "LLM should not be called by start_run."
        )


def test_start_run_uses_planned_state_transition():
    database = InMemoryDatabase()

    director = ResearchDirector(
        database=database,
        llm_client=FakeLLM(),
    )

    run = director.start_run(
        run_id="run_001",
        question="Test research question.",
    )

    assert run.status == ResearchRunStatus.PLANNED

    stored = database.get_research_run(
        "run_001"
    )

    assert stored is not None
    assert stored.status == ResearchRunStatus.PLANNED
