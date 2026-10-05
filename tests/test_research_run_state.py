import pytest
from pydantic import ValidationError

from storage.models.research import (
    ResearchRun,
    ResearchRunStatus,
)


def test_research_run_defaults_to_created():
    run = ResearchRun(
        run_id="run_001",
        research_question="Does chunk size affect RAG retrieval?",
    )

    assert run.status == ResearchRunStatus.CREATED
    assert run.tasks == []
    assert run.completed_tasks == []
    assert run.failed_tasks == []
    assert run.findings == []


def test_research_run_supports_all_lifecycle_states():
    for status in ResearchRunStatus:
        run = ResearchRun(
            run_id="run_001",
            research_question="Test research question.",
            status=status,
        )

        assert run.status == status


def test_research_run_rejects_invalid_status():
    with pytest.raises(ValidationError):
        ResearchRun(
            run_id="run_001",
            research_question="Test research question.",
            status="unknown_state",
        )


def test_research_run_can_track_tasks():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.RUNNING,
        tasks=[
            "task_001",
            "task_002",
            "task_003",
        ],
        completed_tasks=[
            "task_001",
        ],
        failed_tasks=[
            "task_003",
        ],
    )

    assert run.status == ResearchRunStatus.RUNNING
    assert run.tasks == [
        "task_001",
        "task_002",
        "task_003",
    ]
    assert run.completed_tasks == [
        "task_001",
    ]
    assert run.failed_tasks == [
        "task_003",
    ]


def test_research_run_findings_are_separate_from_execution_results():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        findings=[
            "The experiment improved retrieval recall."
        ],
    )

    assert run.findings == [
        "The experiment improved retrieval recall."
    ]


def test_research_run_can_transition_from_created_to_planned():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
    )

    run.transition_to(
        ResearchRunStatus.PLANNED
    )

    assert run.status == ResearchRunStatus.PLANNED


def test_research_run_can_transition_through_normal_lifecycle():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
    )

    run.transition_to(
        ResearchRunStatus.PLANNED
    )

    run.transition_to(
        ResearchRunStatus.RUNNING
    )

    run.transition_to(
        ResearchRunStatus.COMPLETED
    )

    assert run.status == ResearchRunStatus.COMPLETED


def test_research_run_can_become_blocked_and_resume():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.RUNNING,
    )

    run.transition_to(
        ResearchRunStatus.BLOCKED
    )

    assert run.status == ResearchRunStatus.BLOCKED

    run.transition_to(
        ResearchRunStatus.RUNNING
    )

    assert run.status == ResearchRunStatus.RUNNING


def test_research_run_can_fail_from_running():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.RUNNING,
    )

    run.transition_to(
        ResearchRunStatus.FAILED
    )

    assert run.status == ResearchRunStatus.FAILED


@pytest.mark.parametrize(
    "start_status,new_status",
    [
        (
            ResearchRunStatus.CREATED,
            ResearchRunStatus.RUNNING,
        ),
        (
            ResearchRunStatus.CREATED,
            ResearchRunStatus.COMPLETED,
        ),
        (
            ResearchRunStatus.PLANNED,
            ResearchRunStatus.COMPLETED,
        ),
        (
            ResearchRunStatus.RUNNING,
            ResearchRunStatus.CREATED,
        ),
        (
            ResearchRunStatus.COMPLETED,
            ResearchRunStatus.RUNNING,
        ),
        (
            ResearchRunStatus.FAILED,
            ResearchRunStatus.RUNNING,
        ),
    ],
)
def test_invalid_research_run_transitions_are_rejected(
    start_status,
    new_status,
):
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=start_status,
    )

    with pytest.raises(ValueError):
        run.transition_to(new_status)

    assert run.status == start_status


def test_transition_to_same_state_is_allowed():
    run = ResearchRun(
        run_id="run_001",
        research_question="Test research question.",
        status=ResearchRunStatus.RUNNING,
    )

    run.transition_to(
        ResearchRunStatus.RUNNING
    )

    assert run.status == ResearchRunStatus.RUNNING
