from fastapi.testclient import TestClient
import pytest

from agents.autonomous_improvement import AutonomousImprovementEngine
from api.app import create_app
from storage.database.memory import InMemoryDatabase
from storage.models.experiment import ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.research import ResearchRun, ResearchRunStatus, ResearchTask


class FakeRegistry:
    def __init__(self, available=()):
        self.available = set(available)

    def has(self, name):
        return name in self.available


def seeded_database():
    database = InMemoryDatabase()
    run = ResearchRun(
        run_id="run_improvement",
        research_question="Can the system improve its evidence coverage?",
        status=ResearchRunStatus.COMPLETED,
        tasks=["task_missing", "task_failed"],
        completed_tasks=["task_missing", "task_failed"],
    )
    database.save_research_run(run)
    database.save_research_task(ResearchTask(
        task_id="task_missing",
        description="Use a missing capability",
        task_type="research",
        required_tools=["specialized_search"],
    ))
    database.save_research_task(ResearchTask(
        task_id="task_failed",
        description="Execute a tool",
        task_type="research",
        required_tools=["literature_search", "specialized_search"],
    ))
    database.save_execution_result(ExecutionResult(
        result_id="execution_failed",
        run_id=run.run_id,
        task_id="task_failed",
        tool_id="literature_search",
        implementation_id="literature_search_v1",
        status="failed",
        output={},
        metadata={"error_type": "TimeoutError"},
    ))
    return database


def test_analysis_creates_deterministic_evidence_backed_proposals():
    database = seeded_database()
    engine = AutonomousImprovementEngine(database, FakeRegistry({"literature_search"}))

    cycle = engine.analyze_run("run_improvement")
    repeated = engine.analyze_run("run_improvement")
    proposals = database.list_improvement_proposals_for_cycle(cycle.cycle_id)

    assert repeated.cycle_id == cycle.cycle_id
    assert cycle.evidence_sha256
    assert cycle.status == "open"
    assert {item.category for item in proposals} >= {"capability_gap", "execution_reliability", "research_quality"}
    assert all(item.evidence_refs for item in proposals)
    assert all(item.status == "proposed" for item in proposals)
    repeated_gap_proposals = [item for item in proposals if item.category == "capability_gap"]
    assert len(repeated_gap_proposals) == 2
    assert len({item.proposal_id for item in repeated_gap_proposals}) == 2
    assert len({item.evidence_refs[0] for item in repeated_gap_proposals}) == 2


def test_cycle_identity_changes_when_registered_capabilities_change():
    database = seeded_database()
    first = AutonomousImprovementEngine(
        database, FakeRegistry({"literature_search"})
    ).analyze_run("run_improvement")
    second = AutonomousImprovementEngine(
        database, FakeRegistry({"literature_search", "specialized_search"})
    ).analyze_run("run_improvement")

    assert first.cycle_id != second.cycle_id
    assert first.evidence_sha256 != second.evidence_sha256


def test_approval_is_required_before_validation_and_reviewer_is_recorded():
    database = seeded_database()
    engine = AutonomousImprovementEngine(database, FakeRegistry({"literature_search"}))
    cycle = engine.analyze_run("run_improvement")
    proposal = database.list_improvement_proposals_for_cycle(cycle.cycle_id)[0]

    with pytest.raises(ValueError, match="Only approved"):
        engine.validate(proposal.proposal_id, "missing-result")

    approved = engine.approve(proposal.proposal_id, "research-lead")
    assert approved.status == "approved"
    assert approved.approved_by == "research-lead"
    with pytest.raises(ValueError, match="Only proposed"):
        engine.reject(proposal.proposal_id, "reviewer", "Too broad")


def test_rejection_requires_reason_and_closes_cycle_when_all_proposals_terminal():
    database = seeded_database()
    engine = AutonomousImprovementEngine(database, FakeRegistry({"literature_search"}))
    cycle = engine.analyze_run("run_improvement")
    proposals = database.list_improvement_proposals_for_cycle(cycle.cycle_id)

    with pytest.raises(ValueError, match="rejection reason"):
        engine.reject(proposals[0].proposal_id, "reviewer", "  ")

    for proposal in proposals:
        engine.reject(proposal.proposal_id, "reviewer", "Not in current scope")

    refreshed = database.get_improvement_cycle(cycle.cycle_id)
    assert refreshed.status == "completed"
    assert refreshed.completed_at is not None


def test_validation_requires_approved_proposal_and_same_run_experiment():
    database = seeded_database()
    engine = AutonomousImprovementEngine(database, FakeRegistry({"literature_search"}))
    cycle = engine.analyze_run("run_improvement")
    proposal = database.list_improvement_proposals_for_cycle(cycle.cycle_id)[0]
    engine.approve(proposal.proposal_id, "reviewer")

    database.save_experiment_result(ExperimentResult(
        result_id="experiment_other_run",
        experiment_id="experiment_1",
        run_id="another_run",
        task_id="task_1",
        status="completed",
        passed=True,
    ))
    with pytest.raises(ValueError, match="belong to"):
        engine.validate(proposal.proposal_id, "experiment_other_run")

    database.save_experiment_result(ExperimentResult(
        result_id="experiment_same_run",
        experiment_id="experiment_2",
        run_id="run_improvement",
        task_id="task_failed",
        status="completed",
        passed=False,
    ))
    validated = engine.validate(proposal.proposal_id, "experiment_same_run")
    assert validated.status == "validated"
    assert validated.validation_passed is False
    assert validated.validation_experiment_result_id == "experiment_same_run"


def test_analysis_rejects_missing_run_or_incomplete_task_graph():
    database = InMemoryDatabase()
    engine = AutonomousImprovementEngine(database, FakeRegistry())
    with pytest.raises(KeyError, match="not found"):
        engine.analyze_run("missing")

    database.save_research_run(ResearchRun(
        run_id="run_incomplete",
        research_question="Incomplete",
        tasks=["missing_task"],
    ))
    with pytest.raises(ValueError, match="complete persisted task graph"):
        engine.analyze_run("run_incomplete")


def test_improvement_cycle_api_supports_analysis_review_and_validation():
    database = seeded_database()
    app = create_app(
        database=database,
        experiment_adapters={},
        tool_registry_factory=lambda _database: FakeRegistry({"literature_search"}),
    )
    client = TestClient(app)

    response = client.post("/research/runs/run_improvement/improvement-cycles")
    assert response.status_code == 201
    cycle = response.json()
    proposals = client.get(
        f"/research/improvement-cycles/{cycle['cycle_id']}/proposals"
    ).json()
    assert proposals

    proposal_id = proposals[0]["proposal_id"]
    approved = client.post(
        f"/research/improvement-proposals/{proposal_id}/approve",
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    database.save_experiment_result(ExperimentResult(
        result_id="api_experiment_result",
        experiment_id="api_experiment",
        run_id="run_improvement",
        task_id="task_failed",
        status="completed",
        passed=True,
    ))
    validated = client.post(
        f"/research/improvement-proposals/{proposal_id}/validate",
        json={"experiment_result_id": "api_experiment_result"},
    )
    assert validated.status_code == 200
    assert validated.json()["validation_passed"] is True


def test_improvement_api_returns_not_found_for_unknown_resources():
    database = seeded_database()
    app = create_app(
        database=database,
        experiment_adapters={},
        tool_registry_factory=lambda _database: FakeRegistry(),
    )
    client = TestClient(app)
    assert client.get("/research/improvement-cycles/missing").status_code == 404
    assert client.post(
        "/research/improvement-proposals/missing/approve",
        
    ).status_code == 404



def test_improvement_review_identity_comes_from_authenticated_api_key(monkeypatch):
    import json

    reviewer_key = "reviewer-key-0123456789-abcdefghijkl"
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "AURA_API_KEYS",
        json.dumps([{"id": "trusted-operator-42", "key": reviewer_key, "role": "operator"}]),
    )
    database = seeded_database()
    app = create_app(
        database=database,
        experiment_adapters={},
        tool_registry_factory=lambda _database: FakeRegistry({"literature_search"}),
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {reviewer_key}"}

    cycle_response = client.post(
        "/research/runs/run_improvement/improvement-cycles", headers=headers
    )
    assert cycle_response.status_code == 201
    proposal = database.list_improvement_proposals_for_cycle(
        cycle_response.json()["cycle_id"]
    )[0]

    spoofed = client.post(
        f"/research/improvement-proposals/{proposal.proposal_id}/approve",
        headers=headers,
        json={"reviewer": "forged-admin"},
    )
    assert spoofed.status_code == 422

    approved = client.post(
        f"/research/improvement-proposals/{proposal.proposal_id}/approve",
        headers=headers,
    )
    assert approved.status_code == 200
    assert approved.json()["approved_by"] == "trusted-operator-42"
