from fastapi.testclient import TestClient

from api.app import create_app
from storage.database.memory import InMemoryDatabase
from storage.models.research import ResearchPlan, ResearchRun, ResearchRunStatus, ResearchTask


class FakeDirector:
    def __init__(self, database):
        self.database = database

    def create_plan(self, question):
        plan = ResearchPlan(
            question=question,
            objectives=["Investigate the question"],
            success_criteria=["Produce evidence"],
        )
        self.database.save_research_plan(plan)
        return plan

    def create_tasks(self, plan):
        task = ResearchTask(
            task_id="task_api_1",
            description="Review available evidence",
            task_type="analysis",
        )
        self.database.save_research_task(task)
        return [task]

    def start_run(self, run_id, question):
        run = ResearchRun(run_id=run_id, research_question=question)
        run.transition_to(ResearchRunStatus.PLANNED)
        self.database.save_research_run(run)
        return run


def client_with_fakes():
    database = InMemoryDatabase()
    app = create_app(database=database, director=FakeDirector(database), experiment_adapters={})
    return TestClient(app), database


def test_health_endpoint_returns_service_status():
    client, _ = client_with_fakes()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "aura-api"}


def test_create_research_run_persists_plan_tasks_and_run():
    client, database = client_with_fakes()

    response = client.post("/research/runs", json={"question": "Can retrieval improve evidence quality?"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["run"]["status"] == "planned"
    assert payload["run"]["tasks"] == ["task_api_1"]
    assert payload["plan"]["question"] == "Can retrieval improve evidence quality?"
    assert database.get_research_run(payload["run"]["run_id"]) is not None
    assert database.get_research_task("task_api_1") is not None


def test_create_research_run_rejects_blank_question():
    client, _ = client_with_fakes()

    response = client.post("/research/runs", json={"question": "   "})

    assert response.status_code == 422


def test_get_run_and_tasks_return_404_for_unknown_run():
    client, _ = client_with_fakes()

    assert client.get("/research/runs/missing").status_code == 404
    assert client.get("/research/runs/missing/tasks").status_code == 404


def test_report_endpoint_builds_persists_and_retrieves_report():
    client, database = client_with_fakes()
    created = client.post("/research/runs", json={"question": "What is evidence quality?"}).json()
    run_id = created["run"]["run_id"]

    report_response = client.post(f"/research/runs/{run_id}/report")

    assert report_response.status_code == 200
    report = report_response.json()
    assert report["run_id"] == run_id
    assert report["research_question"] == "What is evidence quality?"
    assert database.get_research_report(report["report_id"]) is not None
    assert client.get(f"/research/reports/{report['report_id']}").status_code == 200


def test_run_experiments_endpoint_returns_empty_list_for_new_run():
    client, _ = client_with_fakes()
    created = client.post("/research/runs", json={"question": "How does retrieval work?"}).json()

    response = client.get(f"/research/runs/{created['run']['run_id']}/experiments")

    assert response.status_code == 200
    assert response.json() == []



def test_execute_endpoint_runs_a_task_graph_without_tools():
    client, database = client_with_fakes()
    created = client.post("/research/runs", json={"question": "Can AURA execute a task?"}).json()
    run_id = created["run"]["run_id"]

    response = client.post(f"/research/runs/{run_id}/execute")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert database.get_research_run(run_id).completed_tasks == ["task_api_1"]


def test_execute_endpoint_rejects_unknown_run():
    client, _ = client_with_fakes()

    response = client.post("/research/runs/missing/execute")

    assert response.status_code == 404


def test_report_endpoint_rejects_unknown_run():
    client, _ = client_with_fakes()

    response = client.post("/research/runs/missing/report")

    assert response.status_code == 404
