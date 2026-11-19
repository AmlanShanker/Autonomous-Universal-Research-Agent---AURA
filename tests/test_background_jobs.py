from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from agents.task_executor import ExecutionCancelled, TaskExecutor

from api.app import create_app
from storage.database.memory import InMemoryDatabase
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.job_attempt import ResearchJobAttempt, ResearchJobAttemptStatus
from storage.models.research import ResearchPlan, ResearchRun, ResearchRunStatus, ResearchTask
from workers.research_worker import ResearchWorker
from tools.registry import ToolRegistry


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
            task_id="task_job_1",
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


def build_client():
    database = InMemoryDatabase()
    app = create_app(database=database, director=FakeDirector(database), experiment_adapters={})
    return TestClient(app), database


def create_run(client):
    response = client.post("/research/runs", json={"question": "Can background execution improve AURA?"})
    assert response.status_code == 201
    return response.json()["run"]["run_id"]


def test_enqueue_job_persists_queued_job():
    client, database = build_client()
    run_id = create_run(client)

    response = client.post(f"/research/runs/{run_id}/jobs")

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["run_id"] == run_id
    assert database.get_research_job(payload["job_id"]) is not None


def test_idempotency_key_returns_same_job():
    client, database = build_client()
    run_id = create_run(client)
    headers = {"Idempotency-Key": "research-request-001"}

    first = client.post(f"/research/runs/{run_id}/jobs", headers=headers)
    second = client.post(f"/research/runs/{run_id}/jobs", headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert len(database.list_research_jobs_for_run(run_id)) == 1


def test_only_one_active_job_can_be_queued_for_a_run():
    client, _ = build_client()
    run_id = create_run(client)

    assert client.post(f"/research/runs/{run_id}/jobs").status_code == 202
    response = client.post(f"/research/runs/{run_id}/jobs")

    assert response.status_code == 409


def test_queued_job_can_be_cancelled_and_retrieved():
    client, _ = build_client()
    run_id = create_run(client)
    job = client.post(f"/research/runs/{run_id}/jobs").json()

    cancelled = client.post(f"/research/jobs/{job['job_id']}/cancel")
    retrieved = client.get(f"/research/jobs/{job['job_id']}")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert retrieved.json()["status"] == "cancelled"


def test_worker_executes_queued_job_and_persists_progress():
    client, database = build_client()
    run_id = create_run(client)
    job = client.post(f"/research/runs/{run_id}/jobs").json()

    worker = ResearchWorker(database, worker_id="test-worker", experiment_adapters={})
    completed = worker.process_next()

    assert completed is not None
    assert completed.job_id == job["job_id"]
    assert completed.status == ResearchJobStatus.SUCCEEDED
    assert completed.progress == 100.0
    assert completed.completed_tasks == 1
    assert database.get_research_run(run_id).status == ResearchRunStatus.COMPLETED
    attempts_response = client.get(f"/research/jobs/{job['job_id']}/attempts")
    assert attempts_response.status_code == 200
    assert len(attempts_response.json()) == 1
    assert attempts_response.json()[0]["status"] == "succeeded"
    assert attempts_response.json()[0]["worker_id"] == "test-worker"


def test_worker_claims_each_job_only_once_while_lease_is_active():
    _, database = build_client()
    run_id = "run-claim"
    database.save_research_run(ResearchRun(run_id=run_id, research_question="Claim safety"))
    job = ResearchJob(job_id="job-claim", run_id=run_id)
    database.save_research_job(job)

    first = database.claim_next_research_job("worker-a", lease_seconds=300)
    second = database.claim_next_research_job("worker-b", lease_seconds=300)

    assert first is not None
    assert first.job_id == job.job_id
    assert second is None
    assert first.attempts == 1


def test_job_retry_endpoint_requeues_failed_job_within_bounded_budget():
    client, database = build_client()
    run_id = create_run(client)
    job = ResearchJob(job_id="job-retry", run_id=run_id, status=ResearchJobStatus.FAILED, attempts=1, max_attempts=2)
    database.save_research_job(job)

    response = client.post(f"/research/jobs/{job.job_id}/retry")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert database.get_research_job(job.job_id).attempts == 1


def test_enqueue_rejects_blank_idempotency_key():
    client, _ = build_client()
    run_id = create_run(client)

    response = client.post(
        f"/research/runs/{run_id}/jobs",
        headers={"Idempotency-Key": "   "},
    )

    assert response.status_code == 422


def test_cancel_unknown_job_returns_not_found():
    client, _ = build_client()

    response = client.post("/research/jobs/missing/cancel")

    assert response.status_code == 404



def test_worker_records_failed_attempt_without_exposing_unbounded_error():
    database = InMemoryDatabase()
    job = ResearchJob(job_id="job-missing-run", run_id="missing-run")
    database.save_research_job(job)

    worker = ResearchWorker(database, worker_id="failure-worker", experiment_adapters={})
    result = worker.process_next()

    assert result.status == ResearchJobStatus.QUEUED
    attempts = database.list_research_job_attempts(job.job_id)
    assert len(attempts) == 1
    assert attempts[0].status == ResearchJobAttemptStatus.FAILED
    assert attempts[0].error == "Research run was not found."
    assert attempts[0].finished_at is not None


def test_worker_marks_previous_running_attempt_as_lease_expired():
    database = InMemoryDatabase()
    job = ResearchJob(
        job_id="job-expired-attempt",
        run_id="missing-run",
        status=ResearchJobStatus.RUNNING,
        attempts=1,
        worker_id="old-worker",
        lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    database.save_research_job(job)
    previous = ResearchJobAttempt(
        attempt_id="job-expired-attempt:1",
        job_id=job.job_id,
        attempt_number=1,
        worker_id="old-worker",
    )
    database.save_research_job_attempt(previous)

    worker = ResearchWorker(database, worker_id="replacement-worker", experiment_adapters={})
    worker.process_next()

    attempts = database.list_research_job_attempts(job.job_id)
    assert [item.attempt_number for item in attempts] == [1, 2]
    assert attempts[0].status == ResearchJobAttemptStatus.LEASE_EXPIRED
    assert attempts[0].finished_at is not None
    assert attempts[1].status == ResearchJobAttemptStatus.FAILED


def test_job_attempt_endpoint_returns_not_found_for_unknown_job():
    client, _ = build_client()

    response = client.get("/research/jobs/missing/attempts")

    assert response.status_code == 404


def test_job_attempt_model_rejects_invalid_finish_transition():
    attempt = ResearchJobAttempt(
        attempt_id="attempt-invalid",
        job_id="job-invalid",
        attempt_number=1,
        worker_id="worker",
    )
    attempt.finish(ResearchJobAttemptStatus.SUCCEEDED)
    import pytest

    with pytest.raises(ValueError, match="Only running attempts"):
        attempt.finish(ResearchJobAttemptStatus.FAILED)


def test_database_rejects_multiple_active_jobs_for_one_run():
    database = InMemoryDatabase()
    first = ResearchJob(job_id="job-active-1", run_id="run-active")
    second = ResearchJob(job_id="job-active-2", run_id="run-active")
    database.save_research_job(first)

    import pytest

    with pytest.raises(ValueError, match="already has an active job"):
        database.save_research_job(second)

    assert [job.job_id for job in database.list_research_jobs_for_run("run-active")] == [
        "job-active-1"
    ]


def test_database_releases_active_run_slot_after_job_finishes():
    database = InMemoryDatabase()
    first = ResearchJob(job_id="job-finished", run_id="run-reusable")
    database.save_research_job(first)
    first.status = ResearchJobStatus.SUCCEEDED
    database.save_research_job(first)

    second = ResearchJob(job_id="job-next", run_id="run-reusable")
    database.save_research_job(second)

    assert database.get_research_job(second.job_id) is not None


def test_database_enforces_idempotency_key_uniqueness_across_runs():
    database = InMemoryDatabase()
    database.save_research_job(
        ResearchJob(job_id="job-key-1", run_id="run-key-1", idempotency_key="shared-key")
    )

    import pytest

    with pytest.raises(ValueError, match="Idempotency key"):
        database.save_research_job(
            ResearchJob(job_id="job-key-2", run_id="run-key-2", idempotency_key="shared-key")
        )


def test_concurrent_job_creation_allows_only_one_active_job_per_run():
    database = InMemoryDatabase()

    def create_job(index):
        job = ResearchJob(job_id=f"concurrent-job-{index}", run_id="concurrent-run")
        try:
            database.save_research_job(job)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(create_job, range(8)))

    assert sum(outcomes) == 1
    assert len(database.list_research_jobs_for_run("concurrent-run")) == 1


def test_enqueue_idempotency_key_conflict_across_runs_returns_conflict():
    client, _ = build_client()
    first_run = create_run(client)
    first = client.post(
        f"/research/runs/{first_run}/jobs",
        headers={"Idempotency-Key": "globally-unique-request"},
    )
    assert first.status_code == 202

    second_run = create_run(client)
    second = client.post(
        f"/research/runs/{second_run}/jobs",
        headers={"Idempotency-Key": "globally-unique-request"},
    )

    assert second.status_code == 409


def test_queue_stats_endpoint_returns_zeroed_summary_for_empty_queue():
    client, _ = build_client()

    response = client.get("/research/jobs/queue/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total": 0,
        "queued": 0,
        "running": 0,
        "cancel_requested": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
        "expired_leases": 0,
        "active_workers": 0,
        "oldest_queued_at": None,
    }


def test_queue_stats_counts_statuses_and_distinct_active_workers():
    database = InMemoryDatabase()
    now = datetime.now(timezone.utc)
    jobs = [
        ResearchJob(job_id="stats-queued-old", run_id="stats-run-1", created_at=now - timedelta(minutes=10)),
        ResearchJob(job_id="stats-queued-new", run_id="stats-run-2", created_at=now - timedelta(minutes=2)),
        ResearchJob(job_id="stats-running", run_id="stats-run-3", status=ResearchJobStatus.RUNNING, worker_id="worker-a"),
        ResearchJob(job_id="stats-cancel-requested", run_id="stats-run-4", status=ResearchJobStatus.CANCEL_REQUESTED, worker_id="worker-a"),
        ResearchJob(job_id="stats-succeeded", run_id="stats-run-5", status=ResearchJobStatus.SUCCEEDED),
        ResearchJob(job_id="stats-failed", run_id="stats-run-6", status=ResearchJobStatus.FAILED),
        ResearchJob(job_id="stats-cancelled", run_id="stats-run-7", status=ResearchJobStatus.CANCELLED),
    ]
    for job in jobs:
        database.save_research_job(job)

    stats = database.get_research_job_queue_stats()

    assert stats.total == 7
    assert stats.queued == 2
    assert stats.running == 1
    assert stats.cancel_requested == 1
    assert stats.succeeded == 1
    assert stats.failed == 1
    assert stats.cancelled == 1
    assert stats.active_workers == 1
    assert stats.oldest_queued_at == now - timedelta(minutes=10)


def test_queue_stats_counts_expired_running_and_cancellation_leases():
    database = InMemoryDatabase()
    expired = datetime.now(timezone.utc) - timedelta(seconds=5)
    database.save_research_job(ResearchJob(
        job_id="stats-expired-running",
        run_id="stats-expired-run-1",
        status=ResearchJobStatus.RUNNING,
        worker_id="worker-a",
        lease_expires_at=expired,
    ))
    database.save_research_job(ResearchJob(
        job_id="stats-expired-cancel",
        run_id="stats-expired-run-2",
        status=ResearchJobStatus.CANCEL_REQUESTED,
        worker_id="worker-b",
        lease_expires_at=expired,
    ))
    database.save_research_job(ResearchJob(
        job_id="stats-live",
        run_id="stats-live-run",
        status=ResearchJobStatus.RUNNING,
        worker_id="worker-c",
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))

    stats = database.get_research_job_queue_stats()

    assert stats.expired_leases == 2
    assert stats.active_workers == 3


def test_queue_stats_endpoint_reflects_persisted_jobs():
    client, database = build_client()
    database.save_research_job(ResearchJob(
        job_id="stats-api-job",
        run_id="stats-api-run",
        status=ResearchJobStatus.RUNNING,
        worker_id="api-worker",
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))

    response = client.get("/research/jobs/queue/stats")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["running"] == 1
    assert response.json()["active_workers"] == 1


def test_claim_assigns_a_new_fencing_token_even_for_the_same_worker_id():
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-fence", run_id="run-fence"))

    first = database.claim_next_research_job("reused-worker-id", lease_seconds=60)
    first_token = first.lease_token
    first.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    database.save_research_job(first)

    second = database.claim_next_research_job("reused-worker-id", lease_seconds=60)

    assert second is not None
    assert second.lease_token
    assert second.lease_token != first_token


def test_expired_lease_owner_cannot_renew_or_write_before_reclaim():
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-expired-fence", run_id="run-expired-fence"))
    claimed = database.claim_next_research_job("worker-expired", lease_seconds=60)
    token = claimed.lease_token
    claimed.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    database.save_research_job(claimed)

    assert not database.is_research_job_lease_owner(claimed.job_id, "worker-expired", token)
    assert not database.renew_research_job_lease(claimed.job_id, "worker-expired", token, 60)
    claimed.progress = 55.0
    assert not database.save_research_job_if_lease_owner(claimed, "worker-expired", token)
    assert database.get_research_job(claimed.job_id).progress == 0.0


def test_stale_worker_cannot_mutate_job_after_replacement_claim():
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-reclaimed", run_id="run-reclaimed"))
    stale = database.claim_next_research_job("same-worker-id", lease_seconds=60)
    stale_token = stale.lease_token
    stale.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    database.save_research_job(stale)

    current = database.claim_next_research_job("same-worker-id", lease_seconds=60)
    assert current.lease_token != stale_token

    stale.progress = 99.0
    assert not database.renew_research_job_lease(stale.job_id, "same-worker-id", stale_token, 60)
    assert not database.save_research_job_if_lease_owner(stale, "same-worker-id", stale_token)
    persisted = database.get_research_job(stale.job_id)
    assert persisted.lease_token == current.lease_token
    assert persisted.progress == 0.0


def test_current_lease_owner_can_renew_and_persist_fenced_updates():
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-current-fence", run_id="run-current-fence"))
    claimed = database.claim_next_research_job("current-worker", lease_seconds=60)

    assert database.is_research_job_lease_owner(
        claimed.job_id, "current-worker", claimed.lease_token
    )
    assert database.renew_research_job_lease(
        claimed.job_id, "current-worker", claimed.lease_token, 60
    )
    renewed_expiry = database.get_research_job(claimed.job_id).lease_expires_at
    claimed.progress = 25.0
    assert database.save_research_job_if_lease_owner(
        claimed, "current-worker", claimed.lease_token
    )
    persisted = database.get_research_job(claimed.job_id)
    assert persisted.progress == 25.0
    assert persisted.lease_expires_at == renewed_expiry


def test_task_executor_refuses_to_start_when_worker_lease_is_lost():
    database = InMemoryDatabase()
    run = ResearchRun(run_id="run-lost-lease", research_question="Lease fencing")
    run.transition_to(ResearchRunStatus.PLANNED)
    database.save_research_run(run)
    task = ResearchTask(task_id="task-lost-lease", description="Do not start", task_type="test")
    database.save_research_task(task)
    executor = TaskExecutor(
        tasks=[task],
        database=database,
        run_id=run.run_id,
        tool_registry=ToolRegistry(),
        should_cancel=lambda: True,
    )

    with pytest.raises(ExecutionCancelled, match="lost its active worker lease"):
        executor.execute_task(task)

    assert database.get_research_task(task.task_id).status == "pending"
    assert database.get_research_run(run.run_id).status == ResearchRunStatus.PLANNED


def test_stale_worker_failure_transition_cannot_overwrite_replacement_claim():
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-stale-failure", run_id="run-stale-failure"))
    old_claim = database.claim_next_research_job("reused-worker", lease_seconds=60)
    old_token = old_claim.lease_token
    old_claim.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    database.save_research_job(old_claim)
    replacement = database.claim_next_research_job("reused-worker", lease_seconds=60)

    worker = ResearchWorker(database, worker_id="reused-worker", experiment_adapters={})
    result = worker._finish_failed(old_claim, old_token, "late failure from stale worker")

    assert result is None
    persisted = database.get_research_job(old_claim.job_id)
    assert persisted.status == ResearchJobStatus.RUNNING
    assert persisted.lease_token == replacement.lease_token
    assert persisted.error is None
