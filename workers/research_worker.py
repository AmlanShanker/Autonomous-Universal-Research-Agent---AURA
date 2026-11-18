import os
import socket
import time
import uuid
from datetime import datetime, timezone
from threading import Event, Thread
from collections.abc import Callable

from agents.experiment_adapters import build_default_experiment_adapters
from agents.task_executor import ExecutionCancelled, TaskExecutor
from storage.database.base import Database
from storage.database.mongo import MongoDatabase
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.research import ResearchRunStatus
from tools.registry import ToolRegistry


class ResearchWorker:
    """Poll and execute durable research jobs outside the API process."""

    def __init__(
        self,
        database: Database,
        worker_id: str | None = None,
        tool_registry_factory: Callable[[Database], ToolRegistry] | None = None,
        experiment_adapters: dict | None = None,
        lease_seconds: int = 3600,
    ):
        self.database = database
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.tool_registry_factory = tool_registry_factory or (lambda store: ToolRegistry(database=store))
        self.experiment_adapters = experiment_adapters if experiment_adapters is not None else build_default_experiment_adapters()
        self.lease_seconds = max(60, lease_seconds)

    def process_next(self) -> ResearchJob | None:
        job = self.database.claim_next_research_job(self.worker_id, self.lease_seconds)
        if job is None:
            return None
        self._execute(job)
        return self.database.get_research_job(job.job_id)

    def _execute(self, job: ResearchJob) -> None:
        run = self.database.get_research_run(job.run_id)
        if run is None:
            self._finish_failed(job, "Research run was not found.")
            return

        if run.status == ResearchRunStatus.FAILED:
            run.status = ResearchRunStatus.PLANNED
            run.failed_tasks = []
            self.database.save_research_run(run)

        tasks = [self.database.get_research_task(task_id) for task_id in run.tasks]
        resolved_tasks = [task for task in tasks if task is not None]
        if not resolved_tasks or len(resolved_tasks) != len(run.tasks):
            self._finish_failed(job, "Research run task graph is missing or incomplete.")
            return

        # A worker may have died after persisting a task as running. Re-queue
        # that task for at-least-once recovery; completed tasks remain untouched.
        for task in resolved_tasks:
            if task.status in {"running", "failed", "blocked"}:
                task.status = "pending"
                self.database.save_research_task(task)

        def cancellation_requested() -> bool:
            latest = self.database.get_research_job(job.job_id)
            return latest is None or latest.status == ResearchJobStatus.CANCEL_REQUESTED

        def update_progress(completed: int, total: int) -> None:
            latest = self.database.get_research_job(job.job_id)
            if latest is None or latest.status == ResearchJobStatus.CANCEL_REQUESTED:
                return
            latest.completed_tasks = completed
            latest.total_tasks = total
            latest.progress = round((completed / total) * 100, 2) if total else 0.0
            self.database.save_research_job(latest)
            self.database.renew_research_job_lease(job.job_id, self.worker_id, self.lease_seconds)

        heartbeat_stop = Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(max(10, self.lease_seconds // 3)):
                if not self.database.renew_research_job_lease(job.job_id, self.worker_id, self.lease_seconds):
                    return

        heartbeat_thread = Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            executor = TaskExecutor(
                tasks=resolved_tasks,
                database=self.database,
                run_id=job.run_id,
                tool_registry=self.tool_registry_factory(self.database),
                experiment_adapters=self.experiment_adapters,
                should_cancel=cancellation_requested,
                progress_callback=update_progress,
            )
            executor.run()
        except ExecutionCancelled:
            latest_run = self.database.get_research_run(job.run_id)
            if latest_run is not None and latest_run.status == ResearchRunStatus.RUNNING:
                latest_run.transition_to(ResearchRunStatus.CANCELLED)
                self.database.save_research_run(latest_run)
            latest_job = self.database.get_research_job(job.job_id) or job
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            latest_job.status = ResearchJobStatus.CANCELLED
            latest_job.finished_at = datetime.now(timezone.utc)
            latest_job.lease_expires_at = None
            latest_job.worker_id = None
            self.database.save_research_job(latest_job)
            return
        except Exception as exc:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            self._finish_failed(job, str(exc))
            return

        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        latest_job = self.database.get_research_job(job.job_id) or job
        if latest_job.status == ResearchJobStatus.CANCEL_REQUESTED:
            latest_job.status = ResearchJobStatus.CANCELLED
        else:
            latest_job.status = ResearchJobStatus.SUCCEEDED
            latest_job.completed_tasks = len(resolved_tasks)
            latest_job.total_tasks = len(resolved_tasks)
            latest_job.progress = 100.0
        latest_job.finished_at = datetime.now(timezone.utc)
        latest_job.lease_expires_at = None
        latest_job.worker_id = None
        self.database.save_research_job(latest_job)

    def _finish_failed(self, job: ResearchJob, message: str) -> None:
        latest = self.database.get_research_job(job.job_id) or job
        latest.error = (message or "Unknown worker error.")[:2000]
        latest.lease_expires_at = None
        latest.worker_id = None
        latest.status = (
            ResearchJobStatus.QUEUED
            if latest.attempts < latest.max_attempts
            else ResearchJobStatus.FAILED
        )
        latest.finished_at = None if latest.status == ResearchJobStatus.QUEUED else datetime.now(timezone.utc)
        self.database.save_research_job(latest)

    def run_forever(self, poll_interval: float = 2.0, stop_event: Event | None = None) -> None:
        stop = stop_event or Event()
        while not stop.is_set():
            job = self.process_next()
            if job is None:
                stop.wait(max(0.1, poll_interval))


def build_worker_from_environment() -> ResearchWorker:
    backend = os.getenv("AURA_DATABASE_BACKEND", "mongodb").strip().lower()
    if backend != "mongodb":
        raise RuntimeError("The standalone worker requires AURA_DATABASE_BACKEND=mongodb.")
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is required by the standalone worker.")
    database_name = os.getenv("MONGODB_DATABASE", "aura")
    return ResearchWorker(MongoDatabase(uri, database_name))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    worker = build_worker_from_environment()
    print(f"AURA worker {worker.worker_id} started.")
    worker.run_forever()


if __name__ == "__main__":
    main()
