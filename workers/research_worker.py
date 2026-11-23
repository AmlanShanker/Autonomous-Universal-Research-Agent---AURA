import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from contextvars import copy_context
from threading import Event, Thread
from collections.abc import Callable

from agents.experiment_adapters import build_default_experiment_adapters
from agents.llm_client import LLMClient
from agents.task_executor import ExecutionCancelled, TaskExecutor
from storage.database.base import Database
from storage.database.mongo import MongoDatabase
from config.database import database_uri_from_environment
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.job_attempt import ResearchJobAttempt, ResearchJobAttemptStatus
from storage.models.research import ResearchRunStatus
from tools.registry import ToolRegistry, build_default_tool_registry
from observability import METRICS, bind_context, configure_logging, current_context, new_trace_id, span


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
        configure_logging()
        self.database = database
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.tool_registry_factory = tool_registry_factory or build_default_tool_registry
        self.experiment_adapters = experiment_adapters if experiment_adapters is not None else build_default_experiment_adapters()
        self.lease_seconds = max(60, lease_seconds)

    def process_next(self) -> ResearchJob | None:
        job = self.database.claim_next_research_job(self.worker_id, self.lease_seconds)
        if job is None:
            return None
        lease_token = job.lease_token
        if not lease_token:
            return self.database.get_research_job(job.job_id)

        with bind_context(trace_id=new_trace_id(), job_id=job.job_id, run_id=job.run_id, worker_id=self.worker_id):
            with span("worker.process_job", component="worker"):
                return self._process_claimed_job(job, lease_token)

    def _process_claimed_job(self, job: ResearchJob, lease_token: str) -> ResearchJob | None:
        attempt = self._begin_attempt(job)
        outcome: ResearchJobStatus | None = None
        failure_type: str | None = None
        try:
            outcome = self._execute(job, lease_token)
        except Exception as exc:
            import logging
            failure_type = type(exc).__name__
            logging.getLogger("aura.worker").exception(
                "Research job attempt failed",
                extra={"event": "worker.job_failed", "error_type": failure_type},
            )
            outcome = self._finish_failed(job, lease_token, str(exc))
        finally:
            if outcome is None:
                attempt_status = ResearchJobAttemptStatus.LEASE_EXPIRED
                attempt_error = "Worker lease ownership was lost before the attempt completed."
            elif outcome == ResearchJobStatus.SUCCEEDED:
                attempt_status = ResearchJobAttemptStatus.SUCCEEDED
                attempt_error = None
            elif outcome == ResearchJobStatus.CANCELLED:
                attempt_status = ResearchJobAttemptStatus.CANCELLED
                attempt_error = None
            else:
                attempt_status = ResearchJobAttemptStatus.FAILED
                latest = self.database.get_research_job(job.job_id)
                attempt_error = latest.error if latest is not None else None
            attempt.finish(
                attempt_status,
                attempt_error,
                error_type=failure_type,
                trace_id=current_context().get("trace_id"),
            )
            self.database.save_research_job_attempt(attempt)
            METRICS.increment(
                "aura_worker_jobs",
                outcome=outcome.value if outcome is not None else "lease_lost",
            )
        return self.database.get_research_job(job.job_id)
    def _begin_attempt(self, job: ResearchJob) -> ResearchJobAttempt:
        # A previous RUNNING record means its worker disappeared before it
        # could close the attempt; the new lease claim proves that attempt expired.
        for previous in self.database.list_research_job_attempts(job.job_id):
            if previous.status == ResearchJobAttemptStatus.RUNNING:
                previous.finish(
                    ResearchJobAttemptStatus.LEASE_EXPIRED,
                    "Worker lease expired before the attempt completed.",
                )
                self.database.save_research_job_attempt(previous)

        attempt = ResearchJobAttempt(
            attempt_id=f"{job.job_id}:{job.attempts}",
            job_id=job.job_id,
            attempt_number=job.attempts,
            worker_id=self.worker_id,
        )
        self.database.save_research_job_attempt(attempt)
        return attempt

    def _owns_lease(self, job: ResearchJob, lease_token: str) -> bool:
        return self.database.is_research_job_lease_owner(
            job.job_id, self.worker_id, lease_token
        )

    def _save_owned(self, job: ResearchJob, lease_token: str) -> bool:
        return self.database.save_research_job_if_lease_owner(
            job, self.worker_id, lease_token
        )

    def _execute(self, job: ResearchJob, lease_token: str) -> ResearchJobStatus | None:
        if not self._owns_lease(job, lease_token):
            return None
        run = self.database.get_research_run(job.run_id)
        if run is None:
            return self._finish_failed(job, lease_token, "Research run was not found.")

        if run.status == ResearchRunStatus.COMPLETED:
            latest_job = self.database.get_research_job(job.job_id) or job
            latest_job.status = ResearchJobStatus.SUCCEEDED
            latest_job.progress = 100.0
            latest_job.completed_tasks = len(run.completed_tasks)
            latest_job.total_tasks = len(run.tasks)
            latest_job.finished_at = datetime.now(timezone.utc)
            latest_job.worker_id = None
            latest_job.lease_expires_at = None
            latest_job.lease_token = None
            return ResearchJobStatus.SUCCEEDED if self._save_owned(latest_job, lease_token) else None
        if run.status == ResearchRunStatus.CANCELLED:
            latest_job = self.database.get_research_job(job.job_id) or job
            latest_job.status = ResearchJobStatus.CANCELLED
            latest_job.finished_at = datetime.now(timezone.utc)
            latest_job.worker_id = None
            latest_job.lease_expires_at = None
            latest_job.lease_token = None
            return ResearchJobStatus.CANCELLED if self._save_owned(latest_job, lease_token) else None

        if run.status == ResearchRunStatus.FAILED:
            if not self._owns_lease(job, lease_token):
                return None
            run.status = ResearchRunStatus.PLANNED
            run.failed_tasks = []
            self.database.save_research_run(run)

        tasks = [self.database.get_research_task(task_id) for task_id in run.tasks]
        resolved_tasks = [task for task in tasks if task is not None]
        if not resolved_tasks or len(resolved_tasks) != len(run.tasks):
            return self._finish_failed(job, lease_token, "Research run task graph is missing or incomplete.")

        # A worker may have died after persisting a task as running. Re-queue
        # that task for at-least-once recovery; completed tasks remain untouched.
        for task in resolved_tasks:
            if not self._owns_lease(job, lease_token):
                return None
            if task.status in {"running", "failed", "blocked"}:
                task.status = "pending"
                self.database.save_research_task(task)

        lease_lost = Event()

        def cancellation_requested() -> bool:
            if lease_lost.is_set() or not self._owns_lease(job, lease_token):
                lease_lost.set()
                return True
            latest = self.database.get_research_job(job.job_id)
            return latest is None or latest.status == ResearchJobStatus.CANCEL_REQUESTED

        def update_progress(completed: int, total: int) -> None:
            if cancellation_requested():
                return
            latest = self.database.get_research_job(job.job_id)
            if latest is None or latest.status == ResearchJobStatus.CANCEL_REQUESTED:
                return
            latest.completed_tasks = completed
            latest.total_tasks = total
            latest.progress = round((completed / total) * 100, 2) if total else 0.0
            if not self._save_owned(latest, lease_token):
                lease_lost.set()
                return
            if not self.database.renew_research_job_lease(
                job.job_id, self.worker_id, lease_token, self.lease_seconds
            ):
                lease_lost.set()

        heartbeat_stop = Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(max(10, self.lease_seconds // 3)):
                try:
                    renewed = self.database.renew_research_job_lease(
                        job.job_id, self.worker_id, lease_token, self.lease_seconds
                    )
                except Exception as exc:
                    METRICS.increment("aura_worker_lease_renewals", outcome="error")
                    logging.getLogger("aura.worker").exception(
                        "Worker lease renewal failed",
                        extra={"event": "worker.lease_renewal_failed", "error_type": type(exc).__name__},
                    )
                    lease_lost.set()
                    return
                if not renewed:
                    METRICS.increment("aura_worker_lease_renewals", outcome="rejected")
                    logging.getLogger("aura.worker").warning(
                        "Worker lease ownership was lost",
                        extra={"event": "worker.lease_lost"},
                    )
                    lease_lost.set()
                    return
                METRICS.increment("aura_worker_lease_renewals", outcome="success")

        heartbeat_context = copy_context()
        heartbeat_thread = Thread(
            target=heartbeat_context.run,
            args=(heartbeat,),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            requires_analysis = any(
                not task.required_tools
                and (task.dependencies or task.experiments)
                and task.task_type.strip().lower() != "test"
                and not (
                    task.task_type.strip().lower() == "experiment"
                    and task.experiments
                )
                for task in resolved_tasks
            )
            executor = TaskExecutor(
                tasks=resolved_tasks,
                database=self.database,
                run_id=job.run_id,
                tool_registry=self.tool_registry_factory(self.database),
                experiment_adapters=self.experiment_adapters,
                should_cancel=cancellation_requested,
                progress_callback=update_progress,
                llm_client=LLMClient() if requires_analysis else None,
            )
            executor.run()
        except ExecutionCancelled:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            if not self._owns_lease(job, lease_token):
                return None
            latest_run = self.database.get_research_run(job.run_id)
            if latest_run is not None and latest_run.status == ResearchRunStatus.RUNNING:
                latest_run.transition_to(ResearchRunStatus.CANCELLED)
                self.database.save_research_run(latest_run)
            latest_job = self.database.get_research_job(job.job_id) or job
            latest_job.status = ResearchJobStatus.CANCELLED
            latest_job.finished_at = datetime.now(timezone.utc)
            latest_job.lease_expires_at = None
            latest_job.worker_id = None
            latest_job.lease_token = None
            return ResearchJobStatus.CANCELLED if self._save_owned(latest_job, lease_token) else None
        except Exception:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            raise

        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        if not self._owns_lease(job, lease_token):
            return None
        latest_job = self.database.get_research_job(job.job_id) or job
        latest_run = self.database.get_research_run(job.run_id)
        if latest_job.status == ResearchJobStatus.CANCEL_REQUESTED and (
            latest_run is None or latest_run.status != ResearchRunStatus.COMPLETED
        ):
            latest_job.status = ResearchJobStatus.CANCELLED
        else:
            latest_job.status = ResearchJobStatus.SUCCEEDED
            latest_job.completed_tasks = len(resolved_tasks)
            latest_job.total_tasks = len(resolved_tasks)
            latest_job.progress = 100.0
        latest_job.finished_at = datetime.now(timezone.utc)
        latest_job.lease_expires_at = None
        latest_job.worker_id = None
        latest_job.lease_token = None
        return latest_job.status if self._save_owned(latest_job, lease_token) else None

    def _finish_failed(
        self, job: ResearchJob, lease_token: str, message: str
    ) -> ResearchJobStatus | None:
        if not self._owns_lease(job, lease_token):
            return None
        latest = self.database.get_research_job(job.job_id) or job
        latest.error = (message or "Unknown worker error.")[:2000]
        latest.lease_expires_at = None
        latest.worker_id = None
        latest.lease_token = None
        if latest.status == ResearchJobStatus.CANCEL_REQUESTED:
            run = self.database.get_research_run(job.run_id)
            if run is not None and run.status != ResearchRunStatus.COMPLETED:
                run.status = ResearchRunStatus.CANCELLED
                self.database.save_research_run(run)
            latest.status = ResearchJobStatus.CANCELLED
            latest.finished_at = datetime.now(timezone.utc)
        else:
            latest.status = (
                ResearchJobStatus.QUEUED
                if latest.attempts < latest.max_attempts
                else ResearchJobStatus.FAILED
            )
            latest.finished_at = None if latest.status == ResearchJobStatus.QUEUED else datetime.now(timezone.utc)
        return latest.status if self._save_owned(latest, lease_token) else None

    def run_forever(self, poll_interval: float = 2.0, stop_event: Event | None = None) -> None:
        stop = stop_event or Event()
        while not stop.is_set():
            job = self.process_next()
            if job is None or job.status == ResearchJobStatus.QUEUED:
                stop.wait(max(0.1, poll_interval))


def build_worker_from_environment() -> ResearchWorker:
    backend = os.getenv("AURA_DATABASE_BACKEND", "mongodb").strip().lower()
    if backend != "mongodb":
        raise RuntimeError("The standalone worker requires AURA_DATABASE_BACKEND=mongodb.")
    uri = database_uri_from_environment()
    if not uri:
        raise RuntimeError("AURA_DATABASE_URI, MONGO_URL, or MONGODB_URI is required by the standalone worker.")
    database_name = os.getenv("MONGODB_DATABASE", "aura")
    return ResearchWorker(MongoDatabase(uri, database_name))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    worker = build_worker_from_environment()
    import logging
    logging.getLogger("aura.worker").info(
        "AURA worker started", extra={"event": "worker.started", "component": "worker"}
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
