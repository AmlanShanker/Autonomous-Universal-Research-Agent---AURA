import os
import uuid
from datetime import datetime, timezone
from collections.abc import Callable

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agents.experiment_adapters import build_default_experiment_adapters
from agents.llm_client import LLMClient
from agents.research_director import ResearchDirector
from agents.research_reporter import ResearchReportBuilder
from agents.task_executor import TaskExecutor
from storage.database.base import Database
from storage.database.memory import InMemoryDatabase
from storage.database.mongo import MongoDatabase
from storage.models.experiment import ExperimentResult
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.report import ResearchReport
from storage.models.research import ResearchPlan, ResearchRun, ResearchTask
from tools.registry import ToolRegistry


class ResearchQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ResearchRunCreated(BaseModel):
    run: ResearchRun
    plan: ResearchPlan
    tasks: list[ResearchTask]


def _default_database() -> Database:
    backend = os.getenv("AURA_DATABASE_BACKEND", "memory").strip().lower()
    if backend == "mongodb":
        uri = os.getenv("MONGODB_URI")
        database_name = os.getenv("MONGODB_DATABASE", "aura")
        if not uri:
            raise RuntimeError("MONGODB_URI is required when AURA_DATABASE_BACKEND=mongodb.")
        return MongoDatabase(uri, database_name)
    if backend != "memory":
        raise RuntimeError("AURA_DATABASE_BACKEND must be 'memory' or 'mongodb'.")
    return InMemoryDatabase()


def create_app(
    database: Database | None = None,
    llm_client=None,
    director: ResearchDirector | None = None,
    tool_registry_factory: Callable[[Database], ToolRegistry] | None = None,
    experiment_adapters: dict | None = None,
) -> FastAPI:
    """Create an injectable AURA API application for production and tests."""
    db = database or _default_database()
    adapters = experiment_adapters if experiment_adapters is not None else build_default_experiment_adapters()
    registry_factory = tool_registry_factory or (lambda store: ToolRegistry(database=store))
    configured_director = director

    def get_director() -> ResearchDirector:
        nonlocal configured_director
        if configured_director is None:
            client = llm_client or LLMClient()
            configured_director = ResearchDirector(
                database=db,
                llm_client=client,
                available_experiment_adapters=list(adapters),
            )
        return configured_director

    app = FastAPI(
        title="AURA API",
        description="HTTP interface for AURA's autonomous research engine.",
        version="0.1.0",
    )
    app.state.database = db

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "aura-api"}

    @app.post("/research/plans", response_model=ResearchPlan, tags=["research"])
    def create_research_plan(request: ResearchQuestionRequest) -> ResearchPlan:
        try:
            return get_director().create_plan(request.question.strip())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/research/runs", response_model=ResearchRunCreated, status_code=201, tags=["research"])
    def create_research_run(request: ResearchQuestionRequest) -> ResearchRunCreated:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Research question cannot be blank.")
        try:
            research_director = get_director()
            plan = research_director.create_plan(question)
            tasks = research_director.create_tasks(plan)
            run = research_director.start_run(
                run_id=str(uuid.uuid4()),
                question=question,
            )
            run.tasks = [task.task_id for task in tasks]
            db.save_research_run(run)
            return ResearchRunCreated(run=run, plan=plan, tasks=tasks)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/research/runs/{run_id}", response_model=ResearchRun, tags=["research"])
    def get_research_run(run_id: str) -> ResearchRun:
        run = db.get_research_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return run

    @app.get("/research/runs/{run_id}/tasks", response_model=list[ResearchTask], tags=["research"])
    def get_research_tasks(run_id: str) -> list[ResearchTask]:
        run = db.get_research_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        tasks = [db.get_research_task(task_id) for task_id in run.tasks]
        return [task for task in tasks if task is not None]

    @app.post("/research/runs/{run_id}/execute", response_model=ResearchRun, tags=["execution"])
    def execute_research_run(run_id: str) -> ResearchRun:
        run = db.get_research_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        if run.status.value not in {"planned", "blocked"}:
            raise HTTPException(
                status_code=409,
                detail=f"Run cannot execute from status '{run.status.value}'.",
            )
        tasks = [db.get_research_task(task_id) for task_id in run.tasks]
        resolved_tasks = [task for task in tasks if task is not None]
        if len(resolved_tasks) != len(run.tasks) or not resolved_tasks:
            raise HTTPException(status_code=409, detail="Run task graph is missing or incomplete.")
        try:
            executor = TaskExecutor(
                tasks=resolved_tasks,
                database=db,
                run_id=run_id,
                tool_registry=registry_factory(db),
                experiment_adapters=adapters,
            )
            executor.run()
        except (ValueError, RuntimeError) as exc:
            latest = db.get_research_run(run_id)
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "run_status": latest.status.value if latest else "unknown"},
            ) from exc
        latest = db.get_research_run(run_id)
        return latest or run

    @app.post("/research/runs/{run_id}/jobs", response_model=ResearchJob, status_code=202, tags=["execution"])
    def enqueue_research_run(
        run_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ResearchJob:
        run = db.get_research_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")

        if idempotency_key:
            try:
                existing = db.get_research_job_by_idempotency_key(idempotency_key.strip())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if existing is not None:
                if existing.run_id != run_id:
                    raise HTTPException(status_code=409, detail="Idempotency key is already associated with another run.")
                return existing

        active = [
            job for job in db.list_research_jobs_for_run(run_id)
            if job.status in {ResearchJobStatus.QUEUED, ResearchJobStatus.RUNNING, ResearchJobStatus.CANCEL_REQUESTED}
        ]
        if active:
            raise HTTPException(status_code=409, detail={"message": "This research run already has an active job.", "job_id": active[0].job_id})

        if run.status.value not in {"planned", "blocked"}:
            raise HTTPException(status_code=409, detail=f"Run cannot be queued from status '{run.status.value}'.")
        if not run.tasks or any(db.get_research_task(task_id) is None for task_id in run.tasks):
            raise HTTPException(status_code=409, detail="Run task graph is missing or incomplete.")

        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            run_id=run_id,
            total_tasks=len(run.tasks),
            idempotency_key=idempotency_key,
        )
        try:
            db.save_research_job(job)
        except Exception as exc:
            if idempotency_key:
                existing = db.get_research_job_by_idempotency_key(idempotency_key.strip())
                if existing is not None and existing.run_id == run_id:
                    return existing
            raise HTTPException(status_code=409, detail="Could not create a unique research job.") from exc
        return job

    @app.get("/research/jobs/{job_id}", response_model=ResearchJob, tags=["execution"])
    def get_research_job(job_id: str) -> ResearchJob:
        job = db.get_research_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        return job

    @app.post("/research/jobs/{job_id}/cancel", response_model=ResearchJob, tags=["execution"])
    def cancel_research_job(job_id: str) -> ResearchJob:
        job = db.get_research_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        if job.status == ResearchJobStatus.QUEUED:
            job.status = ResearchJobStatus.CANCELLED
            job.finished_at = datetime.now(timezone.utc)
        elif job.status == ResearchJobStatus.RUNNING:
            job.status = ResearchJobStatus.CANCEL_REQUESTED
        elif job.status == ResearchJobStatus.CANCEL_REQUESTED:
            return job
        else:
            raise HTTPException(status_code=409, detail=f"Job cannot be cancelled from status '{job.status.value}'.")
        db.save_research_job(job)
        return job

    @app.post("/research/jobs/{job_id}/retry", response_model=ResearchJob, tags=["execution"])
    def retry_research_job(job_id: str) -> ResearchJob:
        job = db.get_research_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        if job.status != ResearchJobStatus.FAILED:
            raise HTTPException(status_code=409, detail="Only failed jobs can be retried.")
        if job.max_attempts >= 5:
            raise HTTPException(status_code=409, detail="The maximum manual retry budget has been reached.")
        if job.attempts >= job.max_attempts:
            job.max_attempts += 1
        run = db.get_research_run(job.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        if run.status.value not in {"planned", "blocked", "failed"}:
            raise HTTPException(status_code=409, detail=f"Run cannot be retried from status '{run.status.value}'.")
        job.status = ResearchJobStatus.QUEUED
        job.error = None
        job.finished_at = None
        job.worker_id = None
        job.lease_expires_at = None
        db.save_research_job(job)
        return job

    @app.get("/research/runs/{run_id}/experiments", response_model=list[ExperimentResult], tags=["experiments"])
    def get_run_experiments(run_id: str) -> list[ExperimentResult]:
        if db.get_research_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return db.list_experiment_results_for_run(run_id)

    @app.post("/research/runs/{run_id}/report", response_model=ResearchReport, tags=["reports"])
    def create_run_report(run_id: str) -> ResearchReport:
        run = db.get_research_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        plan = db.get_research_plan(run.research_question)
        if plan is None:
            raise HTTPException(status_code=409, detail="Research plan for this run was not found.")
        tasks = [db.get_research_task(task_id) for task_id in run.tasks]
        report = ResearchReportBuilder().build_and_save(
            db,
            run,
            plan,
            [task for task in tasks if task is not None],
            db.list_execution_results_for_run(run_id),
            db.list_experiment_results_for_run(run_id),
        )
        return report

    @app.get("/research/reports/{report_id}", response_model=ResearchReport, tags=["reports"])
    def get_research_report(report_id: str) -> ResearchReport:
        report = db.get_research_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Research report not found.")
        return report

    return app


app = create_app()
