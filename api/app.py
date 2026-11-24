import os
import re
import uuid
from datetime import datetime, timezone
from collections.abc import Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from agents.experiment_adapters import (
    build_default_experiment_adapters,
    describe_default_experiment_adapters,
)
from agents.llm_client import LLMClient
from agents.research_director import ResearchDirector
from agents.research_reporter import ResearchReportBuilder
from agents.task_executor import TaskExecutor
from agents.autonomous_improvement import AutonomousImprovementEngine
from storage.database.base import Database
from storage.database.memory import InMemoryDatabase
from storage.database.mongo import MongoDatabase
from config.database import database_uri_from_environment
from storage.models.experiment import ExperimentResult
from storage.models.improvement import ImprovementCycle, ImprovementProposal
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.job_attempt import ResearchJobAttempt
from storage.models.job_stats import ResearchJobQueueStats
from storage.models.report import ResearchReport
from storage.models.execution import ExecutionResult
from storage.models.research import ResearchPlan, ResearchRun, ResearchTask
from api.security import APIKeyAuthenticator, InMemoryRateLimiter, client_identity
from observability import (
    METRICS, bind_context, configure_logging, current_span_id, span, trace_id_from_header,
)
from tools.registry import (
    ToolRegistry,
    build_default_tool_registry,
    describe_default_tools,
)


load_dotenv()


def _role_level(role: str) -> int:
    return {"reader": 1, "operator": 2, "admin": 3}.get(role, 0)


class ResearchQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ResearchRunCreated(BaseModel):
    run: ResearchRun
    plan: ResearchPlan
    tasks: list[ResearchTask]


class ImprovementApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImprovementRejectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=2000)


class ImprovementValidationRequest(BaseModel):
    experiment_result_id: str = Field(min_length=1, max_length=200)


def _improvement_reviewer_identity(request: Request) -> str:
    """Return a trusted credential ID, or an explicit marker in unauthenticated local mode."""
    authenticator = request.app.state.api_authenticator
    principal = getattr(request.state, "api_principal", None)
    if authenticator.enabled and principal is not None:
        return principal.key_id
    return "unauthenticated"


def _default_database() -> Database:
    backend = os.getenv("AURA_DATABASE_BACKEND", "memory").strip().lower()
    if backend == "mongodb":
        uri = database_uri_from_environment()
        database_name = os.getenv("MONGODB_DATABASE", "aura")
        if not uri:
            raise RuntimeError(
                "AURA_DATABASE_URI, MONGO_URL, or MONGODB_URI is required when AURA_DATABASE_BACKEND=mongodb."
            )
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
    configure_logging()
    db = database or _default_database()
    adapters = experiment_adapters if experiment_adapters is not None else build_default_experiment_adapters()
    registry_factory = tool_registry_factory or build_default_tool_registry
    configured_director = director

    def get_director() -> ResearchDirector:
        nonlocal configured_director
        if configured_director is None:
            client = llm_client or LLMClient()
            configured_director = ResearchDirector(
                database=db,
                llm_client=client,
                available_experiment_adapters=list(adapters),
                available_tools=registry_factory(db).list_tools(),
                experiment_adapter_specs=describe_default_experiment_adapters(),
                available_tool_specs=describe_default_tools(),
            )
        return configured_director

    app = FastAPI(
        title="AURA API",
        description="HTTP interface for AURA's autonomous research engine.",
        version="0.1.0",
    )
    app.state.database = db
    authenticator = APIKeyAuthenticator.from_environment()
    rate_limiter = InMemoryRateLimiter.from_environment()
    app.state.api_authenticator = authenticator
    app.state.rate_limiter = rate_limiter

    @app.middleware("http")
    async def enforce_api_security(request: Request, call_next):
        trace_id = trace_id_from_header(request.headers.get("traceparent"))
        requested_request_id = request.headers.get("x-request-id", "")
        request_id = (
            requested_request_id
            if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", requested_request_id)
            else uuid.uuid4().hex
        )
        request.state.trace_id = trace_id
        request.state.request_id = request_id
        import logging
        import time
        started = time.perf_counter()
        with bind_context(trace_id=trace_id, request_id=request_id), span("api.request", component="api", operation=request.method):
            try:
                principal = authenticator.authenticate(request.headers.get("authorization"))
                request.state.api_principal = principal
                client_host = request.client.host if request.client else None
                if authenticator.enabled and principal is not None:
                    identity = f"key:{principal.key_id}"
                else:
                    identity = client_identity(client_host)

                response = None
                if authenticator.enabled:
                    parts = [part for part in request.url.path.split("/") if part]
                    if request.url.path == "/research/jobs/queue/stats":
                        response = JSONResponse(
                            status_code=403,
                            content={"detail": "Queue statistics are not tenant-scoped."},
                        )
                    else:
                        resource_run = None
                        if len(parts) >= 3 and parts[0:2] == ["research", "runs"]:
                            resource_run = db.get_research_run(parts[2])
                        elif len(parts) >= 3 and parts[0:2] == ["research", "jobs"] and parts[2] != "queue":
                            resource_job = db.get_research_job(parts[2])
                            if resource_job is not None:
                                resource_run = db.get_research_run(resource_job.run_id)
                        elif len(parts) == 3 and parts[0:2] == ["research", "reports"]:
                            resource_report = db.get_research_report(parts[2])
                            if resource_report is not None:
                                resource_run = db.get_research_run(resource_report.run_id)
                        if resource_run is not None and (
                            principal is None or resource_run.tenant_id != principal.tenant_id
                        ):
                            response = JSONResponse(
                                status_code=404,
                                content={"detail": "Research resource not found."},
                            )

                if response is None:
                    allowed, remaining, reset_seconds = rate_limiter.check(identity)
                    if not allowed:
                        response = JSONResponse(
                            status_code=429,
                            content={"detail": "API rate limit exceeded. Retry after the current window."},
                            headers={"Retry-After": str(max(1, reset_seconds)),
                                     "X-RateLimit-Limit": str(rate_limiter.limit),
                                     "X-RateLimit-Remaining": "0"},
                        )
                    elif authenticator.enabled and request.url.path != "/health" and principal is None:
                        response = JSONResponse(
                            status_code=401,
                            content={"detail": "A valid bearer API key is required."},
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    elif (
                        authenticator.enabled
                        and request.url.path != "/health"
                        and _role_level(principal.role) < (1 if request.method in {"GET", "HEAD", "OPTIONS"} else 2)
                    ):
                        response = JSONResponse(
                            status_code=403,
                            content={"detail": "The API key role is not permitted to perform this operation."},
                        )
                    else:
                        response = await call_next(request)

                elif authenticator.enabled and request.url.path != "/health" and principal is None:
                    response = JSONResponse(
                        status_code=401,
                        content={"detail": "A valid bearer API key is required."},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                elif (
                    authenticator.enabled
                    and request.url.path != "/health"
                    and _role_level(principal.role) < (1 if request.method in {"GET", "HEAD", "OPTIONS"} else 2)
                ):
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "The API key role is not permitted to perform this operation."},
                    )
                else:
                    response = await call_next(request)

                route = request.scope.get("route")
                route_name = getattr(route, "path", "unmatched")
                response.headers["X-RateLimit-Limit"] = str(rate_limiter.limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset-Seconds"] = str(reset_seconds)
                response.headers["X-Request-ID"] = request_id
                response.headers["traceparent"] = f"00-{trace_id}-{current_span_id() or uuid.uuid4().hex[:16]}-01"
                METRICS.increment("aura_http_requests", method=request.method, route=route_name, status=str(response.status_code))
                METRICS.observe("aura_http_request_duration_ms", (time.perf_counter() - started) * 1000, method=request.method, route=route_name)
                logging.getLogger("aura.api").info(
                    "HTTP request completed",
                    extra={"event": "http.request", "method": request.method,
                           "route": route_name, "status_code": response.status_code},
                )
                return response
            except Exception as exc:
                route = request.scope.get("route")
                route_name = getattr(route, "path", "unmatched")
                METRICS.increment("aura_http_requests", method=request.method, route=route_name, status="500")
                METRICS.observe("aura_http_request_duration_ms", (time.perf_counter() - started) * 1000, method=request.method, route=route_name)
                logging.getLogger("aura.api").exception(
                    "HTTP request failed",
                    extra={"event": "http.error", "method": request.method,
                           "route": route_name, "error_type": type(exc).__name__},
                )
                raise

    @app.get("/metrics", include_in_schema=False, tags=["system"])
    def metrics() -> PlainTextResponse:
        """Prometheus-compatible process metrics; protect exposure at the API gateway."""
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

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
    def create_research_run(request: ResearchQuestionRequest, raw_request: Request) -> ResearchRunCreated:
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
            principal = getattr(raw_request.state, "api_principal", None)
            run.tenant_id = principal.tenant_id if principal is not None else "development"
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
                database=db,
                run_id=run_id,
                tool_registry=registry_factory(db),
                experiment_adapters=adapters,
                llm_client=(llm_client or LLMClient()) if requires_analysis else None,
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

        normalized_key = idempotency_key.strip() if idempotency_key is not None else None
        if idempotency_key is not None and (not normalized_key or len(normalized_key) > 200):
            raise HTTPException(status_code=422, detail="Idempotency-Key must contain 1 to 200 characters.")
        if normalized_key:
            try:
                existing = db.get_research_job_by_idempotency_key(normalized_key)
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
            idempotency_key=normalized_key,
        )
        try:
            db.save_research_job(job)
        except Exception as exc:
            if normalized_key:
                existing = db.get_research_job_by_idempotency_key(normalized_key)
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

    @app.get("/research/jobs/queue/stats", response_model=ResearchJobQueueStats, tags=["execution"])
    def get_research_job_queue_stats() -> ResearchJobQueueStats:
        return db.get_research_job_queue_stats()

    @app.get("/research/jobs/{job_id}/attempts", response_model=list[ResearchJobAttempt], tags=["execution"])
    def get_research_job_attempts(job_id: str) -> list[ResearchJobAttempt]:
        if db.get_research_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        return db.list_research_job_attempts(job_id)

    @app.get("/research/jobs/{job_id}/diagnostics", tags=["execution"])
    def get_research_job_diagnostics(job_id: str) -> dict:
        """Assemble durable worker-attempt and task-checkpoint failure context."""
        job = db.get_research_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        run = db.get_research_run(job.run_id)
        task_diagnostics = []
        if run is not None:
            for task_id in run.tasks:
                task = db.get_research_task(task_id)
                checkpoint = db.get_task_checkpoint(job.run_id, task_id)
                task_diagnostics.append({
                    "task_id": task_id,
                    "status": task.status if task is not None else "missing",
                    "checkpoint_status": checkpoint.status if checkpoint is not None else None,
                    "checkpoint_error": checkpoint.error if checkpoint is not None else None,
                })
        return {
            "job_id": job_id,
            "run_id": job.run_id,
            "status": job.status.value,
            "error": job.error,
            "attempts": [attempt.model_dump(mode="json") for attempt in db.list_research_job_attempts(job_id)],
            "tasks": task_diagnostics,
        }

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
        if job.attempts >= job.max_attempts:
            if job.max_attempts >= 5:
                raise HTTPException(status_code=409, detail="The maximum manual retry budget has been reached.")
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

    @app.get("/research/runs/{run_id}/executions", response_model=list[ExecutionResult], tags=["execution"])
    def get_run_executions(run_id: str) -> list[ExecutionResult]:
        if db.get_research_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return db.list_execution_results_for_run(run_id)

    @app.get("/research/runs/{run_id}/experiments", response_model=list[ExperimentResult], tags=["experiments"])
    def get_run_experiments(run_id: str) -> list[ExperimentResult]:
        if db.get_research_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return db.list_experiment_results_for_run(run_id)

    @app.post(
        "/research/runs/{run_id}/improvement-cycles",
        response_model=ImprovementCycle,
        status_code=201,
        tags=["improvement"],
    )
    def create_improvement_cycle(run_id: str) -> ImprovementCycle:
        if db.get_research_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        try:
            engine = AutonomousImprovementEngine(db, registry_factory(db))
            return engine.analyze_run(run_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get(
        "/research/runs/{run_id}/improvement-cycles",
        response_model=list[ImprovementCycle],
        tags=["improvement"],
    )
    def list_improvement_cycles(run_id: str) -> list[ImprovementCycle]:
        if db.get_research_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return db.list_improvement_cycles_for_run(run_id)

    @app.get(
        "/research/improvement-cycles/{cycle_id}",
        response_model=ImprovementCycle,
        tags=["improvement"],
    )
    def get_improvement_cycle(cycle_id: str) -> ImprovementCycle:
        cycle = db.get_improvement_cycle(cycle_id)
        if cycle is None:
            raise HTTPException(status_code=404, detail="Improvement cycle not found.")
        return cycle

    @app.get(
        "/research/improvement-cycles/{cycle_id}/proposals",
        response_model=list[ImprovementProposal],
        tags=["improvement"],
    )
    def list_improvement_proposals(cycle_id: str) -> list[ImprovementProposal]:
        if db.get_improvement_cycle(cycle_id) is None:
            raise HTTPException(status_code=404, detail="Improvement cycle not found.")
        return db.list_improvement_proposals_for_cycle(cycle_id)

    @app.post(
        "/research/improvement-proposals/{proposal_id}/approve",
        response_model=ImprovementProposal,
        tags=["improvement"],
    )
    def approve_improvement(
        proposal_id: str,
        request: Request,
        payload: ImprovementApprovalRequest | None = None,
    ) -> ImprovementProposal:
        try:
            return AutonomousImprovementEngine(db, registry_factory(db)).approve(
                proposal_id, _improvement_reviewer_identity(request)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/research/improvement-proposals/{proposal_id}/reject",
        response_model=ImprovementProposal,
        tags=["improvement"],
    )
    def reject_improvement(
        proposal_id: str, payload: ImprovementRejectionRequest, request: Request
    ) -> ImprovementProposal:
        try:
            return AutonomousImprovementEngine(db, registry_factory(db)).reject(
                proposal_id, _improvement_reviewer_identity(request), payload.reason
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/research/improvement-proposals/{proposal_id}/validate",
        response_model=ImprovementProposal,
        tags=["improvement"],
    )
    def validate_improvement(proposal_id: str, request: ImprovementValidationRequest) -> ImprovementProposal:
        try:
            return AutonomousImprovementEngine(db, registry_factory(db)).validate(
                proposal_id, request.experiment_result_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
