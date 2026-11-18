# AURA

## Autonomous Universal Research Agent

AURA is an agentic AI research system designed to autonomously:

- Plan research
- Gather evidence
- Run experiments
- Identify missing capabilities
- Build and validate its own tools
- Learn from previous research
- Critique its own conclusions
- Produce reproducible research artifacts

## Vision

AURA should eventually allow a user to provide a research question and
receive a reproducible research package containing the research plan,
sources, experiments, generated tools, results, evaluations, and final report.


## FastAPI backend

Install the project dependencies, including the API test extras:

```powershell
uv sync --all-groups
```

Start the development API server:

```powershell
uv run uvicorn api.app:app --reload
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Configuration

- `GROQ_API_KEY` is required when an endpoint first needs the default LLM client.
- The API uses an in-memory database by default.
- Set `AURA_DATABASE_BACKEND=mongodb`, `MONGODB_URI`, and optionally `MONGODB_DATABASE` to use MongoDB.
- Keep credentials in the local environment; do not commit `.env` files or share secrets.

### Main endpoints

- `GET /health` — service health.
- `POST /research/plans` — generate a research plan.
- `POST /research/runs` — create a plan, task graph, and planned run.
- `GET /research/runs/{run_id}` — inspect run state.
- `GET /research/runs/{run_id}/tasks` — inspect the run's tasks.
- `POST /research/runs/{run_id}/execute` — execute the task graph synchronously (development/compatibility path).
- `POST /research/runs/{run_id}/jobs` — enqueue a durable background execution job (optional `Idempotency-Key` header).
- `GET /research/jobs/{job_id}` — inspect job state and task progress.
- `POST /research/jobs/{job_id}/cancel` — cancel a queued job or request cooperative cancellation of a running job.
- `POST /research/jobs/{job_id}/retry` — requeue a failed job within its bounded retry budget.

### Background worker

Background jobs are persisted in the same MongoDB database as research artifacts. The API process only enqueues jobs; a separate worker claims them atomically using MongoDB and executes them through the existing `TaskExecutor`.

Configure the API and worker with the same settings:

- `AURA_DATABASE_BACKEND=mongodb`
- `MONGODB_URI`
- `MONGODB_DATABASE` (optional; defaults to `aura`)
- `GROQ_API_KEY` when the research engine needs the default LLM client

Start the API in one terminal:

```powershell
$env:AURA_DATABASE_BACKEND = "mongodb"
uv run uvicorn api.app:app --reload
```

Start one or more workers in separate terminals:

```powershell
$env:AURA_DATABASE_BACKEND = "mongodb"
uv run python -m workers.research_worker
```

The standalone worker requires MongoDB because an in-memory database cannot be shared between separate processes. In-memory job storage remains available for unit tests and embedded development. Workers use leases for crash recovery, bounded attempts, and cooperative cancellation between tasks. Since a worker may be restarted after a crash, external tool operations should be designed to tolerate at-least-once execution.

Only application-registered trusted tools and experiment adapters execute. LLM-generated source code is never executed. Cancellation cannot interrupt an external tool call already in progress.
- `GET /research/runs/{run_id}/experiments` — retrieve experiment results.
- `POST /research/runs/{run_id}/report` — build and persist a research report.
- `GET /research/reports/{report_id}` — retrieve a saved report.

Run execution is synchronous in this phase. Long-running/background worker execution and job queue infrastructure are intentionally reserved for Phase 7.
