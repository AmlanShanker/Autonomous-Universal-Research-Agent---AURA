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

- Set `LLM_PROVIDER=groq` (the default) or `LLM_PROVIDER=adaption` to select the primary LLM provider.
- Groq requires `GROQ_API_KEY`; `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`.
- Adaption Labs requires `ADAPTION_API_KEY` and `ADAPTION_MODEL`.
- Optionally set `LLM_FALLBACK_PROVIDER` to the other supported provider (`groq` or `adaption`). It must be different from `LLM_PROVIDER` and requires its own API key and model configuration.
- Leave `LLM_FALLBACK_PROVIDER` blank or unset to disable fallback configuration.
- This release phase validates and stores the fallback configuration; automatic fallback execution is introduced in a subsequent phase.
- Both providers use the existing OpenAI-compatible client; provider selection does not require changing agent code.
- Copy `.env.example` to `.env` as a starting point and fill in credentials locally.
- The API uses an in-memory database by default.
- Set `AURA_DATABASE_BACKEND=mongodb`, `MONGODB_URI`, and optionally `MONGODB_DATABASE` to use MongoDB.
- Keep credentials in the local environment; do not commit `.env` files or share secrets.

### Main endpoints

- `GET /health` — service health.
- `POST /research/plans` — generate a research plan.
- `POST /research/runs` — create a plan, task graph, and planned run.
- `GET /research/runs/{run_id}` — inspect run state.
- `GET /research/runs/{run_id}/tasks` — inspect the run's tasks.
- `POST /research/runs/{run_id}/execute` — execute the task graph synchronously.
- `GET /research/runs/{run_id}/experiments` — retrieve experiment results.
- `POST /research/runs/{run_id}/report` — build and persist a research report.
- `GET /research/reports/{report_id}` — retrieve a saved report.

Run execution is synchronous in this phase. Long-running/background worker execution and job queue infrastructure are intentionally reserved for Phase 7.
