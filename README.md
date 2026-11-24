# AURA


## Command-line interface

After installing the project, the `aura` command can start the API or background worker:

```powershell
aura api
aura api --host 0.0.0.0 --port 8000
aura worker
```

The worker requires the configured persistent MongoDB backend. The API defaults to local development settings unless deployment configuration enables authentication.


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
- Transient connection failures, timeouts, rate limits, and server errors are retried up to `LLM_RETRY_ATTEMPTS` times per provider (default `2`) with exponential backoff controlled by `LLM_RETRY_BACKOFF_SECONDS` (default `0.5` seconds). Set retries to `0` to disable them. The OpenAI SDK's internal retries are disabled so this limit remains predictable.
- `LLM_TIMEOUT_SECONDS` sets the request timeout for both providers (default `30` seconds; minimum `0.1`).
- After primary retries are exhausted, a configured fallback provider receives its own bounded retry budget. If both providers fail, `LLMFallbackError` retains both original exceptions as `primary_error` and `fallback_error`; its message contains provider names and exception types, not credential values.
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
- `POST /research/runs/{run_id}/execute` — execute the task graph synchronously (development/compatibility path).
- `POST /research/runs/{run_id}/jobs` — enqueue a durable background execution job (optional `Idempotency-Key` header).
- `GET /research/jobs/{job_id}` — inspect job state and task progress.
- `GET /research/jobs/queue/stats` — inspect queue counts, active workers, expired leases, and oldest queued job.
- `GET /research/jobs/{job_id}/attempts` — inspect each worker claim, including successful, failed, cancelled, and lease-expired attempts.
- `POST /research/jobs/{job_id}/cancel` — cancel a queued job or request cooperative cancellation of a running job.
- `POST /research/jobs/{job_id}/retry` — requeue a failed job within its bounded retry budget.
- `GET /research/runs/{run_id}/experiments` — retrieve experiment results.
- `POST /research/runs/{run_id}/report` — build and persist a research report.
- `GET /research/reports/{report_id}` — retrieve a saved report.

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

The standalone worker requires MongoDB because an in-memory database cannot be shared between separate processes. In-memory job storage remains available for unit tests and embedded development. Workers use expiring leases for crash recovery, bounded attempts, and cooperative cancellation between tasks. Task execution also persists a per-task checkpoint keyed by run and task, including an input fingerprint and lifecycle status. A worker restart skips a task only when its checkpoint is completed and its current task definition has the same fingerprint; changed task inputs invalidate that checkpoint. Tool and internal-analysis result identities are deterministic for the same run, task, implementation, and dependency evidence, and completed results are reused. Experiment identities include the task position and declarative specification, so completed experiment results are reused while changed specifications receive new identities. Failed operations remain retryable. These protections prevent duplicate persisted results and avoid repeating completed operations after a checkpointed restart, but they cannot provide exactly-once guarantees for an external side effect if a process crashes after the side effect and before its result is durably saved; external adapters should remain safe for at-least-once invocation. Every successful claim receives a fresh, opaque fencing token, including when the same worker ID reclaims a job. Lease renewal and worker-owned job updates require the matching worker ID and token and are rejected after expiry, so an old worker cannot overwrite a replacement worker's job state. Task execution checks lease ownership before starting and before persisting task-side changes; an external tool call already in progress cannot be forcibly interrupted. Each worker claim is recorded as a durable job-attempt entry with its worker ID, timestamps, outcome, and a bounded error summary; expired attempts are marked when a replacement worker successfully claims the job. Job creation is protected against concurrent duplicate submissions: each run may have at most one queued/running/cancellation-requested job, and idempotency keys are unique across jobs. MongoDB enforces these constraints with unique indexes; the in-memory backend enforces them under its job lock. Before upgrading an existing MongoDB deployment, resolve any pre-existing duplicate active jobs for the same run so the new unique index can be created. Since a worker may be restarted after a crash, external tool operations should be designed to tolerate at-least-once execution.

Only application-registered trusted tools and experiment adapters execute. LLM-generated source code is never executed. Cancellation cannot interrupt an external tool call already in progress.


### Research source provenance and quality

Literature-search results are normalized into durable source records with a stable identity based on DOI, arXiv ID, or normalized URL. Duplicate records within a search are collapsed, preferring the record with richer bibliographic metadata. Each source records its provider, retrieval query and timestamp, identifiers, authors, abstract, publication year, and a transparent metadata-completeness credibility heuristic with per-factor reasons. The heuristic is an evidence-triage signal, not a peer-review verdict or a guarantee that a source is correct; users should assess venue quality, methodology, conflicts, and the original publication. Sources are persisted separately from task execution records in both the in-memory and MongoDB backends. MongoDB creates unique indexes for source and canonical identities.


### Experiment reproducibility

Experiment parameter inputs are saved as immutable, SHA-256-verified snapshots with monotonically increasing versions for a stable experiment identity. Changing parameters creates a new version and a distinct result identity; re-running an unchanged specification reuses its completed result. Each result records observed Python version, implementation, and platform separately from any declared environment requirements, adapter version, code revision, dependency-lock digest, and package versions. AURA records environment metadata but does not install dependencies or claim that a manifest alone guarantees bit-for-bit reproducibility. Trusted adapters may return named text artifacts; artifacts are stored separately with content hashes and a 1 MB per-artifact limit. Arbitrary generated code and executable artifacts are not run.


### Tool execution security and resource limits

AURA routes registered tool calls through an application-level execution guard. Only the explicitly allowlisted built-in adapters (`literature_search` and `dataset_download`) may run; generated tool specifications and generated implementations remain non-executable. The guard bounds serialized input (64 KiB), serialized output (4 MiB), and concurrent executions per task-executor process (4), and records resource-use metadata with successful tool results. This is not an operating-system sandbox and does not isolate a compromised Python process.

Network adapters use explicit connect/read timeouts and bounded response/download sizes. Dataset downloads reject local/private literal hosts, credentials in URLs, nonstandard ports, and redirects; downloaded files remain data and are never executed. The dataset limit is 250 MiB and arXiv responses are capped at 5 MiB. Deployments handling untrusted workloads should additionally run AURA in a restricted OS/container environment with network egress controls and filesystem permissions.


### API authentication, authorization, and rate limits

The API supports bearer API keys and role-based authorization. Authentication is disabled by default for local development; enable it for shared or deployed environments with `AURA_API_AUTH_ENABLED=true`. When enabled, the application fails at startup if `AURA_API_KEYS` is missing, malformed, or empty.

Configure `AURA_API_KEYS` as a JSON array of objects with a unique `id`, a randomly generated `key` of at least 32 characters, and a `role` (`reader`, `operator`, or `admin`). For example, the JSON value can contain `{"id":"service-a","key":"REPLACE_WITH_A_RANDOM_SECRET_OF_32_OR_MORE_CHARACTERS","role":"operator"}`. Send the key as `Authorization: Bearer YOUR_API_KEY`. Generate key material locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"`; keep it in the deployment secret manager or local environment, never in source control. AURA stores only SHA-256 digests of configured keys in process memory and does not return or log the raw key.

The `reader` role can use safe HTTP methods (GET, HEAD, OPTIONS); `operator` and `admin` can also use mutating methods. `GET /health` remains public. Invalid or missing credentials receive HTTP 401, insufficient roles receive HTTP 403, and rate-limit exhaustion receives HTTP 429 with a `Retry-After` header.

`AURA_API_RATE_LIMIT` sets the request limit per identity (default 120) and `AURA_API_RATE_WINDOW_SECONDS` sets the fixed-window duration (default 60 seconds). Authenticated requests are limited per API key id; unauthenticated traffic is limited by the direct client IP. Forwarded IP headers are deliberately not trusted. The limiter is in-memory and local to each API process, so multi-process or multi-instance deployments should enforce a shared limit at a trusted API gateway or edge proxy as well. Authentication keys must be rotated by updating deployment configuration and restarting the API.


### System observability

The API, worker, and instrumented agent components emit structured JSON logs to standard output. Set `AURA_LOG_LEVEL` to
`DEBUG`, `INFO`, `WARNING`, or `ERROR` (default: `INFO`). Log records
include available request, trace, run, job, task, and worker correlation IDs.
Prompts, tool payloads, authorization headers, and configured API keys are not
intentionally logged; common bearer tokens, key assignments, and MongoDB
credentials are redacted by the formatter.

The API accepts a W3C `traceparent` header and an optional `X-Request-ID`.
It returns the propagated trace context and request ID in response headers.
Worker jobs receive a new trace ID, which is carried into task, tool, and LLM
spans. These spans and IDs provide in-process correlation; this implementation
does not export to an external OpenTelemetry collector.

Prometheus-compatible process metrics are available at `GET /metrics`.
Metrics include API request counts and durations, task/tool/LLM outcomes,
worker job outcomes, and span durations. Counters and histogram buckets are
process-local; they reset on restart and are not aggregated across replicas.
Protect the endpoint with the same API authentication policy or a trusted
gateway before exposing it outside a private network.

For a durable background job, `GET /research/jobs/{job_id}/diagnostics`
combines its current status, persisted worker attempts, and task checkpoint
errors. Attempt diagnostics include the error type and trace ID when a worker
caught an exception. The existing `/attempts` endpoint remains available for
the raw attempt history.


## Research evaluation benchmarks

Phase 21 adds versioned, labeled offline benchmark datasets for AURA's production research-memory retriever. The suite currently includes core platform queries and edge cases with overlapping topics and multiple relevant memories.

Run the complete benchmark suite without LLM or external API calls:

```powershell
uv run python -m evaluation.run_benchmarks
```

The runner prints machine-readable JSON containing the benchmark version, SHA-256 digest of each dataset, Python runtime, per-query rankings, aggregate metrics, and threshold results. To save the full result:

```powershell
uv run python -m evaluation.run_benchmarks --output artifacts/research-evaluation.json
```

The benchmark reports macro-averaged **Precision@k**, **Recall@k**, **Mean Reciprocal Rank (MRR)**, and **Normalized Discounted Cumulative Gain (nDCG@k)**. Each dataset declares minimum acceptance thresholds; the command exits non-zero when a threshold is missed. CI runs this suite on every pull request.

Benchmark labels are explicit relevance judgments for this fixed corpus, not proof that retrieved content is factually correct or that citations support a claim. The suite measures retrieval behavior only; broader research-quality dimensions remain available in each run's deterministic quality assessment. Add new versioned datasets rather than silently changing labels in an existing version, and review relevance judgments when benchmark versions change.

## Governed autonomous improvement

Phase 22 adds evidence-driven improvement cycles over completed research runs. Create a cycle with:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/research/runs/<RUN_ID>/improvement-cycles"
```

List cycles and proposals with `GET /research/runs/{run_id}/improvement-cycles` and `GET /research/improvement-cycles/{cycle_id}/proposals`. Proposals include a stable ID, evidence digest, source references, rationale, and a bounded recommended action.

Every proposal starts in `proposed` state. An authorized operator/admin must explicitly approve or reject it using the `/research/improvement-proposals/{proposal_id}/approve` or `/reject` endpoint. Rejections require a reason. An approved proposal can be marked validated only against a persisted experiment result belonging to the same research run; the validation record retains the result ID and pass/fail outcome. Cycles close when every proposal is rejected or validated.

The detector uses missing registered tools, failed execution records, and the deterministic research-quality dimensions. Analysis is deterministic for a given persisted evidence snapshot and creates an idempotent cycle keyed by its SHA-256 digest. The default quality floor is 0.80.

**Governance and safety boundary:** proposals are recommendations, not executable plans. Approval does not execute code or automatically change tools, prompts, dependencies, configuration, or deployments. Experiments remain constrained to application-registered trusted adapters and their existing input contracts. A human must implement and review any capability change separately, then evaluate it using recorded evidence. When API authentication is enabled, reviewer attribution is derived from the authenticated API key ID; clients cannot supply or spoof the reviewer field. In unauthenticated local-development mode, the audit identity is explicitly recorded as `unauthenticated`. Use a distinct API key for each operator when individual accountability is required, because a shared key identifies only the credential, not the human using it. Deployments should restrict write access to authenticated operator/admin identities.

## Production deployment (Docker Compose)

Phase 20 provides a production-oriented container image and a lightweight Compose deployment:

- **API** — FastAPI/Uvicorn, one process per container, authenticated API access, and an HTTP health check.
- **Worker** — separate durable-job worker.
- **Database** — the default deployment connects directly to your existing managed/online MongoDB cluster. MongoDB is not downloaded, run, or persisted on the laptop by default.

### Configure and start (recommended: managed MongoDB)

1. Copy the template and edit the local file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Set `AURA_DATABASE_URI` to your existing MongoDB connection URI. Keep the URI private and ensure its database user has only the permissions AURA needs. If `AURA_DATABASE_URI` is blank, the existing `MONGO_URL` setting is used for backward compatibility. Set `MONGODB_DATABASE` to the database name used by AURA. Also replace the example `AURA_API_KEYS` value with a generated key of at least 32 characters and configure your LLM provider key.

   Generate API key material locally:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Keep `.env` private. It is ignored by Git and must not be committed. Compose forces API authentication on. The configured API key must have the appropriate role (`reader`, `operator`, or `admin`).

3. Start only the API and worker:

   ```powershell
   docker compose up --build -d
   docker compose ps
   Invoke-RestMethod http://127.0.0.1:8000/health
   ```

4. Follow service logs:

   ```powershell
   docker compose logs -f api worker
   ```

The API and worker connect to the same online MongoDB cluster. The database itself stays in the managed service; only application containers, image layers, temporary files, and bounded operational logs occupy local disk. The API binds to `127.0.0.1:8000` by default. Put a TLS-terminating reverse proxy in front of it for remote access. Do not expose the database URI or publish a database port.

### Optional local MongoDB (offline development / isolated testing)

Local MongoDB is disabled by default and only starts when the `local-mongo` Compose profile is explicitly enabled. To keep the normal online URI untouched, create a separate environment file for local development (for example, copy `.env` to `.env.local`) and set `AURA_DATABASE_URI` there to:

```text
mongodb://aura_app:YOUR_URL_SAFE_APP_PASSWORD@mongodb:27017/aura?authSource=aura
```

Start the local database first and wait until it reports healthy:

```powershell
docker compose --env-file .env.local --profile local-mongo up -d mongodb
docker compose --env-file .env.local --profile local-mongo ps
```

Then start the application services against it:

```powershell
docker compose --env-file .env.local --profile local-mongo up --build -d api worker
```

The local database uses the named `aura-mongodb-data` volume and can consume disk as research data accumulates. Use it only when needed. Return to the managed database by starting the default stack with the regular `.env` file. Do not run `docker compose down -v` unless you intentionally want to permanently delete the local MongoDB volume and all data in it.

### Container releases

Pushing a version tag matching `v*` runs `.github/workflows/publish-container.yml` and publishes the image to GitHub Container Registry (GHCR), tagged with the version and (for non-prerelease tags) `latest`. For example:

```powershell
git tag v0.2.0
git push origin v0.2.0
```

To deploy a published version, set `AURA_IMAGE` in `.env` to the exact image tag, then run:

```powershell
docker compose pull
docker compose up -d --no-build
```

### Operations, upgrades, and rollback

- Check container readiness with `docker compose ps`; the API health check probes `/health`.
- Inspect structured service logs with `docker compose logs -f api worker`. Prometheus-compatible process metrics are exposed at `/metrics`; protect this endpoint at the same authenticated API boundary or a trusted gateway.
- Back up the managed MongoDB database using the provider's supported backup tooling and test restores. For local MongoDB, back up the named volume with MongoDB's supported tooling; a Docker volume is persistence, not a backup.
- Before an upgrade, take a database backup and record the current image tag. Deploy the new immutable image tag, then verify API health and worker startup.
- To roll back, set `AURA_IMAGE` to the previous known-good tag and run `docker compose pull` followed by `docker compose up -d --no-build`. Restore the database backup only if the newer application performed an incompatible data migration; the current application does not automatically roll back database contents.

### Deployment boundaries

The container runs as a non-root user and does not execute LLM-generated source code. API and worker use the same configured MongoDB database. Request-rate limiting and observability metrics are process-local, so multiple replicas do not share counters or rate-limit state. Use a shared gateway/limiter and an external metrics collector when scaling horizontally. Use a restricted host/container runtime, network egress controls, TLS, and managed secrets for deployments exposed to untrusted users.
