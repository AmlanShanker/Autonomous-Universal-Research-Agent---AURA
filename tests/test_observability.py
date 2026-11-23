import json
import logging

from fastapi.testclient import TestClient

from observability import (
    JSONFormatter,
    MetricsRegistry,
    bind_context,
    trace_id_from_header,
    current_span_id,
    span,
)
from api.app import create_app
from storage.database.memory import InMemoryDatabase
from storage.models.job import ResearchJob
from storage.models.job_attempt import ResearchJobAttempt, ResearchJobAttemptStatus


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("aura.test", logging.INFO, __file__, 1, message, (), None)


def test_json_formatter_includes_correlation_context_and_redacts_secrets():
    formatter = JSONFormatter()
    with bind_context(trace_id="a" * 32, request_id="request-1", job_id="job-1"):
        rendered = formatter.format(
            _record("Authorization: Bearer abcdef123456 api_key=private MongoDB mongodb://user:pass@db.example/aura")
        )

    payload = json.loads(rendered)
    assert payload["trace_id"] == "a" * 32
    assert payload["request_id"] == "request-1"
    assert payload["job_id"] == "job-1"
    assert "abcdef123456" not in rendered
    assert "private" not in rendered
    assert "user:pass" not in rendered
    assert "[REDACTED]" in rendered


def test_traceparent_accepts_valid_w3c_trace_id_and_rejects_invalid_values():
    trace_id = "0123456789abcdef0123456789abcdef"
    assert trace_id_from_header(f"00-{trace_id}-0123456789abcdef-01") == trace_id
    assert len(trace_id_from_header("not-a-traceparent")) == 32
    assert trace_id_from_header(f"00-{'0' * 32}-0123456789abcdef-01") != "0" * 32


def test_metrics_registry_renders_bounded_prometheus_histogram():
    metrics = MetricsRegistry()
    metrics.increment("aura_test_events", outcome="ok")
    metrics.observe("aura_test_duration_ms", 7, component="test")
    metrics.observe("aura_test_duration_ms", 120, component="test")

    rendered = metrics.render()
    assert 'aura_test_events_total{outcome="ok"} 1' in rendered
    assert 'aura_test_duration_ms_bucket{component="test",le="10"} 1' in rendered
    assert 'aura_test_duration_ms_bucket{component="test",le="+Inf"} 2' in rendered
    assert 'aura_test_duration_ms_count{component="test"} 2' in rendered


def test_api_propagates_trace_headers_and_exposes_metrics(monkeypatch):
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "false")
    client = TestClient(create_app(database=InMemoryDatabase(), experiment_adapters={}))

    trace_id = "0123456789abcdef0123456789abcdef"
    response = client.get(
        "/health",
        headers={
            "traceparent": f"00-{trace_id}-0123456789abcdef-01",
            "X-Request-ID": "request-test-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-test-1"
    assert response.headers["traceparent"].startswith(f"00-{trace_id}-")

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "aura_http_requests_total" in metrics.text
    assert "aura_http_request_duration_ms_bucket" in metrics.text


def test_job_diagnostics_returns_404_for_unknown_job(monkeypatch):
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "false")
    client = TestClient(create_app(database=InMemoryDatabase(), experiment_adapters={}))

    response = client.get("/research/jobs/missing/diagnostics")

    assert response.status_code == 404

 
def test_job_diagnostics_includes_durable_attempt_failure_context(monkeypatch):
    monkeypatch.setenv("AURA_API_AUTH_ENABLED", "false")
    database = InMemoryDatabase()
    database.save_research_job(ResearchJob(job_id="job-1", run_id="run-1"))
    attempt = ResearchJobAttempt(
        attempt_id="job-1:1",
        job_id="job-1",
        attempt_number=1,
        worker_id="worker-1",
    )
    attempt.finish(
        ResearchJobAttemptStatus.FAILED,
        "adapter failed",
        error_type="RuntimeError",
        trace_id="a" * 32,
    )
    database.save_research_job_attempt(attempt)
    client = TestClient(create_app(database=database, experiment_adapters={}))

    response = client.get("/research/jobs/job-1/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-1"
    assert payload["attempts"][0]["error_type"] == "RuntimeError"
    assert payload["attempts"][0]["trace_id"] == "a" * 32

 
def test_nested_spans_keep_parent_relationships():
    formatter = JSONFormatter()
    with span("outer", component="test"):
        parent_id = current_span_id()
        with span("inner", component="test"):
            child_id = current_span_id()
            payload = json.loads(formatter.format(_record("nested event")))

    assert parent_id is not None
    assert child_id is not None
    assert child_id != parent_id
    assert payload["span_id"] == child_id
    assert payload["parent_span_id"] == parent_id
