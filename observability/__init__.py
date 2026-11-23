"""Lightweight, dependency-free observability primitives for AURA.

The module provides JSON logs, request/job/task correlation, local spans, and
Prometheus text exposition. It deliberately avoids logging prompts, payloads,
credentials, or arbitrary object representations.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from logging import LogRecord
from threading import Lock
from typing import Iterator


_CONTEXT_FIELDS = ("trace_id", "request_id", "run_id", "job_id", "task_id", "worker_id")
_context = {
    name: contextvars.ContextVar(f"aura_{name}", default=None)
    for name in _CONTEXT_FIELDS
}
_span_id = contextvars.ContextVar("aura_span_id", default=None)
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(mongodb(?:\+srv)?://)[^\s/@]+@"), r"\1[REDACTED]@"),
)


def _redact(value):
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        for pattern, replacement in _SECRET_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    return value


class JSONFormatter(logging.Formatter):
    """Emit one structured JSON object per log record."""

    def format(self, record: LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name, variable in _context.items():
            value = variable.get()
            if value:
                payload[name] = value
        span_id = _span_id.get()
        if span_id:
            payload["span_id"] = span_id
        for name in ("event", "duration_ms", "status_code", "method", "route", "component", "error_type", "provider", "operation"):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), ensure_ascii=False, separators=(",", ":"))


def configure_logging() -> None:
    """Configure a single JSON stream handler; safe to call repeatedly."""
    root = logging.getLogger()
    level_name = os.getenv("AURA_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError("AURA_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    root.setLevel(level)
    if not any(getattr(handler, "_aura_json_handler", False) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        handler._aura_json_handler = True
        root.addHandler(handler)


@contextmanager
def bind_context(**values) -> Iterator[None]:
    """Temporarily bind non-empty correlation identifiers to the current context."""
    tokens = []
    for name, value in values.items():
        if name not in _context or value is None:
            continue
        tokens.append((_context[name], _context[name].set(str(value))))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def current_context() -> dict[str, str]:
    return {name: value for name, variable in _context.items() if (value := variable.get())}


def new_trace_id() -> str:
    return uuid.uuid4().hex


def trace_id_from_header(value: str | None) -> str:
    """Accept a valid W3C traceparent trace-id; otherwise create a fresh ID."""
    if value:
        match = re.fullmatch(r"\s*\d{2}-([0-9a-fA-F]{32})-[0-9a-fA-F]{16}-[0-9a-fA-F]{2}\s*", value)
        if match and match.group(1) != "0" * 32:
            return match.group(1).lower()
    return new_trace_id()


class MetricsRegistry:
    """Thread-safe counters and duration summaries with Prometheus exposition."""

    def __init__(self):
        self._lock = Lock()
        self._counters = defaultdict(float)
        self._observations = defaultdict(lambda: {"count": 0, "sum": 0.0, "buckets": [0] * 10})
        self._gauges = {}

    @staticmethod
    def _key(name: str, labels: dict | None = None):
        normalized = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        return name, normalized

    def increment(self, name: str, amount: float = 1, **labels) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += amount

    def observe(self, name: str, value: float, **labels) -> None:
        bounds = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
        with self._lock:
            observation = self._observations[self._key(name, labels)]
            observation["count"] += 1
            observation["sum"] += float(value)
            for index, bound in enumerate(bounds):
                if value <= bound:
                    observation["buckets"][index] += 1
                    break

    def gauge(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = float(value)

    @staticmethod
    def _labels(labels: tuple) -> str:
        if not labels:
            return ""
        encoded = ",".join(
            f'{key}="{str(value).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for key, value in labels
        )
        return "{" + encoded + "}"

    def render(self) -> str:
        lines = []
        with self._lock:
            counters = list(self._counters.items())
            observations = [(key, dict(value, buckets=list(value["buckets"]))) for key, value in self._observations.items()]
            gauges = list(self._gauges.items())
        for (name, labels), value in sorted(counters):
            lines.append(f"{name}_total{self._labels(labels)} {value:g}")
        bounds = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
        for (name, labels), observation in sorted(observations):
            cumulative = 0
            for index, bound in enumerate(bounds):
                cumulative += observation["buckets"][index]
                lines.append(f'{name}_bucket{self._labels(labels + (("le", str(bound)),))} {cumulative}')
            lines.append(f'{name}_bucket{self._labels(labels + (("le", "+Inf"),))} {observation["count"]}')
            lines.append(f"{name}_count{self._labels(labels)} {observation['count']}")
            lines.append(f"{name}_sum{self._labels(labels)} {observation['sum']:g}")
        for (name, labels), value in sorted(gauges):
            lines.append(f"{name}{self._labels(labels)} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")


METRICS = MetricsRegistry()


@contextmanager
def span(name: str, **attributes) -> Iterator[None]:
    """Record a local correlated span and duration without exporting payload data."""
    logger = logging.getLogger("aura.trace")
    current_span = uuid.uuid4().hex[:16]
    token = _span_id.set(current_span)
    started = time.perf_counter()
    safe_attributes = {key: value for key, value in attributes.items() if key in {
        "component", "operation", "provider", "status"
    }}
    logger.info("span started", extra={"event": "span.started", **safe_attributes})
    try:
        yield
    except Exception as exc:
        METRICS.increment("aura_spans", outcome="error", span_name=name)
        logger.exception(
            "span failed",
            extra={"event": "span.failed", "component": name, "error_type": type(exc).__name__},
        )
        raise
    else:
        METRICS.increment("aura_spans", outcome="success", span_name=name)
    finally:
        duration = (time.perf_counter() - started) * 1000
        METRICS.observe("aura_span_duration_ms", duration, component=name)
        logger.info(
            "span finished",
            extra={"event": "span.finished", "component": name, "duration_ms": round(duration, 3)},
        )
        _span_id.reset(token)
