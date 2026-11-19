import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from tools.base import ResearchTool


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Resource limits for application-owned, trusted tool adapters."""

    allowed_tools: frozenset[str] = frozenset({
        "literature_search",
        "dataset_download",
    })
    max_input_bytes: int = 64 * 1024
    max_output_bytes: int = 4 * 1024 * 1024
    max_concurrent_executions: int = 4
    admission_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("max_input_bytes", "max_output_bytes", "max_concurrent_executions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.admission_timeout_seconds <= 0:
            raise ValueError("admission_timeout_seconds must be positive.")


class ToolExecutionSandbox:
    _default_slots = threading.BoundedSemaphore(4)

    """
    Enforce a bounded execution boundary for trusted built-in adapters.

    This is an application-level allowlist and resource guard, not an OS
    security boundary. It does not execute generated source code. Network
    adapters must additionally enforce their own connect/read timeouts and
    transfer limits.
    """

    def __init__(self, policy: ToolExecutionPolicy | None = None):
        self.policy = policy or ToolExecutionPolicy()
        self._slots = (
            self._default_slots
            if policy is None
            else threading.BoundedSemaphore(self.policy.max_concurrent_executions)
        )

    @staticmethod
    def _encoded_size(value: Any) -> int:
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool inputs and outputs must be finite JSON data.") from exc
        return len(encoded)

    def execute(self, tool: ResearchTool, **inputs: Any) -> tuple[dict, dict]:
        if tool.name not in self.policy.allowed_tools:
            raise PermissionError(
                f"Tool '{tool.name}' is not permitted by the execution policy."
            )

        input_size = self._encoded_size(inputs)
        if input_size > self.policy.max_input_bytes:
            raise ValueError(
                f"Tool inputs exceed the {self.policy.max_input_bytes}-byte limit."
            )

        if not self._slots.acquire(timeout=self.policy.admission_timeout_seconds):
            raise TimeoutError("Tool execution capacity is exhausted.")

        started = time.monotonic()
        try:
            output = tool.execute(**inputs)
            if not isinstance(output, dict):
                raise ValueError("Tool output must be a JSON object.")
            output_size = self._encoded_size(output)
            if output_size > self.policy.max_output_bytes:
                raise ValueError(
                    f"Tool output exceeds the {self.policy.max_output_bytes}-byte limit."
                )
            return output, {
                "sandbox_policy": "trusted_builtin_v1",
                "input_bytes": input_size,
                "output_bytes": output_size,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            }
        finally:
            self._slots.release()
