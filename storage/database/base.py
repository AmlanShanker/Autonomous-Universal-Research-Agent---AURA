from abc import ABC, abstractmethod

from memory.research_memory import ResearchMemoryItem
from storage.models.experiment import (\n    ExperimentArtifact, ExperimentDefinition, ExperimentParameterSnapshot, ExperimentResult,\n)
from storage.models.context import ResearchContext
from storage.models.execution import ExecutionResult
from storage.models.evidence import ResearchSource
from storage.models.report import ResearchReport
from storage.models.job import ResearchJob
from storage.models.job_attempt import ResearchJobAttempt
from storage.models.job_stats import ResearchJobQueueStats
from storage.models.task_checkpoint import TaskCheckpoint
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
    ToolImplementation,
)


class Database(ABC):

    @abstractmethod
    def save_research_source(self, source: ResearchSource) -> None:
        pass

    @abstractmethod
    def get_research_source(self, source_id: str) -> ResearchSource | None:
        pass

    @abstractmethod
    def get_research_source_by_canonical_id(self, canonical_id: str) -> ResearchSource | None:
        pass

    @abstractmethod
    def list_research_sources(self) -> list[ResearchSource]:
        pass

    @abstractmethod
    @abstractmethod
    def save_research_source(self, source: ResearchSource) -> None:
        pass

    @abstractmethod
    def get_research_source(self, source_id: str) -> ResearchSource | None:
        pass

    @abstractmethod
    def get_research_source_by_canonical_id(self, canonical_id: str) -> ResearchSource | None:
        pass

    @abstractmethod
    def list_research_sources(self) -> list[ResearchSource]:
        pass

    def save_research_report(self, report: ResearchReport) -> None:
        pass

    @abstractmethod
    def get_research_report(self, report_id: str) -> ResearchReport | None:
        pass

    @abstractmethod
    def save_research_plan(self, plan: ResearchPlan) -> None:
        pass

    @abstractmethod
    def get_research_plan(self, question: str) -> ResearchPlan | None:
        pass

    @abstractmethod
    def list_execution_results_for_run(self, run_id: str) -> list[ExecutionResult]:
        pass

    @abstractmethod
    def list_experiment_results_for_run(self, run_id: str) -> list[ExperimentResult]:
        pass

    @abstractmethod
    def save_research_job(self, job: ResearchJob) -> None:
        pass

    @abstractmethod
    def get_research_job(self, job_id: str) -> ResearchJob | None:
        pass

    @abstractmethod
    def get_research_job_by_idempotency_key(self, key: str) -> ResearchJob | None:
        pass

    @abstractmethod
    def list_research_jobs_for_run(self, run_id: str) -> list[ResearchJob]:
        pass

    @abstractmethod
    def claim_next_research_job(self, worker_id: str, lease_seconds: int = 3600) -> ResearchJob | None:
        pass

    @abstractmethod
    def renew_research_job_lease(self, job_id: str, worker_id: str, lease_token: str, lease_seconds: int = 3600) -> bool:
        pass

    @abstractmethod
    def is_research_job_lease_owner(self, job_id: str, worker_id: str, lease_token: str) -> bool:
        pass

    @abstractmethod
    def save_research_job_if_lease_owner(self, job: ResearchJob, worker_id: str, lease_token: str) -> bool:
        pass

    @abstractmethod
    def get_research_job_queue_stats(self) -> ResearchJobQueueStats:
        pass

    @abstractmethod
    def save_research_job_attempt(self, attempt: ResearchJobAttempt) -> None:
        pass

    @abstractmethod
    def list_research_job_attempts(self, job_id: str) -> list[ResearchJobAttempt]:
        pass

    @abstractmethod
    def save_task_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        pass

    @abstractmethod
    def get_task_checkpoint(self, run_id: str, task_id: str) -> TaskCheckpoint | None:
        pass

    @abstractmethod
    def save_research_task(self, task: ResearchTask) -> None:
        pass

    @abstractmethod
    def save_research_run(self, run: ResearchRun) -> None:
        pass

    @abstractmethod
    def save_tool(self, tool: ToolDefinition) -> None:
        pass

    @abstractmethod
    def save_tool_implementation(self, implementation: ToolImplementation) -> None:
        pass

    @abstractmethod
    def save_execution_result(self, result: ExecutionResult) -> None:
        pass

    @abstractmethod
    def save_experiment(self, experiment: ExperimentDefinition) -> None:
        pass

    @abstractmethod
    def save_experiment_parameter_snapshot(self, snapshot: ExperimentParameterSnapshot) -> None:
        pass

    @abstractmethod
    def get_experiment_parameter_snapshot(self, snapshot_id: str) -> ExperimentParameterSnapshot | None:
        pass

    @abstractmethod
    def save_experiment_artifact(self, artifact: ExperimentArtifact) -> None:
        pass

    @abstractmethod
    def get_experiment_artifact(self, artifact_id: str) -> ExperimentArtifact | None:
        pass

    @abstractmethod
    def list_experiment_artifacts(self, result_id: str) -> list[ExperimentArtifact]:
        pass

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> ExperimentDefinition | None:
        pass

    @abstractmethod
    def save_experiment_result(self, result: ExperimentResult) -> None:
        pass

    @abstractmethod
    def get_experiment_result(self, result_id: str) -> ExperimentResult | None:
        pass

    @abstractmethod
    def list_experiments_for_run(self, run_id: str) -> list[ExperimentDefinition]:
        pass

    @abstractmethod
    def save_research_memory(self, item: ResearchMemoryItem) -> None:
        pass

    @abstractmethod
    def update_research_memory(self, item: ResearchMemoryItem) -> None:
        pass

    @abstractmethod
    def delete_research_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def list_research_memory(self) -> list[ResearchMemoryItem]:
        pass

    @abstractmethod
    def get_research_task(self, task_id: str) -> ResearchTask | None:
        pass

    @abstractmethod
    def get_research_run(self, run_id: str) -> ResearchRun | None:
        pass

    @abstractmethod
    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        pass

    @abstractmethod
    def get_tool_implementation(
        self, implementation_id: str
    ) -> ToolImplementation | None:
        pass

    @abstractmethod
    def get_execution_result(self, result_id: str) -> ExecutionResult | None:
        pass

    @abstractmethod
    def get_research_memory(self, memory_id: str) -> ResearchMemoryItem | None:
        pass

    @abstractmethod
    def list_research_memory_for_run(self, run_id: str) -> list[ResearchMemoryItem]:
        pass

    @abstractmethod
    def search_research_memory(self, query: str) -> list[ResearchMemoryItem]:
        pass
