from datetime import datetime, timedelta, timezone
from threading import Lock

from memory.research_memory import ResearchMemoryItem
from storage.database.base import Database
from storage.models.experiment import ExperimentDefinition, ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.report import ResearchReport
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
    ToolImplementation,
)


class InMemoryDatabase(Database):
    def __init__(self):
        self.research_plans: dict[str, ResearchPlan] = {}
        self.research_tasks: dict[str, ResearchTask] = {}
        self.research_runs: dict[str, ResearchRun] = {}
        self.tools: dict[str, ToolDefinition] = {}
        self.tool_implementations: dict[str, ToolImplementation] = {}
        self.execution_results: dict[str, ExecutionResult] = {}
        self.experiments: dict[str, ExperimentDefinition] = {}
        self.experiment_results: dict[str, ExperimentResult] = {}
        self.research_memory: dict[str, ResearchMemoryItem] = {}
        self.research_reports: dict[str, ResearchReport] = {}
        self.research_jobs: dict[str, ResearchJob] = {}
        self._job_lock = Lock()

    def save_research_report(self, report: ResearchReport) -> None:
        self.research_reports[report.report_id] = report

    def get_research_report(self, report_id: str) -> ResearchReport | None:
        return self.research_reports.get(report_id)

    def save_research_plan(self, plan: ResearchPlan) -> None:
        self.research_plans[plan.question] = plan

    def get_research_plan(self, question: str) -> ResearchPlan | None:
        return self.research_plans.get(question)

    def list_execution_results_for_run(self, run_id: str) -> list[ExecutionResult]:
        return [item for item in self.execution_results.values() if item.run_id == run_id]

    def list_experiment_results_for_run(self, run_id: str) -> list[ExperimentResult]:
        return [item for item in self.experiment_results.values() if item.run_id == run_id]

    def save_research_job(self, job: ResearchJob) -> None:
        with self._job_lock:
            job.touch()
            self.research_jobs[job.job_id] = job

    def get_research_job(self, job_id: str) -> ResearchJob | None:
        with self._job_lock:
            return self.research_jobs.get(job_id)

    def get_research_job_by_idempotency_key(self, key: str) -> ResearchJob | None:
        with self._job_lock:
            return next((job for job in self.research_jobs.values() if job.idempotency_key == key), None)

    def claim_next_research_job(self, worker_id: str, lease_seconds: int = 3600) -> ResearchJob | None:
        now = datetime.now(timezone.utc)
        with self._job_lock:
            candidates = sorted(self.research_jobs.values(), key=lambda job: job.created_at)
            for job in candidates:
                expired = job.status == ResearchJobStatus.RUNNING and job.lease_expires_at is not None and job.lease_expires_at <= now
                if job.status != ResearchJobStatus.QUEUED and not expired:
                    continue
                if job.attempts >= job.max_attempts:
                    job.status = ResearchJobStatus.FAILED
                    job.error = job.error or "Maximum worker attempts exceeded."
                    job.finished_at = now
                    job.touch()
                    continue
                job.status = ResearchJobStatus.RUNNING
                job.attempts += 1
                job.worker_id = worker_id
                job.lease_expires_at = now + timedelta(seconds=lease_seconds)
                job.started_at = job.started_at or now
                job.error = None
                job.touch()
                return job
        return None

    def renew_research_job_lease(self, job_id: str, worker_id: str, lease_seconds: int = 3600) -> bool:
        with self._job_lock:
            job = self.research_jobs.get(job_id)
            if job is None or job.worker_id != worker_id or job.status not in {ResearchJobStatus.RUNNING, ResearchJobStatus.CANCEL_REQUESTED}:
                return False
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            job.touch()
            return True

    def save_research_task(self, task: ResearchTask) -> None:
        self.research_tasks[task.task_id] = task

    def save_research_run(self, run: ResearchRun) -> None:
        self.research_runs[run.run_id] = run

    def save_tool(self, tool: ToolDefinition) -> None:
        self.tools[tool.tool_id] = tool

    def save_tool_implementation(self, implementation: ToolImplementation) -> None:
        self.tool_implementations[implementation.implementation_id] = implementation

    def save_execution_result(self, result: ExecutionResult) -> None:
        self.execution_results[result.result_id] = result

    def save_experiment(self, experiment: ExperimentDefinition) -> None:
        self.experiments[experiment.experiment_id] = experiment

    def get_experiment(self, experiment_id: str) -> ExperimentDefinition | None:
        return self.experiments.get(experiment_id)

    def save_experiment_result(self, result: ExperimentResult) -> None:
        self.experiment_results[result.result_id] = result

    def get_experiment_result(self, result_id: str) -> ExperimentResult | None:
        return self.experiment_results.get(result_id)

    def list_experiments_for_run(self, run_id: str) -> list[ExperimentDefinition]:
        return [item for item in self.experiments.values() if item.run_id == run_id]

    def save_research_memory(self, item: ResearchMemoryItem) -> None:
        if item.memory_id in self.research_memory:
            raise ValueError(f"Research memory item '{item.memory_id}' already exists.")
        self.research_memory[item.memory_id] = item

    def update_research_memory(self, item: ResearchMemoryItem) -> None:
        if item.memory_id not in self.research_memory:
            raise KeyError(f"Research memory '{item.memory_id}' was not found.")
        self.research_memory[item.memory_id] = item

    def delete_research_memory(self, memory_id: str) -> bool:
        return self.research_memory.pop(memory_id, None) is not None

    def list_research_memory(self) -> list[ResearchMemoryItem]:
        return list(self.research_memory.values())

    def get_research_task(self, task_id: str) -> ResearchTask | None:
        return self.research_tasks.get(task_id)

    def get_research_run(self, run_id: str) -> ResearchRun | None:
        return self.research_runs.get(run_id)

    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        return self.tools.get(tool_id)

    def get_tool_implementation(self, implementation_id: str) -> ToolImplementation | None:
        return self.tool_implementations.get(implementation_id)

    def get_execution_result(self, result_id: str) -> ExecutionResult | None:
        return self.execution_results.get(result_id)

    def get_research_memory(self, memory_id: str) -> ResearchMemoryItem | None:
        return self.research_memory.get(memory_id)

    def list_research_memory_for_run(self, run_id: str) -> list[ResearchMemoryItem]:
        return [item for item in self.research_memory.values() if item.run_id == run_id]

    def search_research_memory(self, query: str) -> list[ResearchMemoryItem]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return []
        now = datetime.now(timezone.utc)
        results = []
        for item in self.research_memory.values():
            if item.archived or (item.expires_at is not None and item.expires_at <= now):
                continue
            searchable_text = (
                f"{item.research_question} {item.content} {' '.join(item.tags)}"
            ).lower()
            if any(term in searchable_text for term in query_terms):
                results.append(item)
        return results
