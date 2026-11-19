from datetime import datetime, timedelta, timezone
import uuid

from pymongo import MongoClient, ReturnDocument

from memory.research_memory import ResearchMemoryItem
from storage.database.base import Database
from storage.models.experiment import ExperimentDefinition, ExperimentResult
from storage.models.execution import ExecutionResult
from storage.models.report import ResearchReport
from storage.models.job import ResearchJob, ResearchJobStatus
from storage.models.job_attempt import ResearchJobAttempt
from storage.models.job_stats import ResearchJobQueueStats
from storage.models.research import (
    ResearchPlan,
    ResearchRun,
    ResearchTask,
    ToolDefinition,
    ToolImplementation,
)


class MongoDatabase(Database):
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string, tz_aware=True)
        self.db = self.client[database_name]
        self.research_plans = self.db["research_plans"]
        self.research_tasks = self.db["research_tasks"]
        self.research_runs = self.db["research_runs"]
        self.tools = self.db["tools"]
        self.tool_implementations = self.db["tool_implementations"]
        self.execution_results = self.db["execution_results"]
        self.experiments = self.db["experiments"]
        self.experiment_results = self.db["experiment_results"]
        self.research_memory = self.db["research_memory"]
        self.research_reports = self.db["research_reports"]
        self.research_jobs = self.db["research_jobs"]
        self.research_job_attempts = self.db["research_job_attempts"]
        self.research_job_attempts.create_index([("job_id", 1), ("attempt_number", 1)], unique=True)
        self.research_jobs.create_index(
            "idempotency_key",
            unique=True,
            partialFilterExpression={"idempotency_key": {"$type": "string"}},
        )
        self.research_jobs.create_index(
            "run_id",
            unique=True,
            partialFilterExpression={
                "status": {
                    "$in": [
                        ResearchJobStatus.QUEUED.value,
                        ResearchJobStatus.RUNNING.value,
                        ResearchJobStatus.CANCEL_REQUESTED.value,
                    ]
                }
            },
            name="one_active_research_job_per_run",
        )
        self.research_jobs.create_index([("status", 1), ("created_at", 1)])

    def save_research_report(self, report: ResearchReport) -> None:
        self.research_reports.replace_one(
            {"report_id": report.report_id}, report.model_dump(mode="json"), upsert=True
        )

    def get_research_report(self, report_id: str) -> ResearchReport | None:
        document = self.research_reports.find_one({"report_id": report_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchReport.model_validate(document)

    def save_research_plan(self, plan: ResearchPlan) -> None:
        self.research_plans.replace_one({"question": plan.question}, plan.model_dump(), upsert=True)

    def get_research_plan(self, question: str) -> ResearchPlan | None:
        document = self.research_plans.find_one({"question": question})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchPlan.model_validate(document)

    def list_execution_results_for_run(self, run_id: str) -> list[ExecutionResult]:
        results = []
        for document in self.execution_results.find({"run_id": run_id}):
            document.pop("_id", None)
            results.append(ExecutionResult.model_validate(document))
        return results

    def list_experiment_results_for_run(self, run_id: str) -> list[ExperimentResult]:
        results = []
        for document in self.experiment_results.find({"run_id": run_id}):
            document.pop("_id", None)
            results.append(ExperimentResult.model_validate(document))
        return results

    def save_research_job(self, job: ResearchJob) -> None:
        job.touch()
        self.research_jobs.replace_one(
            {"job_id": job.job_id},
            job.model_dump(),
            upsert=True,
        )

    def get_research_job(self, job_id: str) -> ResearchJob | None:
        document = self.research_jobs.find_one({"job_id": job_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchJob.model_validate(document)

    def get_research_job_by_idempotency_key(self, key: str) -> ResearchJob | None:
        document = self.research_jobs.find_one({"idempotency_key": key})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchJob.model_validate(document)

    def list_research_jobs_for_run(self, run_id: str) -> list[ResearchJob]:
        jobs = []
        for document in self.research_jobs.find({"run_id": run_id}).sort("created_at", -1):
            document.pop("_id", None)
            jobs.append(ResearchJob.model_validate(document))
        return jobs

    def claim_next_research_job(self, worker_id: str, lease_seconds: int = 3600) -> ResearchJob | None:
        now = datetime.now(timezone.utc)
        lease_expiry = now + timedelta(seconds=lease_seconds)
        self.research_jobs.update_many(
            {
                "status": ResearchJobStatus.CANCEL_REQUESTED.value,
                "lease_expires_at": {"$lte": now},
            },
            {
                "$set": {
                    "status": ResearchJobStatus.CANCELLED.value,
                    "finished_at": now,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "lease_token": None,
                    "updated_at": now,
                }
            },
        )
        self.research_jobs.update_many(
            {
                "status": ResearchJobStatus.RUNNING.value,
                "lease_expires_at": {"$lte": now},
                "$expr": {"$gte": ["$attempts", "$max_attempts"]},
            },
            {
                "$set": {
                    "status": ResearchJobStatus.FAILED.value,
                    "error": "Maximum worker attempts exceeded.",
                    "finished_at": now,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "lease_token": None,
                    "updated_at": now,
                }
            },
        )
        document = self.research_jobs.find_one_and_update(
            {
                "$expr": {"$lt": ["$attempts", "$max_attempts"]},
                "$or": [
                    {"status": ResearchJobStatus.QUEUED.value},
                    {
                        "status": ResearchJobStatus.RUNNING.value,
                        "lease_expires_at": {"$lte": now},
                    },
                ],
            },
            {
                "$set": {
                    "status": ResearchJobStatus.RUNNING.value,
                    "worker_id": worker_id,
                    "lease_expires_at": lease_expiry,
                    "lease_token": uuid.uuid4().hex,
                    "started_at": now,
                    "error": None,
                    "updated_at": now,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchJob.model_validate(document)

    def renew_research_job_lease(
        self, job_id: str, worker_id: str, lease_token: str, lease_seconds: int = 3600
    ) -> bool:
        now = datetime.now(timezone.utc)
        result = self.research_jobs.update_one(
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "lease_token": lease_token,
                "status": {"$in": [ResearchJobStatus.RUNNING.value, ResearchJobStatus.CANCEL_REQUESTED.value]},
                "lease_expires_at": {"$gt": now},
            },
            {
                "$set": {
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            },
        )
        return result.modified_count == 1

    def is_research_job_lease_owner(self, job_id: str, worker_id: str, lease_token: str) -> bool:
        return self.research_jobs.find_one(
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "lease_token": lease_token,
                "status": {"$in": [ResearchJobStatus.RUNNING.value, ResearchJobStatus.CANCEL_REQUESTED.value]},
                "lease_expires_at": {"$gt": datetime.now(timezone.utc)},
            },
            {"_id": 1},
        ) is not None

    def save_research_job_if_lease_owner(
        self, job: ResearchJob, worker_id: str, lease_token: str
    ) -> bool:
        now = datetime.now(timezone.utc)
        if job.status in {ResearchJobStatus.RUNNING, ResearchJobStatus.CANCEL_REQUESTED}:
            if job.worker_id != worker_id or job.lease_token != lease_token:
                return False
        job.touch()
        update_fields = job.model_dump()
        active_statuses = {ResearchJobStatus.RUNNING, ResearchJobStatus.CANCEL_REQUESTED}
        if job.status in active_statuses:
            # Keep the database's latest heartbeat values rather than replacing
            # them with an older lease timestamp from this worker's snapshot.
            update_fields.pop("worker_id", None)
            update_fields.pop("lease_token", None)
            update_fields.pop("lease_expires_at", None)
            # A progress snapshot read before an API cancellation must not
            # change CANCEL_REQUESTED back to RUNNING.
            if job.status == ResearchJobStatus.RUNNING:
                update_fields["status"] = {
                    "$cond": [
                        {"$eq": ["$status", ResearchJobStatus.CANCEL_REQUESTED.value]},
                        ResearchJobStatus.CANCEL_REQUESTED.value,
                        ResearchJobStatus.RUNNING.value,
                    ]
                }
            pipeline_fields = {
                key: {"$literal": value}
                for key, value in update_fields.items()
                if key != "status"
            }
            if "status" in update_fields:
                pipeline_fields["status"] = update_fields["status"]
            update = [{"$set": pipeline_fields}]
        else:
            update = {"$set": update_fields}
        result = self.research_jobs.update_one(
            {
                "job_id": job.job_id,
                "worker_id": worker_id,
                "lease_token": lease_token,
                "status": {"$in": [ResearchJobStatus.RUNNING.value, ResearchJobStatus.CANCEL_REQUESTED.value]},
                "lease_expires_at": {"$gt": now},
            },
            update,
        )
        return result.matched_count == 1

    def get_research_job_queue_stats(self) -> ResearchJobQueueStats:
        now = datetime.now(timezone.utc)
        grouped = list(self.research_jobs.aggregate([
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "oldest_queued_at": {
                        "$min": {
                            "$cond": [
                                {"$eq": ["$status", ResearchJobStatus.QUEUED.value]},
                                "$created_at",
                                None,
                            ]
                        }
                    },
                    "workers": {
                        "$addToSet": {
                            "$cond": [
                                {"$in": ["$status", [
                                    ResearchJobStatus.RUNNING.value,
                                    ResearchJobStatus.CANCEL_REQUESTED.value,
                                ]]},
                                "$worker_id",
                                None,
                            ]
                        }
                    },
                }
            }
        ]))
        counts = {item["_id"]: item["count"] for item in grouped}
        workers = {
            worker
            for item in grouped
            for worker in item.get("workers", [])
            if worker
        }
        oldest = [
            item["oldest_queued_at"]
            for item in grouped
            if item.get("oldest_queued_at") is not None
        ]
        expired = self.research_jobs.count_documents({
            "status": {"$in": [
                ResearchJobStatus.RUNNING.value,
                ResearchJobStatus.CANCEL_REQUESTED.value,
            ]},
            "lease_expires_at": {"$lte": now},
        })
        return ResearchJobQueueStats(
            total=sum(counts.values()),
            queued=counts.get(ResearchJobStatus.QUEUED.value, 0),
            running=counts.get(ResearchJobStatus.RUNNING.value, 0),
            cancel_requested=counts.get(ResearchJobStatus.CANCEL_REQUESTED.value, 0),
            succeeded=counts.get(ResearchJobStatus.SUCCEEDED.value, 0),
            failed=counts.get(ResearchJobStatus.FAILED.value, 0),
            cancelled=counts.get(ResearchJobStatus.CANCELLED.value, 0),
            expired_leases=expired,
            active_workers=len(workers),
            oldest_queued_at=min(oldest) if oldest else None,
        )

    def save_research_job_attempt(self, attempt: ResearchJobAttempt) -> None:
        self.research_job_attempts.replace_one(
            {"attempt_id": attempt.attempt_id}, attempt.model_dump(mode="json"), upsert=True
        )

    def list_research_job_attempts(self, job_id: str) -> list[ResearchJobAttempt]:
        attempts = []
        for document in self.research_job_attempts.find({"job_id": job_id}).sort("attempt_number", 1):
            document.pop("_id", None)
            attempts.append(ResearchJobAttempt.model_validate(document))
        return attempts

    def save_research_task(self, task: ResearchTask) -> None:
        self.research_tasks.replace_one({"task_id": task.task_id}, task.model_dump(), upsert=True)

    def save_research_run(self, run: ResearchRun) -> None:
        self.research_runs.replace_one({"run_id": run.run_id}, run.model_dump(), upsert=True)

    def save_tool(self, tool: ToolDefinition) -> None:
        self.tools.replace_one({"tool_id": tool.tool_id}, tool.model_dump(), upsert=True)

    def save_tool_implementation(self, implementation: ToolImplementation) -> None:
        self.tool_implementations.replace_one(
            {"implementation_id": implementation.implementation_id},
            implementation.model_dump(),
            upsert=True,
        )

    def save_execution_result(self, result: ExecutionResult) -> None:
        self.execution_results.replace_one(
            {"result_id": result.result_id}, result.model_dump(), upsert=True
        )

    def save_experiment(self, experiment: ExperimentDefinition) -> None:
        self.experiments.replace_one(
            {"experiment_id": experiment.experiment_id},
            experiment.model_dump(mode="json"),
            upsert=True,
        )

    def get_experiment(self, experiment_id: str) -> ExperimentDefinition | None:
        document = self.experiments.find_one({"experiment_id": experiment_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ExperimentDefinition.model_validate(document)

    def save_experiment_result(self, result: ExperimentResult) -> None:
        self.experiment_results.replace_one(
            {"result_id": result.result_id},
            result.model_dump(mode="json"),
            upsert=True,
        )

    def get_experiment_result(self, result_id: str) -> ExperimentResult | None:
        document = self.experiment_results.find_one({"result_id": result_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ExperimentResult.model_validate(document)

    def list_experiments_for_run(self, run_id: str) -> list[ExperimentDefinition]:
        experiments = []
        for document in self.experiments.find({"run_id": run_id}):
            document.pop("_id", None)
            experiments.append(ExperimentDefinition.model_validate(document))
        return experiments

    def save_research_memory(self, item: ResearchMemoryItem) -> None:
        self.research_memory.replace_one(
            {"memory_id": item.memory_id}, item.model_dump(), upsert=True
        )

    def update_research_memory(self, item: ResearchMemoryItem) -> None:
        result = self.research_memory.replace_one(
            {"memory_id": item.memory_id}, item.model_dump(), upsert=False
        )
        if result.matched_count == 0:
            raise KeyError(f"Research memory '{item.memory_id}' was not found.")

    def delete_research_memory(self, memory_id: str) -> bool:
        result = self.research_memory.delete_one({"memory_id": memory_id})
        return result.deleted_count > 0

    @staticmethod
    def _memory_from_document(document) -> ResearchMemoryItem:
        document.pop("_id", None)
        return ResearchMemoryItem.model_validate(document)

    def list_research_memory(self) -> list[ResearchMemoryItem]:
        return [
            self._memory_from_document(document)
            for document in self.research_memory.find()
        ]

    def get_research_task(self, task_id: str) -> ResearchTask | None:
        document = self.research_tasks.find_one({"task_id": task_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchTask(**document)

    def get_research_run(self, run_id: str) -> ResearchRun | None:
        document = self.research_runs.find_one({"run_id": run_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ResearchRun(**document)

    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        document = self.tools.find_one({"tool_id": tool_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ToolDefinition(**document)

    def get_tool_implementation(self, implementation_id: str) -> ToolImplementation | None:
        document = self.tool_implementations.find_one({"implementation_id": implementation_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ToolImplementation(**document)

    def get_execution_result(self, result_id: str) -> ExecutionResult | None:
        document = self.execution_results.find_one({"result_id": result_id})
        if document is None:
            return None
        document.pop("_id", None)
        return ExecutionResult(**document)

    def get_research_memory(self, memory_id: str) -> ResearchMemoryItem | None:
        document = self.research_memory.find_one({"memory_id": memory_id})
        if document is None:
            return None
        return self._memory_from_document(document)

    def list_research_memory_for_run(self, run_id: str) -> list[ResearchMemoryItem]:
        return [
            self._memory_from_document(document)
            for document in self.research_memory.find({"run_id": run_id})
        ]

    def search_research_memory(self, query: str) -> list[ResearchMemoryItem]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return []
        now = datetime.now(timezone.utc)
        results = []
        for document in self.research_memory.find():
            item = self._memory_from_document(document)
            if item.archived or (item.expires_at is not None and item.expires_at <= now):
                continue
            searchable_text = (
                f"{item.research_question} {item.content} {' '.join(item.tags)}"
            ).lower()
            if any(term in searchable_text for term in query_terms):
                results.append(item)
        return results

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False
