"""
Job storage backed by Redis.

We store each job as a JSON blob under `job:{job_id}`. This keeps the
worker and API process in sync without needing a separate database.
"""

import json
from typing import Optional

import redis

from app.models import JobResponse, JobStatus

JOB_KEY_PREFIX = "job:"
JOB_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def save_job(r: redis.Redis, job: JobResponse) -> None:
    key = f"{JOB_KEY_PREFIX}{job.job_id}"
    r.set(key, job.model_dump_json(), ex=JOB_TTL_SECONDS)


def load_job(r: redis.Redis, job_id: str) -> Optional[JobResponse]:
    key = f"{JOB_KEY_PREFIX}{job_id}"
    raw = r.get(key)
    if raw is None:
        return None
    return JobResponse(**json.loads(raw))


def update_job_status(r: redis.Redis, job_id: str, status: JobStatus, error: str | None = None) -> None:
    """
    Update the status (and optionally error) of an existing job.

    Raises KeyError if the job doesn't exist in Redis (e.g. it expired,
    or was never created). This used to fail silently, which meant a
    FAILED status from the processor could be dropped entirely, leaving
    a client polling a job that would never change from "queued".
    """
    job = load_job(r, job_id)
    if job is None:
        raise KeyError(f"Cannot update status for unknown job: {job_id}")

    job.status = status
    if error is not None:
        job.error = error
    save_job(r, job)
