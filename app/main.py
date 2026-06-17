"""
FastAPI application for Klipp.

Endpoints:
    POST /jobs          - submit a video URL for clipping
    GET  /jobs/{job_id} - check job status / get results
    GET  /clips/{name}  - download a produced clip file
    GET  /skill         - Pharos agent-skill discovery schema (public, no auth)
    GET  /healthz       - health check
"""

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse
from redis import RedisError

from app import config
from app.models import ClipRequest, JobResponse, JobStatus
from app.processor import CLIPS_DIR, process_job
from app.queue import get_queue, get_redis_connection
from app.storage import load_job, save_job

app = FastAPI(title="Klipp", version="1.0.0")

# ---------------------------------------------------------------------------
# Optional API key auth. If KLIPP_API_KEY is set in the environment, all
# /jobs and /clips endpoints require the caller to pass it in the
# X-API-Key header. Set it to a random secret in production.
# ---------------------------------------------------------------------------
_API_KEY = config.API_KEY or ""
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_SKILL_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "PHAROS_SKILL_SCHEMA.json"


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Dependency that enforces the API key when KLIPP_API_KEY is configured."""
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _base_url(request_headers: dict | None = None) -> str:
    """
    Build the base URL used to construct clip download URLs.
    Reads KLIPP_BASE_URL from the environment (set this to your deployment's public URL).
    Falls back to an empty string (produces relative URLs like /clips/...).
    """
    return os.environ.get("KLIPP_BASE_URL", "").rstrip("/")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/skill")
def get_skill_schema():
    """
    Agent-skill discovery endpoint for Pharos (and any other agent
    framework). Returns the schema describing how to call this service -
    no API key required, since an agent needs to read this before it has
    any reason to have one.
    """
    try:
        with open(_SKILL_SCHEMA_PATH) as f:
            return JSONResponse(content=json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Skill schema unavailable: {exc}")


@app.post("/jobs", response_model=JobResponse, dependencies=[Security(_require_api_key)])
def create_job(request: ClipRequest):
    job_id = uuid.uuid4().hex
    job = JobResponse(job_id=job_id, status=JobStatus.QUEUED)

    try:
        r = get_redis_connection()
        save_job(r, job)
    except RedisError:
        raise HTTPException(status_code=503, detail="Could not reach job storage. Please try again.")

    try:
        queue = get_queue()
        queue.enqueue(
            process_job,
            job_id,
            request.model_dump(),
            _base_url(),
            job_timeout="2h",
        )
    except RedisError:
        # The job record exists but was never queued - mark it as failed
        # so it doesn't sit at "queued" forever.
        job.status = JobStatus.FAILED
        job.error = "Failed to enqueue job for processing."
        try:
            save_job(r, job)
        except RedisError:
            pass
        raise HTTPException(status_code=503, detail="Could not enqueue job. Please try again.")

    return job


@app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Security(_require_api_key)])
def get_job(job_id: str):
    try:
        r = get_redis_connection()
        job = load_job(r, job_id)
    except RedisError:
        raise HTTPException(status_code=503, detail="Could not reach job storage. Please try again.")

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/clips/{filename}", dependencies=[Security(_require_api_key)])
def get_clip(filename: str):
    # Guard against path traversal.
    safe_name = os.path.basename(filename)
    file_path = os.path.join(CLIPS_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(file_path, media_type="video/mp4", filename=safe_name)

