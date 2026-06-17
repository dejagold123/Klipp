"""
Orchestrates the full Klipp pipeline for a single job:

  download -> transcribe -> chunk -> rank -> clip -> save results
"""

import logging
import os
import shutil
import time
from typing import Optional

import requests

from app import config, downloader, transcriber, llm, clipper
from app.models import ClipRequest, JobMetadata, JobResponse, JobStatus
from app.queue import get_redis_connection
from app.storage import load_job, save_job, update_job_status

logger = logging.getLogger(__name__)

WORK_DIR = config.WORK_DIR
CLIPS_DIR = config.CLIPS_DIR


def _notify_webhook(webhook_url: Optional[str], job: JobResponse) -> None:
    """
    Best-effort POST of the final job result to a webhook, if one was
    requested (per-job `webhook_url`, falling back to the global
    PHAROS_WEBHOOK_URL) and Pharos notifications are enabled. Failures here
    are logged, not raised - a broken webhook shouldn't flip a successful
    job to FAILED.
    """
    if not config.PHAROS_ENABLED:
        return

    target = webhook_url or config.PHAROS_WEBHOOK_URL
    if not target:
        return

    try:
        requests.post(target, json=job.model_dump(mode="json"), timeout=10)
    except requests.RequestException as exc:
        logger.warning(f"Webhook delivery to {target} failed for job {job.job_id}: {exc}")


def process_job(job_id: str, request_data: dict, base_url: str = "") -> None:
    """
    Entry point invoked by the RQ worker. Runs the full pipeline and
    updates the job's status/result in Redis as it progresses.

    Any exception is caught and recorded as a FAILED status so the API
    can report a clear error to the client.
    """
    r = get_redis_connection()
    request = ClipRequest(**request_data)
    job_work_dir = os.path.join(WORK_DIR, job_id)
    started_at = time.time()

    # OpenAI is the supported, working default for both providers. A
    # request can opt into an experimental provider explicitly; otherwise
    # we fall back to the deployment-wide DEFAULT_*_PROVIDER setting
    # (itself "openai" unless overridden via env var).
    stt_provider = request.stt_provider or config.DEFAULT_STT_PROVIDER
    llm_provider = request.llm_provider or config.DEFAULT_LLM_PROVIDER

    try:
        os.makedirs(job_work_dir, exist_ok=True)

        update_job_status(r, job_id, JobStatus.DOWNLOADING)
        video_path, video_title = downloader.download_video(request.video_url, job_work_dir, job_id)
        video_duration = clipper.get_video_duration(video_path)

        update_job_status(r, job_id, JobStatus.TRANSCRIBING)
        audio_path = transcriber.extract_audio(video_path, job_work_dir, job_id)
        segments = transcriber.transcribe_audio(audio_path, provider=stt_provider)
        transcript_length = sum(len(seg.text.split()) for seg in segments)

        update_job_status(r, job_id, JobStatus.RANKING)
        chunks = llm.chunk_segments(segments)
        moments = llm.rank_moments(
            chunks,
            max_clips=request.max_clips,
            min_clip_seconds=request.min_clip_seconds,
            max_clip_seconds=request.max_clip_seconds,
            provider=llm_provider,
        )

        update_job_status(r, job_id, JobStatus.CLIPPING)
        clip_results = clipper.cut_clips(video_path, moments, CLIPS_DIR, job_id, base_url)

        job = load_job(r, job_id)
        if job is None:
            job = JobResponse(job_id=job_id, status=JobStatus.DONE)
        job.status = JobStatus.DONE
        job.error = None
        job.clips = clip_results
        job.metadata = JobMetadata(
            video_title=video_title,
            duration_seconds=video_duration,
            transcript_length=transcript_length,
            processing_time_seconds=time.time() - started_at,
        )
        save_job(r, job)
        _notify_webhook(request.webhook_url, job)

    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        try:
            update_job_status(r, job_id, JobStatus.FAILED, error=str(exc))
            failed_job = load_job(r, job_id)
        except KeyError:
            # Job record is missing entirely (e.g. expired) - create a
            # fresh one so the failure is still visible to the client.
            failed_job = JobResponse(job_id=job_id, status=JobStatus.FAILED, error=str(exc))
            save_job(r, failed_job)
        _notify_webhook(request.webhook_url, failed_job)

    finally:
        # Clean up the working directory (downloaded video + extracted audio),
        # but keep the final clips in CLIPS_DIR.
        shutil.rmtree(job_work_dir, ignore_errors=True)
