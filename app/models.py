"""
Pydantic models shared across the Klipp app.
"""

from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from app import config


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    RANKING = "ranking"
    CLIPPING = "clipping"
    DONE = "done"
    FAILED = "failed"


class ClipRequest(BaseModel):
    """Incoming request to create a new clipping job."""

    video_url: str = Field(..., description="URL of the source video (e.g. YouTube link)")
    max_clips: int = Field(default=3, ge=1, le=10, description="Max number of clips to produce")
    min_clip_seconds: int = Field(default=15, ge=5, le=170, description="Minimum clip duration in seconds")
    max_clip_seconds: int = Field(default=60, ge=5, le=180, description="Maximum clip duration in seconds")
    llm_provider: Optional[str] = Field(
        default=None,
        description=(
            "Moment-ranking provider. Defaults to DEFAULT_LLM_PROVIDER (openai) "
            f"if omitted. One of: {config.AVAILABLE_LLM_PROVIDERS}"
        ),
    )
    stt_provider: Optional[str] = Field(
        default=None,
        description=(
            "Transcription provider. Defaults to DEFAULT_STT_PROVIDER (openai) "
            f"if omitted. One of: {config.AVAILABLE_STT_PROVIDERS}"
        ),
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional URL to POST the final job result to when it finishes (done or failed).",
    )

    @field_validator("video_url", "webhook_url")
    @classmethod
    def validate_url_fields(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"{info.field_name} must be an http(s) URL")
        if not parsed.netloc:
            raise ValueError(f"{info.field_name} must include a host")
        return v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in config.AVAILABLE_LLM_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {config.AVAILABLE_LLM_PROVIDERS}")
        return v

    @field_validator("stt_provider")
    @classmethod
    def validate_stt_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in config.AVAILABLE_STT_PROVIDERS:
            raise ValueError(f"stt_provider must be one of {config.AVAILABLE_STT_PROVIDERS}")
        return v

    @model_validator(mode="after")
    def validate_clip_lengths(self) -> "ClipRequest":
        if self.max_clip_seconds < self.min_clip_seconds:
            raise ValueError("max_clip_seconds must be >= min_clip_seconds")
        return self


class TranscriptSegment(BaseModel):
    """A single timestamped segment from Whisper."""

    start: float
    end: float
    text: str


class RankedMoment(BaseModel):
    """A moment selected by the LLM as worth clipping."""

    start: float
    end: float
    reason: str
    score: float = Field(ge=0, le=10)


class ClipResult(BaseModel):
    """Metadata about a produced clip file."""

    clip_filename: str
    download_url: str
    start: float
    end: float
    duration: float
    reason: str
    score: float


class JobMetadata(BaseModel):
    """Extra context about a completed job, populated once available."""

    video_title: Optional[str] = None
    duration_seconds: Optional[float] = None
    transcript_length: Optional[int] = None
    processing_time_seconds: Optional[float] = None


class JobResponse(BaseModel):
    """Response returned when a job is created or queried."""

    job_id: str
    status: JobStatus
    error: Optional[str] = None
    clips: list[ClipResult] = Field(default_factory=list)
    metadata: Optional[JobMetadata] = None
