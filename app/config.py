# Configuration for Klipp - AI-powered video clip extraction
#
# This is the single source of truth for paths, provider defaults, and
# Pharos agent-skill integration settings. app/main.py and app/processor.py
# import their values from here rather than re-declaring their own
# defaults, so the two processes can't silently disagree about where clips
# live or what "the default provider" means.

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CORE CONFIGURATION
# ============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CLIPS_DIR = os.getenv("KLIPP_CLIPS_DIR", "/data/clips")
WORK_DIR = os.getenv("KLIPP_WORK_DIR", "/tmp/klipp")

# Ensure directories exist
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================================
# PROVIDER CONFIGURATION (Speech-to-Text)
# ============================================================================

# OpenAI Whisper is the supported, working default. Vosk and wav2vec are
# opt-in, experimental alternatives for avoiding per-call API costs - set
# DEFAULT_STT_PROVIDER, or pass `stt_provider` on an individual /jobs
# request, to switch.
DEFAULT_STT_PROVIDER = os.getenv("DEFAULT_STT_PROVIDER", "openai")
AVAILABLE_STT_PROVIDERS = ["openai", "vosk", "wav2vec"]

STT_CONFIG = {
    "openai": {
        "model": "whisper-1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "cost": "~$0.02 per minute of audio",
    },
    "vosk": {
        "model_name": "en",  # language code, passed to vosk.Model(lang=...)
        "sample_rate": 16000,
        "cost": "FREE (offline, no API key)",
    },
    "wav2vec": {
        "model_name": "facebook/wav2vec2-base-960h",
        "device": "cuda",  # or "cpu" for CPU-only
        "cost": "FREE (local inference; needs a GPU to be fast enough for production use)",
    },
}

# ============================================================================
# PROVIDER CONFIGURATION (Language Models)
# ============================================================================

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
AVAILABLE_LLM_PROVIDERS = ["openai", "mistral", "llama", "phi", "local"]

LLM_CONFIG = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "temperature": 0.3,
        "cost": "~$0.15 per 1M input tokens",
    },
    "mistral": {
        "model_name": "mistral-7b-instruct-v0.1",
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
        "temperature": 0.7,
        "max_tokens": 1000,
        "cost": "Cheap - needs a Mistral API key (free tier available, not zero-signup)",
    },
    "llama": {
        "model_path": os.getenv("LLAMA_MODEL_PATH", "./models/llama-2-7b-chat.gguf"),
        "n_ctx": 2048,
        "n_threads": 4,
        "cost": "FREE (local inference) - you must supply the model weights yourself",
    },
    "phi": {
        "model_name": "microsoft/phi-3-mini-4k-instruct",
        "device": "cuda",  # or "cpu"
        "temperature": 0.7,
        "cost": "FREE (local inference); downloads several GB of weights on first use",
    },
    "local": {
        "type": "ollama",  # Ollama local inference
        "endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "mistral"),
        "cost": "FREE (requires Ollama running and reachable from the app)",
    },
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

YDLP_CONFIG = {
    "format": "best[ext=mp4]",
    "quiet": True,
    "no_warnings": True,
}

FFMPEG_CONFIG = {
    "audio_sample_rate": 16000,
    "audio_channels": 1,
    "audio_bitrate": "64k",
}

# ============================================================================
# JOB PROCESSING
# ============================================================================

MAX_JOB_TIMEOUT = 3600  # 1 hour in seconds
MAX_CONCURRENT_JOBS = 5
JOB_RETRY_ATTEMPTS = 3
JOB_RETRY_DELAY = 60  # seconds

# Clip constraints
MIN_CLIP_DURATION = 5  # seconds
MAX_CLIP_DURATION = 180  # seconds
DEFAULT_MAX_CLIPS = 3
DEFAULT_MIN_CLIP_SECONDS = 15
DEFAULT_MAX_CLIP_SECONDS = 60

# Audio chunking for Whisper (max 25MB)
MAX_AUDIO_SIZE_MB = 24
AUDIO_CHUNK_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024  # bytes

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# API
# ============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))

# ============================================================================
# SECURITY
# ============================================================================

# Optional API key, required on /jobs and /clips when set. Matches the
# KLIPP_API_KEY name used in .env.example and app/main.py - this used to be
# read under a different name ("API_KEY") here, which meant setting it in
# this file had no effect on the actual auth check.
API_KEY = os.getenv("KLIPP_API_KEY", None)

# ============================================================================
# PHAROS INTEGRATION
# ============================================================================

# Master switch for the agent-skill webhook notification feature below.
PHAROS_ENABLED = os.getenv("PHAROS_ENABLED", "true").lower() == "true"

# Default webhook to notify on job completion/failure when a request
# doesn't supply its own `webhook_url`. Leave unset to disable by default.
PHAROS_WEBHOOK_URL = os.getenv("PHAROS_WEBHOOK_URL", None)
PHAROS_SKILL_ID = "videoclipextractor-skill-001"

# ============================================================================
# DEVELOPMENT
# ============================================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
