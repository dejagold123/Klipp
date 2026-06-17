# Klipp

Klipp is an autonomous video intelligence pipeline. Give it any video URL and it extracts the most compelling short-form clips automatically — no manual scrubbing required. It's also built as a Pharos-compatible agent skill, so an autonomous agent can discover and call it without a human in the loop.

**How it works:**
1. Downloads the video via `yt-dlp`
2. Extracts audio and transcribes it (OpenAI Whisper API by default, with automatic chunking for long videos; Vosk or wav2vec available as free, offline, opt-in alternatives)
3. Chunks the transcript into timestamped segments
4. Ranks the most engaging moments — hooks, emotional peaks, punchy one-liners (GPT-4o-mini by default; Mistral, local Llama-2, local Phi-3, or Ollama available as opt-in alternatives)
5. Cuts clips with FFmpeg at exact timestamps
6. Returns clip files + metadata (score, reason, timestamps, video title, processing time) via a REST API, and optionally POSTs the result to a webhook when the job finishes

Jobs run asynchronously — submit a URL, poll for status (or get notified via webhook), download your clips.

---

## Architecture

```
POST /jobs (video_url, optional llm_provider/stt_provider/webhook_url)
        │
        ▼
  Redis Queue (RQ)
        │
        ▼
  Worker: download (yt-dlp) → also captures video title
        │
        ▼
  Worker: extract + transcribe audio
        │  (OpenAI Whisper by default; Vosk/wav2vec if requested)
  [auto-splits audio > 24MB into chunks, offsets timestamps, merges]
        │
        ▼
  Worker: chunk transcript → rank moments
        │  (GPT-4o-mini by default; Mistral/Llama-2/Phi-3/Ollama if requested)
  [filters by min/max duration, skips malformed LLM output]
        │
        ▼
  Worker: cut clips (FFmpeg, re-encode for accurate timestamps)
        │
        ▼
  Save result + metadata → fire webhook (if configured)
        │
        ▼
  GET /jobs/{id} → status + clip metadata
  GET /clips/{filename} → download clip
  GET /skill → agent-discoverable schema (Pharos / OpenAI function calling / LangChain)
```

The API and worker run in the **same container** so clip files are accessible to both processes without external storage.

---

## Project Structure

```
klipp/
├── app/
│   ├── main.py         # FastAPI app — /jobs, /clips, /skill, /healthz endpoints
│   ├── worker.py       # RQ worker entrypoint
│   ├── queue.py        # Redis/RQ connection setup
│   ├── processor.py    # pipeline orchestration per job, provider resolution, webhook firing
│   ├── config.py       # single source of truth: paths, provider defaults/lists, Pharos settings
│   ├── llm.py          # transcript chunking + multi-provider moment ranking
│   ├── downloader.py   # yt-dlp video download (also returns video title)
│   ├── transcriber.py  # audio extraction + multi-provider transcription
│   ├── clipper.py      # FFmpeg clipping with timestamp clamping
│   ├── models.py       # Pydantic models + input validation
│   └── storage.py      # Redis-backed job state
├── clips/                       # output clips (served via /clips endpoint)
├── PHAROS_SKILL_SCHEMA.json     # agent-skill schema served at GET /skill
├── start.sh                     # starts worker + API in one container
├── nixpacks.toml                # Nixpacks build config (used if deploying via Railway; optional otherwise)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- Redis (`redis-server`)
- FFmpeg on your `PATH`
- OpenAI API key (required for the default providers; see [Provider Options](#provider-options) if you want to avoid this)

### Setup

```bash
git clone <your-repo-url>
cd klipp
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Start Redis

```bash
redis-server
```

### Run the API (terminal 1)

```bash
export $(cat .env | xargs)
uvicorn app.main:app --reload
```

### Run the worker (terminal 2)

```bash
export $(cat .env | xargs)
python -m app.worker
```

---

## Provider Options

OpenAI (Whisper for transcription, GPT-4o-mini for ranking) is the **default and the only path that's fully tested end-to-end**. The alternatives below exist to avoid per-call API costs and are wired up correctly, but are genuinely experimental — treat them as "available," not "production-hardened."

| Role | Provider | Cost | Notes |
|---|---|---|---|
| Transcription | `openai` (default) | ~$0.02/min | Whisper API |
| Transcription | `vosk` | Free, offline | Auto-downloads a small model on first use; CPU-only, fine on a low-resource host |
| Transcription | `wav2vec` | Free, local | Needs a GPU to be fast enough for real use |
| Ranking | `openai` (default) | ~$0.15/1M input tokens | GPT-4o-mini |
| Ranking | `mistral` | Cheap | Needs a free-tier `MISTRAL_API_KEY` from Mistral — not zero-signup, just cheap |
| Ranking | `llama` | Free, local | You must supply your own GGUF weights at `LLAMA_MODEL_PATH`; not bundled |
| Ranking | `phi` | Free, local | Downloads several GB of weights on first use |
| Ranking | `local` (Ollama) | Free | Requires an Ollama server reachable from the app — most hosting platforms don't run one for you out of the box |

Set a deployment-wide default with `DEFAULT_STT_PROVIDER` / `DEFAULT_LLM_PROVIDER`, or override per-request with `stt_provider` / `llm_provider` (see API Reference below).

---

## API Reference

### Authentication (optional)

If `KLIPP_API_KEY` is set in the environment, `/jobs` and `/clips` require it in an `X-API-Key` header. `/skill` and `/healthz` are always open, since an agent needs to read the skill schema before it has any reason to hold a key.

```bash
curl -H "X-API-Key: your-secret" http://localhost:8000/jobs/...
```

### `POST /jobs` — Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "max_clips": 3,
    "min_clip_seconds": 15,
    "max_clip_seconds": 60
  }'
```

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `video_url` | string | required | Any `http(s)` URL supported by yt-dlp |
| `max_clips` | int | `3` | Max clips to produce (1–10) |
| `min_clip_seconds` | int | `15` | Minimum clip duration in seconds (≥5) |
| `max_clip_seconds` | int | `60` | Maximum clip duration in seconds (≤180) |
| `stt_provider` | string | `openai` | One of `openai`, `vosk`, `wav2vec` |
| `llm_provider` | string | `openai` | One of `openai`, `mistral`, `llama`, `phi`, `local` |
| `webhook_url` | string | none | If set, the final job result is POSTed here when it finishes (done or failed) |

**Response:**

```json
{ "job_id": "a3f9c2...", "status": "queued", "clips": [] }
```

---

### `GET /jobs/{job_id}` — Poll status

```bash
curl http://localhost:8000/jobs/a3f9c2...
```

Job status progresses through:

`queued` → `downloading` → `transcribing` → `ranking` → `clipping` → `done`

(or `failed` with an `error` field explaining what went wrong)

**Response when done:**

```json
{
  "job_id": "a3f9c2...",
  "status": "done",
  "clips": [
    {
      "clip_filename": "a3f9c2..._clip1.mp4",
      "download_url": "/clips/a3f9c2..._clip1.mp4",
      "start": 42.1,
      "end": 78.4,
      "duration": 36.3,
      "reason": "Strong hook — opens with a surprising claim",
      "score": 9.2
    }
  ],
  "metadata": {
    "video_title": "Example Video Title",
    "duration_seconds": 612.4,
    "transcript_length": 1840,
    "processing_time_seconds": 94.2
  }
}
```

`download_url` is relative unless `KLIPP_BASE_URL` is set, in which case it's an absolute URL pointing at your deployment.

---

### `GET /clips/{filename}` — Download a clip

```bash
curl -O http://localhost:8000/clips/a3f9c2..._clip1.mp4
```

### `GET /skill` — Agent discovery schema

Returns `PHAROS_SKILL_SCHEMA.json`, describing inputs, outputs, and available actions in a format autonomous agent frameworks (Pharos, OpenAI function calling, LangChain) can consume directly. No API key required.

```bash
curl http://localhost:8000/skill
```

### `GET /healthz` — Health check

```bash
curl http://localhost:8000/healthz
# { "status": "ok" }
```

---

## Agent / Pharos Integration

Klipp is built to be called by an autonomous agent, not just a human with `curl`:

- **Discovery**: an agent calls `GET /skill` to learn the input/output schema and available actions before ever submitting a job.
- **Async notification**: instead of polling `GET /jobs/{id}` in a loop, an agent can pass `webhook_url` on submission and get the final result POSTed to it directly. Set `PHAROS_WEBHOOK_URL` as a deployment-wide fallback if you want every job notified somewhere by default, and `PHAROS_ENABLED=false` to turn webhook delivery off entirely.
- **Provider choice**: an agent (or you) can pick `stt_provider`/`llm_provider` per request, e.g. to force the free path when cost matters more than speed.

---

## Deployment

Klipp needs, on whatever host you pick:

- Python 3.11+ and FFmpeg installed
- A reachable Redis instance (set `REDIS_URL`)
- Persistent disk for the `clips` directory
- The API and worker processes able to see the **same** clips/work directories — either run them in one container/process group (what `start.sh` does), or as separate services sharing a volume if your platform supports that

Environment variables you'll generally want to set:

```
# Required for the default providers
OPENAI_API_KEY=sk-...

# Storage paths
KLIPP_CLIPS_DIR=/data/clips
KLIPP_WORK_DIR=/tmp/klipp

# Optional - absolute clip URLs, API auth, provider defaults, webhooks
KLIPP_BASE_URL=https://your-deployment.example.com
KLIPP_API_KEY=
DEFAULT_STT_PROVIDER=openai
DEFAULT_LLM_PROVIDER=openai
MISTRAL_API_KEY=
PHAROS_ENABLED=true
PHAROS_WEBHOOK_URL=
```

Once running, verify it:

```bash
curl https://<your-host>/healthz
curl https://<your-host>/skill
curl -X POST https://<your-host>/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=VIDEO_ID", "max_clips": 2}'
```

### Currently running on Railway

This is just what I'm using right now since I don't have a VPS set up yet, not a hard requirement of the project — `nixpacks.toml` and `start.sh` exist in the repo for it, but neither is needed if you're deploying elsewhere (Docker, a VPS, etc.):

1. Push this repo to a GitHub repository.
2. Create a new [Railway](https://railway.app) project → **Deploy from GitHub repo**.
3. Add a **Redis** plugin — Railway injects `REDIS_URL` automatically.
4. Add the environment variables above in the service settings.
5. Railway uses `nixpacks.toml` to install FFmpeg and Python automatically, and runs `start.sh`, which starts both the worker and the API in the same container.

---

## Design Decisions

**Why single container for API + worker?**
Some hosts (Railway included) don't support shared volumes across separate services. Running both in one container/process group via `start.sh` sidesteps that entirely by sharing the local filesystem - no S3 or external file store needed. If your host does support a shared volume between services, splitting them apart works too; the pipeline itself doesn't care.

**Why always re-encode clips instead of stream-copying?**
FFmpeg's stream copy snaps to the nearest keyframe, producing inaccurate cut points that can be seconds off. Re-encoding with `-ss` (input seek) + `-t` (output duration) gives frame-accurate clips at the cost of slightly longer processing time.

**Why split audio before Whisper?**
The Whisper API rejects files over 25MB. For longer videos, we extract a compressed 64kbps mono 16kHz audio track via FFmpeg, then split it into sub-24MB chunks if needed. Each chunk is transcribed separately, timestamps are offset by the chunk's position in the original audio, and the results are merged into one continuous timeline.

**Why OpenAI as the default despite supporting free alternatives?**
It's the only combination that's actually been run end-to-end reliably. Vosk and Mistral are real, working opt-in options if you want to avoid API costs, but Llama-2/Phi-3/Ollama require resources (multi-GB local weights, a GPU, or a self-hosted Ollama server) that a typical low-resource deployment won't have by default. `DEFAULT_STT_PROVIDER`/`DEFAULT_LLM_PROVIDER` let you flip the deployment-wide default once you've validated a free path works on whatever you're running.

**Why GPT-4o-mini for ranking?**
Fast, cheap, and capable enough for transcript analysis. The ranking prompt is strict JSON — we use `response_format: json_object` and treat the model's output as untrusted: individual malformed moments are skipped, durations are re-validated client-side, and empty/invalid responses raise clear errors rather than silently returning no clips.

---

## Known Gaps

- No automated test suite yet.
- Llama-2/Phi-3 local inference is unverified on a low-resource (CPU-only, limited RAM) host - expect it to be slow or to not fit at all without a beefier machine.

---
