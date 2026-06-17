# How to Clone the Klipp Repository

A complete step-by-step guide for cloning and setting up the Klipp project locally.

---

## Prerequisites

Before you clone the repository, make sure you have the following installed:

- **Git** — Version control system ([install here](https://git-scm.com/downloads))
- **Python 3.11+** — Programming language ([install here](https://www.python.org/downloads/))
- **Redis** — In-memory data store for job queues ([install here](https://redis.io/download))
- **FFmpeg** — Video processing tool ([install here](https://ffmpeg.org/download.html))
- **OpenAI API Key** — For transcription and analysis ([get here](https://platform.openai.com/api-keys))

### Installation Verification

Run these commands to verify your installations:

```bash
git --version
python --version
redis-server --version
ffmpeg -version
```

---

## Step 1: Clone the Repository

Use Git to clone the Klipp repository to your local machine:

```bash
git clone https://github.com/dejagold123/Klipp.git
cd Klipp
```

This creates a directory called `Klipp` with all the project files.

---

## Step 2: Set Up Python Virtual Environment

Create and activate a Python virtual environment to isolate project dependencies:

### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt when activated.

---

## Step 3: Install Dependencies

Install all required Python packages from the requirements file:

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI — Web framework for the REST API
- Redis-py — Python Redis client
- RQ (Redis Queue) — Job queue library
- yt-dlp — Video downloader
- OpenAI — For Whisper transcription and GPT-4o-mini ranking
- FFmpeg-python — Python wrapper for FFmpeg
- Pydantic — Data validation

---

## Step 4: Configure Environment Variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Open `.env` in your text editor and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
KLIPP_CLIPS_DIR=/data/clips
KLIPP_WORK_DIR=/tmp/klipp
```

**Important:** Never commit your `.env` file to version control. The `.gitignore` should already exclude it.

---

## Step 5: Start Redis

Open a new terminal window and start the Redis server:

```bash
redis-server
```

You should see output like:
```
* Ready to accept connections
```

Keep this terminal open — Redis needs to run in the background.

---

## Step 6: Run the Application

### Terminal 1 — Start the API Server

```bash
source venv/bin/activate   # or: venv\Scripts\activate on Windows
export $(cat .env | xargs)
uvicorn app.main:app --reload
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Start the Worker

In a new terminal window:

```bash
source venv/bin/activate   # or: venv\Scripts\activate on Windows
export $(cat .env | xargs)
python -m app.worker
```

You should see:
```
Worker started, listening to queues...
```

---

## Step 7: Test the Setup

Open another terminal and test the health endpoint:

```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{"status": "ok"}
```

---

## Step 8: Submit Your First Job

Submit a test video URL to extract clips:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "max_clips": 3,
    "min_clip_seconds": 15,
    "max_clip_seconds": 60
  }'
```

This returns a job ID. Save it and poll for status:

```bash
curl http://localhost:8000/jobs/<job_id>
```

Job status progresses through: `queued` → `downloading` → `transcribing` → `ranking` → `clipping` → `done`

---

## Step 9: Download Your Clips

Once the job is complete, download generated clips:

```bash
curl -O http://localhost:8000/clips/<clip_filename>.mp4
```

---

## Troubleshooting

### Redis connection error
**Problem:** `ConnectionError: Error 111 connecting to 127.0.0.1:6379`  
**Solution:** Make sure Redis is running in a separate terminal (`redis-server`)

### FFmpeg not found
**Problem:** `FileNotFoundError: ffmpeg not found`  
**Solution:** Install FFmpeg and ensure it's in your system PATH:
```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### OpenAI API error
**Problem:** `AuthenticationError: Incorrect API key provided`  
**Solution:** Double-check your API key in `.env` and ensure it's valid

### Python version error
**Problem:** `Python 3.11+ is required`  
**Solution:** Install a compatible Python version and create a new virtual environment

### Import errors after cloning
**Problem:** `ModuleNotFoundError: No module named 'app'`  
**Solution:** Make sure you're in the `Klipp` directory and have activated the virtual environment

---

## Project Structure Reference

```
Klipp/
├── app/
│   ├── main.py         # FastAPI endpoints
│   ├── worker.py       # RQ worker
│   ├── processor.py    # Pipeline orchestration
│   ├── config.py       # Paths, provider defaults, Pharos settings
│   ├── llm.py          # Transcript chunking & ranking
│   ├── downloader.py   # Video download (yt-dlp)
│   ├── transcriber.py  # Audio extraction & transcription
│   ├── clipper.py      # FFmpeg clipping
│   ├── models.py       # Pydantic data models
│   ├── queue.py        # Redis connection
│   └── storage.py      # Job state management
├── clips/              # Output directory for generated clips
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore file
├── start.sh            # Production startup script (runs API + worker together)
├── nixpacks.toml       # Build config used if deploying via Railway (optional otherwise)
└── README.md           # Main documentation
```

---

## Next Steps

- Read the [API Reference](README.md#api-reference) in the main README for detailed endpoint documentation
- Explore the [source code](https://github.com/dejagold123/Klipp/tree/main/app) to understand the architecture
- See [Deployment](README.md#deployment) in the main README for how to run this somewhere other than your laptop
- Submit issues or contribute improvements on GitHub

---

## Useful Commands Reference

| Task | Command |
|------|---------|
| Activate venv (Linux/macOS) | `source venv/bin/activate` |
| Activate venv (Windows) | `venv\Scripts\activate` |
| Deactivate venv | `deactivate` |
| Start Redis | `redis-server` |
| Start API | `uvicorn app.main:app --reload` |
| Start Worker | `python -m app.worker` |
| Test health | `curl http://localhost:8000/healthz` |
| Submit job | `curl -X POST http://localhost:8000/jobs -H "Content-Type: application/json" -d '{...}'` |

---

Happy cloning! 🚀
