#!/bin/bash
# Starts the RQ worker in the background, then the FastAPI server in the foreground.
# Both share the same filesystem, so clips written by the worker are served by the API.

set -e

mkdir -p "${KLIPP_CLIPS_DIR:-/data/clips}"
mkdir -p "${KLIPP_WORK_DIR:-/tmp/klipp}"

echo "Starting Klipp worker..."
python -m app.worker &
WORKER_PID=$!

echo "Starting Klipp API..."
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

# If uvicorn exits, kill the worker too
kill $WORKER_PID
