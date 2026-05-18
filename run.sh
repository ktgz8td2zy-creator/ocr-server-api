#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

export OCR_ENGINE="${OCR_ENGINE:-rapidocr}"
HOST="${OCR_HOST:-0.0.0.0}"
PORT="${OCR_PORT:-9418}"

echo "Starting OCR Server [engine=$OCR_ENGINE] on ${HOST}:${PORT}"
echo "API docs: http://localhost:${PORT}/docs"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "${OCR_WORKERS:-1}"
