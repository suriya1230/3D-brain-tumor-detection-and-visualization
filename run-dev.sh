#!/usr/bin/env bash
# Starts backend and frontend together. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d backend/.venv ]; then
  echo "backend/.venv missing - see README.md, step 1"
  exit 1
fi

trap 'kill 0' EXIT
# Frontend must run on 3001 - backend/app/main.py's CORS allowlist only
# accepts that origin by default (see ALLOW_ORIGINS).
( cd backend && .venv/bin/uvicorn app.main:app --port 8000 --reload ) &
( cd frontend && npm run dev -- -p 3001 ) &
wait
