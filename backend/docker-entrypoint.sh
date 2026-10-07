#!/bin/sh
# Container entrypoint: apply migrations, then serve.
set -eu

alembic upgrade head

# Single worker on purpose: the BM25 index is in-process memory (CLAUDE.md rule
# 4). Do NOT add --workers. Access logs come from app/middleware.py.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log \
    --no-server-header
