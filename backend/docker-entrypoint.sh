#!/bin/sh
# Container entrypoint: apply migrations, then serve.
set -eu

# uvicorn reads these when --workers is absent, so a platform or .env that sets one would start
# several workers, each with its own BM25 index (and a second startup would fail the first
# worker's running ingestions). Refuse instead of serving a quietly inconsistent app.
for name in WEB_CONCURRENCY UVICORN_WORKERS; do
    value=$(printenv "$name" || true)
    if [ -n "$value" ] && [ "$value" != "1" ]; then
        echo "$name=$value: GroundedRAG must run ONE worker (the BM25 index is in-process)." >&2
        exit 1
    fi
done

alembic upgrade head

# Single worker on purpose: the BM25 index is in-process memory (CLAUDE.md rule
# 4). Do NOT raise --workers. Access logs come from app/middleware.py.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --no-access-log \
    --no-server-header
