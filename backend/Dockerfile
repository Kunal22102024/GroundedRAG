FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root runtime user (CLAUDE.md rule 30).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first so code changes do not invalidate the layer cache.
# INSTALL_DEV=true (default) adds pytest/ruff so `docker compose exec backend
# pytest` works; pass --build-arg INSTALL_DEV=false for a lean image.
ARG INSTALL_DEV=true
COPY requirements.txt requirements-dev.txt ./
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        pip install -r requirements-dev.txt; \
    else \
        pip install -r requirements.txt; \
    fi

COPY . .

# Upload directory is created here, owned by the app user, so a fresh named
# volume mounted at /data/uploads inherits writable ownership.
RUN mkdir -p /data/uploads \
    && chown -R app:app /data /app \
    && chmod +x /app/docker-entrypoint.sh

USER app

EXPOSE 8000

# Runs migrations, then a single uvicorn worker (see docker-entrypoint.sh).
ENTRYPOINT ["/app/docker-entrypoint.sh"]
