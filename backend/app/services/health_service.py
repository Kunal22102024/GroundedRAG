"""Readiness checks for the dependencies the backend needs to serve traffic.

``api/`` must not touch Chroma or the database directly (CLAUDE.md rule 1), so
the probes live here. Failure details are deliberately limited to exception
class names: connection errors can embed hosts or credentials.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.core import embeddings, vectorstore
from app.db import database

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 4.0
# The embedding probe is a real API call (measured 0.8 to 6 s), so it gets more time, and its
# result is cached: every probe costs one request out of a free-tier 100/minute.
EMBEDDING_TIMEOUT_SECONDS = 20.0
EMBEDDING_OK_TTL_SECONDS = 300.0
EMBEDDING_FAILURE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one dependency check."""

    ok: bool
    detail: str | None = None


def check_postgres() -> CheckResult:
    """Verify PostgreSQL answers ``SELECT 1``."""
    try:
        database.ping()
    except Exception as exc:
        logger.warning("postgres readiness check failed", extra={"error": type(exc).__name__})
        return CheckResult(ok=False, detail=type(exc).__name__)
    return CheckResult(ok=True)


def check_chroma() -> CheckResult:
    """Verify ChromaDB answers its heartbeat."""
    try:
        vectorstore.heartbeat()
    except Exception as exc:
        logger.warning("chroma readiness check failed", extra={"error": type(exc).__name__})
        return CheckResult(ok=False, detail=type(exc).__name__)
    return CheckResult(ok=True)


_embedding_lock = threading.Lock()
_embedding_cache: tuple[float, CheckResult] | None = None


def reset_embedding_cache() -> None:
    """Forget the cached embedding probe result (tests)."""
    global _embedding_cache
    with _embedding_lock:
        _embedding_cache = None


def check_embeddings() -> CheckResult:
    """Verify the Gemini embedding service answers and accepts the API key (cached)."""
    global _embedding_cache
    with _embedding_lock:
        if _embedding_cache is not None and time.monotonic() < _embedding_cache[0]:
            return _embedding_cache[1]
    try:
        state = embeddings.probe()
    except embeddings.EmbeddingConfigError:
        result = CheckResult(ok=False, detail="GOOGLE_API_KEY not set")
    except Exception as exc:  # EmbeddingError subclasses, or anything unexpected
        logger.warning("embedding readiness check failed", extra={"error": type(exc).__name__})
        result = CheckResult(ok=False, detail=type(exc).__name__)
    else:
        # A 429 still proves the service is up and the key is valid.
        result = CheckResult(ok=True, detail="rate limited" if state == "rate_limited" else None)
    ttl = EMBEDDING_OK_TTL_SECONDS if result.ok else EMBEDDING_FAILURE_TTL_SECONDS
    with _embedding_lock:
        _embedding_cache = (time.monotonic() + ttl, result)
    return result


async def _run_with_timeout(
    check: Callable[[], CheckResult], timeout: float | None = None
) -> CheckResult:
    """Run a blocking check in a worker thread, bounded by a timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(check), timeout=CHECK_TIMEOUT_SECONDS if timeout is None else timeout
        )
    except TimeoutError:
        return CheckResult(ok=False, detail="timeout")


async def run_readiness() -> dict[str, CheckResult]:
    """Run every readiness check concurrently and return results by dependency name."""
    names = ("postgres", "chroma", "embeddings")
    results = await asyncio.gather(
        _run_with_timeout(check_postgres),
        _run_with_timeout(check_chroma),
        _run_with_timeout(check_embeddings, EMBEDDING_TIMEOUT_SECONDS),
    )
    return dict(zip(names, results, strict=True))
