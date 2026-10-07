"""Readiness checks for the dependencies the backend needs to serve traffic.

``api/`` must not touch Chroma or the database directly (CLAUDE.md rule 1), so
the probes live here. Failure details are deliberately limited to exception
class names: connection errors can embed hosts or credentials.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core import vectorstore
from app.db import database

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 4.0


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


async def _run_with_timeout(check: Callable[[], CheckResult]) -> CheckResult:
    """Run a blocking check in a worker thread, bounded by ``CHECK_TIMEOUT_SECONDS``."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(check), timeout=CHECK_TIMEOUT_SECONDS)
    except TimeoutError:
        return CheckResult(ok=False, detail="timeout")


async def run_readiness() -> dict[str, CheckResult]:
    """Run every readiness check concurrently and return results by dependency name."""
    names = ("postgres", "chroma")
    results = await asyncio.gather(
        _run_with_timeout(check_postgres),
        _run_with_timeout(check_chroma),
    )
    return dict(zip(names, results, strict=True))
