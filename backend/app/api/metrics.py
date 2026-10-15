"""Evaluation dashboard (admins only): live quality, latency, cost and the queries that went wrong.

Everything is aggregated from ``query_events`` (one row per chat turn or analysis run, failures
included). Question text is shown here on purpose: an administrator needs it to see why a query
failed or was refused (README, Decisions).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Query

from app.deps import AdminUser, DbSession
from app.schemas import QueryEventOut
from app.services import metrics_service

router = APIRouter(prefix="/metrics", tags=["metrics"])

Scope = Literal["all", "chat", "analysis"]


@router.get("/summary", summary="Dashboard figures for a time window")
def summary(
    _: AdminUser,
    db: DbSession,
    days: int = Query(default=7, ge=1, le=365),
    kind: Scope = "all",
) -> dict[str, Any]:
    """Retrieval relevance, groundedness, abstention, latency, token cost, suggestion use, per day."""
    return metrics_service.summary(db, days, kind)


@router.get("/events", response_model=list[QueryEventOut], summary="Problem queries")
def events(
    _: AdminUser,
    db: DbSession,
    outcome: Literal["failed", "refused", "flagged"] = "failed",
    kind: Scope = "all",
    limit: int = Query(default=50, ge=1, le=200),
    before: datetime | None = None,
) -> list[QueryEventOut]:
    """Recent failed, refused or flagged queries, newest first; page with ``before``."""
    rows = metrics_service.list_events(db, outcome, kind, limit, before)
    return [QueryEventOut.model_validate(row) for row in rows]
