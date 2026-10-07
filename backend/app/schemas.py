"""Pydantic request/response models. API contracts live here."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness probe result."""

    status: Literal["ok"]
    version: str


class CheckStatus(BaseModel):
    """Result of one readiness dependency check."""

    ok: bool
    # Short, non-sensitive explanation (e.g. an exception class name).
    detail: str | None = None


class ReadyResponse(BaseModel):
    """Readiness probe result: one entry per dependency."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]
