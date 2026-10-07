"""Liveness (`/health`) and readiness (`/ready`) endpoints. No authentication."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.schemas import CheckStatus, HealthResponse, ReadyResponse
from app.services import health_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and serving requests. Touches no dependency."""
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "A dependency is unavailable"}},
)
async def ready() -> JSONResponse:
    """Readiness: Postgres and ChromaDB are reachable. Returns 503 if any check fails."""
    results = await health_service.run_readiness()
    checks = {
        name: CheckStatus(ok=result.ok, detail=result.detail) for name, result in results.items()
    }
    all_ok = all(check.ok for check in checks.values())
    body = ReadyResponse(status="ready" if all_ok else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if all_ok else 503, content=body.model_dump())
