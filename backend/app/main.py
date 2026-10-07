"""FastAPI application factory.

Run with a single uvicorn worker only: the BM25 index lives in process memory
(CLAUDE.md rule 4). Routers are mounted under ``/api``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import auth, health
from app.config import get_settings
from app.constants import API_PREFIX
from app.db import database
from app.middleware import RequestContextMiddleware, configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks."""
    settings = get_settings()
    logger.info(
        "backend starting",
        extra={"version": __version__, "collection": settings.collection_name},
    )
    yield
    database.dispose_engine()
    logger.info("backend stopped")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 with the request ID; the details stay in the logs."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


def create_app() -> FastAPI:
    """Build the application: settings, logging, middleware and routers."""
    settings = get_settings()  # fail fast on bad configuration
    configure_logging(settings.log_level)

    app = FastAPI(
        title="GroundedRAG",
        version=__version__,
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    api_router = APIRouter(prefix=API_PREFIX)
    api_router.include_router(health.router)
    api_router.include_router(auth.router)
    app.include_router(api_router)
    return app


app = create_app()
