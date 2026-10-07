"""Request ID, structured JSON logging and request timing.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) so streamed
responses such as the ``/api/chat`` SSE stream pass through untouched.

Privacy (CLAUDE.md rule 21): request logs carry only method, path, status and
duration. Query strings, bodies, questions, document text and tokens are never
logged.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"

# Context for log lines emitted while a request is being handled.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Only accept caller-supplied IDs that cannot be used to forge log lines.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Liveness probes fire every few seconds; keep them out of the INFO stream.
_QUIET_PATHS = frozenset({"/api/health", "/healthz"})

# Attributes that are part of every LogRecord (or uvicorn's ANSI-colored
# duplicate of the message) and so must not be copied into the JSON payload.
_RESERVED_LOG_ATTRS = frozenset(
    set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
    | {"message", "asctime", "taskName", "color_message"}
)

logger = logging.getLogger("app.request")


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Route all logging (including uvicorn's) through one JSON handler on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers; hand its server/error records to the
    # root handler.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # uvicorn logs access lines only if ``uvicorn.access.hasHandlers()``, so
    # propagating this logger to the root handler would silently re-enable them
    # despite --no-access-log. Its lines include the full request path with the
    # query string; RequestContextMiddleware logs a privacy-safe line instead.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False

    # HTTP client libraries can log URLs and payload details at INFO/DEBUG.
    for name in ("httpx", "httpcore", "urllib3", "google", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)


class RequestContextMiddleware:
    """Attach a request ID, time the request and emit one access log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get(REQUEST_ID_HEADER)
        if supplied and _VALID_REQUEST_ID.match(supplied):
            request_id = supplied
        else:
            request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        started = time.perf_counter()
        status = 500  # stays 500 if the app raises before sending a response

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            path = scope["path"]
            log = logger.debug if path in _QUIET_PATHS and status < 400 else logger.info
            log(
                "request",
                extra={
                    "method": scope["method"],
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)
