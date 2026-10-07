"""Request ID, JSON logging and access-log privacy."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware import JsonFormatter, configure_logging, request_id_var
from tests.conftest import LogCollector


async def _boom() -> None:
    raise RuntimeError("kaboom")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app.add_api_route("/api/_test/boom", _boom)
    # raise_server_exceptions=False lets us observe the real 500 response.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_response_carries_a_generated_request_id(client: TestClient) -> None:
    response = client.get("/api/health")
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32 and request_id.isalnum()


def test_valid_caller_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/api/health", headers={"X-Request-ID": "trace-1234.abc"})
    assert response.headers["x-request-id"] == "trace-1234.abc"


@pytest.mark.parametrize("bad", ["short", "has space in it", "x" * 65, "semi;colon-12345"])
def test_unsafe_caller_request_id_is_replaced(client: TestClient, bad: str) -> None:
    response = client.get("/api/health", headers={"X-Request-ID": bad})
    assert response.headers["x-request-id"] != bad


def test_access_log_has_method_path_status_duration(
    client: TestClient, request_logs: LogCollector
) -> None:
    client.get("/api/does-not-exist")
    record = request_logs.records[-1]
    assert (record.method, record.path, record.status) == ("GET", "/api/does-not-exist", 404)  # type: ignore[attr-defined]
    assert record.duration_ms >= 0  # type: ignore[attr-defined]


def test_access_log_never_contains_query_string(
    client: TestClient, request_logs: LogCollector
) -> None:
    client.get("/api/does-not-exist?q=what+is+the+secret+salary")
    rendered = json.dumps([JsonFormatter().format(record) for record in request_logs.records])
    assert "salary" not in rendered
    assert "q=" not in rendered


def test_successful_health_probes_are_logged_below_info(
    client: TestClient, request_logs: LogCollector
) -> None:
    client.get("/api/health")
    assert request_logs.records[-1].levelno == logging.DEBUG


def test_unhandled_exception_returns_generic_500_with_request_id(
    client: TestClient, request_logs: LogCollector
) -> None:
    response = client.get("/api/_test/boom", headers={"X-Request-ID": "trace-boom-0001"})
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error", "request_id": "trace-boom-0001"}
    assert "kaboom" not in response.text
    assert request_logs.records[-1].status == 500  # type: ignore[attr-defined]


def test_uvicorn_access_logging_stays_disabled() -> None:
    """uvicorn's access lines carry query strings; they must not be re-enabled.

    uvicorn only logs access when ``uvicorn.access`` has handlers (directly or by
    propagation), so configure_logging must leave it with neither.
    """
    configure_logging("INFO")
    access = logging.getLogger("uvicorn.access")
    assert access.hasHandlers() is False
    assert logging.getLogger("uvicorn.error").propagate is True


def test_json_formatter_drops_uvicorn_color_message() -> None:
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, "Started", (), None)
    record.color_message = "\x1b[36mStarted\x1b[0m"
    payload = json.loads(JsonFormatter().format(record))
    assert "color_message" not in payload


def test_json_formatter_emits_one_json_object_with_extras_and_request_id() -> None:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.chunk_count = 7
    token = request_id_var.set("req-abcdef12")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["chunk_count"] == 7
    assert payload["request_id"] == "req-abcdef12"
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_includes_traceback_for_exceptions() -> None:
    try:
        raise ValueError("bad")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("app.test", logging.ERROR, __file__, 1, "failed", (), exc_info)
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError" in payload["exc"]
