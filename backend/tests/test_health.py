"""Liveness and readiness endpoints."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.core import vectorstore
from app.main import app
from app.services import health_service
from app.services.health_service import CheckResult


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


OK = CheckResult(ok=True)


def patch_checks(
    monkeypatch: pytest.MonkeyPatch,
    postgres: CheckResult = OK,
    chroma: CheckResult = OK,
) -> None:
    monkeypatch.setattr(health_service, "check_postgres", lambda: postgres)
    monkeypatch.setattr(health_service, "check_chroma", lambda: chroma)


def test_health_is_alive_without_touching_dependencies(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_ready_reports_ok_when_all_dependencies_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_checks(monkeypatch)
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "postgres": {"ok": True, "detail": None},
            "chroma": {"ok": True, "detail": None},
        },
    }


def test_ready_returns_503_when_a_dependency_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_checks(monkeypatch, chroma=CheckResult(ok=False, detail="ConnectError"))
    response = client.get("/api/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["postgres"]["ok"] is True
    assert body["checks"]["chroma"] == {"ok": False, "detail": "ConnectError"}


def test_ready_times_out_slow_dependency(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow() -> CheckResult:
        time.sleep(0.3)
        return CheckResult(ok=True)

    monkeypatch.setattr(health_service, "CHECK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(health_service, "check_postgres", slow)
    monkeypatch.setattr(health_service, "check_chroma", lambda: CheckResult(ok=True))
    response = client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["postgres"] == {"ok": False, "detail": "timeout"}


def test_postgres_failure_reports_only_the_exception_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> None:
        raise RuntimeError("password=hunter2 host=internal-db")

    monkeypatch.setattr(health_service.database, "ping", boom)
    result = health_service.check_postgres()
    assert result == CheckResult(ok=False, detail="RuntimeError")


def test_chroma_check_passes_when_heartbeat_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vectorstore, "heartbeat", lambda: 1)
    assert health_service.check_chroma() == CheckResult(ok=True)


def test_chroma_check_fails_when_server_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def down() -> int:
        raise vectorstore.VectorStoreError("cannot connect to ChromaDB: ConnectError")

    monkeypatch.setattr(vectorstore, "heartbeat", down)
    result = health_service.check_chroma()
    assert result == CheckResult(ok=False, detail="VectorStoreError")


def test_openapi_is_served_under_api_prefix(client: TestClient) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/health" in paths and "/api/ready" in paths
