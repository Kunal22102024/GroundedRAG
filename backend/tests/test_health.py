"""Liveness and readiness endpoints."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.core import embeddings, vectorstore
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
    embeddings_check: CheckResult = OK,
) -> None:
    monkeypatch.setattr(health_service, "check_postgres", lambda: postgres)
    monkeypatch.setattr(health_service, "check_chroma", lambda: chroma)
    monkeypatch.setattr(health_service, "check_embeddings", lambda: embeddings_check)


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
            "embeddings": {"ok": True, "detail": None},
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
    monkeypatch.setattr(health_service, "check_embeddings", lambda: CheckResult(ok=True))
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


# --- Embedding service readiness (a real API call, so cached) --------------------------------


def test_embeddings_check_reports_a_missing_key_without_calling_the_api() -> None:
    result = health_service.check_embeddings()
    assert result == CheckResult(ok=False, detail="GOOGLE_API_KEY not set")


def test_embeddings_check_passes_and_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(embeddings, "probe", lambda: calls.append(1) or "ok")
    assert health_service.check_embeddings() == CheckResult(ok=True)
    assert health_service.check_embeddings() == CheckResult(ok=True)
    assert len(calls) == 1, "a probe costs an API request, so a good result is cached"


def test_embeddings_check_treats_rate_limiting_as_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings, "probe", lambda: "rate_limited")
    assert health_service.check_embeddings() == CheckResult(ok=True, detail="rate limited")


def test_embeddings_check_reports_a_rejected_key_by_class_name_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected() -> str:
        raise embeddings.EmbeddingAuthError(
            "The embedding service rejected the API key. Check GOOGLE_API_KEY."
        )

    monkeypatch.setattr(embeddings, "probe", rejected)
    assert health_service.check_embeddings() == CheckResult(ok=False, detail="EmbeddingAuthError")


def test_failed_embedding_checks_are_retried_sooner_than_good_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"fail": True, "calls": 0}

    def probe() -> str:
        state["calls"] += 1
        if state["fail"]:
            raise embeddings.EmbeddingError("down")
        return "ok"

    now = [1000.0]
    monkeypatch.setattr(health_service.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(embeddings, "probe", probe)
    assert health_service.check_embeddings().ok is False
    state["fail"] = False
    now[0] += (
        health_service.EMBEDDING_FAILURE_TTL_SECONDS + 1
    )  # long before the 5-minute success TTL
    assert health_service.check_embeddings().ok is True and state["calls"] == 2


def test_openapi_is_served_under_api_prefix(client: TestClient) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/health" in paths and "/api/ready" in paths
