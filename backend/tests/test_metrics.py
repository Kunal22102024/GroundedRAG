"""The evaluation dashboard: one measurement per chat turn or analysis run, and the figures built
from them (real Postgres and Chroma, fake embeddings, scripted model).

What matters: every turn is measured, failures included; a broken measurement never breaks the
answer; the figures mean what the dashboard says; only admins can read them.
"""

# The fixtures are imported from test_chat and used by name as test arguments.
# ruff: noqa: F401, F811

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import REFUSAL_MESSAGE
from app.db.models import QueryEvent, User
from app.services import metrics_service
from tests.conftest import FakeChatModel
from tests.test_analysis_api import compare, done, summary
from tests.test_chat import Headers, admin, ask, client, library, of, user

LEAVE_Q = "How many days of paid annual leave do full-time employees receive?"
UNAVAILABLE = RuntimeError("503 UNAVAILABLE. The model is overloaded.")


def events(db: Session) -> list[QueryEvent]:
    db.expire_all()
    return list(db.scalars(select(QueryEvent).order_by(QueryEvent.created_at)))


def seed(db: Session, **fields: Any) -> None:
    row = {"kind": "chat", "outcome": "answered", "model": "m", **fields}
    db.add(QueryEvent(**row))
    db.commit()


# --- One measurement per chat turn ----------------------------------------------------------------


def test_an_answered_turn_is_measured_with_relevance_citations_tokens_and_timings(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Full-time staff get twenty five days [1].")
    finished = of(ask(client, admin, LEAVE_Q), "done")[0]
    (event,) = events(db)
    assert event.kind == "chat" and event.outcome == "answered" and event.source == "typed"
    assert event.question == LEAVE_Q
    assert event.guard_reason == "relevant"
    assert event.best_similarity == pytest.approx(finished["guard"]["score"], abs=1e-3)
    assert event.mean_similarity is not None and event.retrieved >= 1
    assert event.cited == 1 and event.cited_documents == 1
    assert event.cited_document_ids == [library["leave"]["id"]]
    assert event.input_tokens == 120 and event.output_tokens == 12
    assert str(event.message_id) == finished["message_id"]
    assert str(event.conversation_id) == finished["conversation_id"]
    assert event.total_ms and event.timings["embed_ms"] >= 0
    assert event.model == get_settings().llm_model


def test_a_guard_refusal_is_measured_without_tokens(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    ask(client, admin, "Who won the football world cup in Brazil?")
    (event,) = events(db)
    assert event.outcome == "refused" and event.guard_reason == "below_threshold"
    assert event.total_tokens == 0 and fake_llm.calls == []


def test_a_model_refusal_is_a_refusal_that_passed_the_guard(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply(REFUSAL_MESSAGE)
    ask(client, admin, LEAVE_Q)
    (event,) = events(db)
    assert event.outcome == "refused" and event.guard_reason == "relevant"
    assert event.input_tokens == 120


def test_a_failed_turn_is_measured_with_its_error_code_and_question(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    for _ in range(5):
        fake_llm.fail(UNAVAILABLE)
    assert of(ask(client, admin, LEAVE_Q), "error")[0]["code"] == "unavailable"
    (event,) = events(db)
    assert event.outcome == "failed" and event.error_code == "unavailable"
    assert event.question == LEAVE_Q and event.message_id is None
    assert event.guard_reason == "relevant", "what the turn got to before failing is kept"


def test_repaired_citations_are_measured(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Twenty five days [7].").reply("Twenty five days [1] and more [9].")
    ask(client, admin, LEAVE_Q)
    (event,) = events(db)
    assert event.retried and event.citations_removed == 1 and event.cited == 1


def test_how_a_question_was_asked_is_recorded(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Twenty five days [1].")
    ask(client, admin, LEAVE_Q, source="followup")
    assert events(db)[0].source == "followup"
    with client.stream(
        "POST", "/api/chat", headers=admin, json={"question": "x?", "source": "bot"}
    ) as r:
        assert r.status_code == 422


def test_a_broken_measurement_never_costs_the_answer(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database gone")

    monkeypatch.setattr(metrics_service, "get_sessionmaker", broken)
    fake_llm.reply("Twenty five days [1].")
    result = ask(client, admin, LEAVE_Q)
    assert of(result, "done")[0]["answer"] == "Twenty five days [1]." and not of(result, "error")
    assert events(db) == []


# --- Analysis runs ------------------------------------------------------------------------------------


def test_summaries_comparisons_and_cache_hits_are_measured(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Leave is twenty five days [1].")
    done(summary(client, admin, library["leave"], "executive"))
    done(summary(client, admin, library["leave"], "executive"))  # saved: no model call
    fake_llm.reply("Leave [1].").reply("Gateway [1].")
    compare(client, admin, LEAVE_Q, library["leave"], library["gateway"])

    first, cached, comparison = events(db)
    assert (first.kind, first.outcome, first.total_tokens) == ("summary", "answered", 132)
    assert first.document_ids == [library["leave"]["id"]]
    assert cached.timings.get("cached") is True and cached.total_tokens == 0
    assert comparison.kind == "compare" and comparison.question == LEAVE_Q
    assert set(comparison.document_ids) == {library["leave"]["id"], library["gateway"]["id"]}
    assert comparison.cited >= 1 and comparison.total_tokens >= 132


def test_a_failed_summary_is_measured_as_failed(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.block()
    summary(client, admin, library["leave"], "faq")
    (event,) = events(db)
    assert event.kind == "summary" and event.outcome == "failed" and event.error_code == "blocked"


# --- The figures ----------------------------------------------------------------------------------------


def test_the_summary_figures_mean_what_they_say(
    client: TestClient, admin: Headers, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "llm_input_usd_per_mtok", 1.0)
    monkeypatch.setattr(get_settings(), "llm_output_usd_per_mtok", 4.0)
    for i in range(6):  # answered, 2 citations each, one of them with a removed marker
        seed(
            db, guard_reason="relevant", best_similarity=0.70 + i / 100, mean_similarity=0.5,
            cited=2, cited_documents=1, citations_removed=1 if i == 0 else 0,
            input_tokens=1000, output_tokens=100, total_tokens=1100, total_ms=1000.0 * (i + 1),
            timings={"embed_ms": 100.0, "generate_ms": 500.0},
        )  # fmt: skip
    seed(
        db, outcome="refused", guard_reason="below_threshold", best_similarity=0.31, total_ms=300.0
    )
    seed(db, outcome="refused", guard_reason="relevant", best_similarity=0.66, total_ms=900.0)
    seed(db, outcome="failed", error_code="unavailable", total_ms=10000.0)
    seed(db, kind="summary", input_tokens=5000, output_tokens=500, total_tokens=5500)

    figures = client.get("/api/metrics/summary?days=7", headers=admin).json()
    totals = figures["totals"]
    assert (totals["queries"], totals["answered"], totals["refused"], totals["failed"]) == (
        10,
        7,
        2,
        1,
    )
    assert totals["failure_rate"] == 0.1
    assert figures["abstention"] == {"guard_refusals": 1, "model_refusals": 1, "rate": 0.25}

    grounded = figures["groundedness"]
    assert grounded["answers"] == 6 and grounded["with_citations_rate"] == 1.0
    assert grounded["citation_precision"] == round(12 / 13, 4)
    assert grounded["invalid_citations_removed"] == 1

    relevance = figures["relevance"]
    assert relevance["guard_reasons"] == {"relevant": 7, "below_threshold": 1}
    assert relevance["best_similarity_refused_mean"] == pytest.approx((0.31 + 0.66) / 2, abs=1e-4)
    bins = {b["from"]: b["count"] for b in relevance["histogram"]}
    assert bins[0.3] == 1 and bins[0.65] == 1 and bins[0.7] == 5 and bins[0.75] == 1
    assert sum(bins.values()) == 8

    latency = figures["latency"]
    assert latency["max_ms"] == 6000.0, "failed turns are not part of latency"
    assert {s["stage"] for s in latency["stages"]} == {"embed", "generate"}

    cost = figures["cost"]
    assert (cost["input_tokens"], cost["output_tokens"]) == (11000, 1100)
    assert cost["usd"] == pytest.approx(11000 / 1e6 * 1 + 1100 / 1e6 * 4, abs=1e-4)
    assert {k["kind"] for k in cost["by_kind"]} == {"chat", "summary"}
    assert figures["daily"][-1]["queries"] == 10

    chat_only = client.get("/api/metrics/summary?days=7&kind=chat", headers=admin).json()
    assert chat_only["totals"]["queries"] == 9 and chat_only["cost"]["input_tokens"] == 6000


def test_cost_is_left_out_until_prices_are_set(
    client: TestClient, admin: Headers, db: Session
) -> None:
    seed(db, input_tokens=10, output_tokens=1, total_tokens=11)
    cost = client.get("/api/metrics/summary", headers=admin).json()["cost"]
    assert cost["usd"] is None and cost["priced"] is False and cost["input_tokens"] == 10


def test_problem_queries_are_listed_and_paged(
    client: TestClient, admin: Headers, db: Session
) -> None:
    now = datetime.now(UTC)
    for i in range(3):
        seed(db, outcome="failed", error_code="unavailable", question=f"q{i}?",
             created_at=now - timedelta(minutes=i))  # fmt: skip
    seed(db, outcome="answered", uncited=True, question="flagged?")
    seed(db, outcome="refused", guard_reason="below_threshold", question="off topic?")

    failed = client.get("/api/metrics/events?outcome=failed&limit=2", headers=admin).json()
    assert [e["question"] for e in failed] == ["q0?", "q1?"]
    older = client.get(
        "/api/metrics/events", headers=admin,
        params={"outcome": "failed", "before": failed[-1]["created_at"]},
    ).json()  # fmt: skip
    assert [e["question"] for e in older] == ["q2?"]
    assert [
        e["question"]
        for e in client.get("/api/metrics/events?outcome=flagged", headers=admin).json()
    ] == ["flagged?"]
    refused = client.get("/api/metrics/events?outcome=refused", headers=admin).json()
    assert refused[0]["guard_reason"] == "below_threshold"


def test_only_admins_see_the_dashboard(client: TestClient, admin: Headers, user: Headers) -> None:
    for path in ("/api/metrics/summary", "/api/metrics/events"):
        assert client.get(path, headers=user).status_code == 403
        assert client.get(path).status_code == 401
        assert client.get(path, headers=admin).status_code == 200


def test_old_measurements_are_pruned(client: TestClient, admin: Headers, db: Session) -> None:
    seed(db, question="old?", created_at=datetime.now(UTC) - timedelta(days=200))
    seed(db, question="new?")
    assert metrics_service.prune(db) == 1
    assert [e.question for e in events(db)] == ["new?"]


def test_deleting_a_user_keeps_their_measurements(
    client: TestClient,
    admin: Headers,
    user: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    fake_llm.reply("Twenty five days [1].")
    ask(client, user, LEAVE_Q)
    db.execute(delete(User).where(User.email == "worker@example.com"))
    db.commit()
    (event,) = events(db)
    assert event.user_id is None and event.question == LEAVE_Q
