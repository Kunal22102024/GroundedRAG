"""Summaries, extraction, comparison and conflict detection through the API (real Postgres and Chroma,
fake embeddings, scripted model: no network, no key).

What matters: every result is cited from the document it is about, a saved result costs no second
model call, nobody analyses a document above their level, a deleted document's text leaves every
saved result, and a failed extra (the conflict check) never costs the user their answer.
"""

# The fixtures below are imported from test_chat and used by name as test arguments, which is how
# pytest shares them between modules; ruff reads that as an unused import that is then redefined.
# ruff: noqa: F401, F811

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import REFUSAL_MESSAGE
from app.db.models import Document, DocumentInsight, Message, User
from app.services import analysis_service, ingestion
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, ask, client, library, of, user
from tests.test_documents_api import one

LEAVE_Q = "How many days of paid annual leave do full-time employees receive?"
HANDBOOK = b"# Handbook Leave\n\nEmployees receive thirty days of paid annual leave each year.\n"


# --- helpers ---------------------------------------------------------------------------------------


def post_stream(
    client: TestClient, headers: Headers, path: str, body: dict[str, Any]
) -> list[tuple[str, dict]]:
    with client.stream("POST", path, headers=headers, json=body) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        name, data = "", {}
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events


def done(events: list[tuple[str, dict]]) -> dict[str, Any]:
    assert events[-1][0] == "done", events[-1]
    return events[-1][1]


def summary(
    client: TestClient, headers: Headers, doc: dict, kind: str, **extra: Any
) -> list[tuple[str, dict]]:
    return post_stream(
        client, headers, "/api/analysis/summary", {"document_id": doc["id"], "kind": kind, **extra}
    )


def json_reply(fake: FakeChatModel, payload: dict) -> None:
    fake.reply(json.dumps(payload))


def insight_count(db: Session) -> int:
    db.expire_all()
    return int(db.scalar(select(func.count()).select_from(DocumentInsight)) or 0)


@pytest.fixture
def handbook(client: TestClient, admin: Headers) -> dict[str, Any]:
    """A second document that contradicts the leave policy (30 days, not 25)."""
    return one(client, admin, "handbook.md", HANDBOOK, department="HR", access_level="Internal")


# --- Summaries ----------------------------------------------------------------------------------------


def test_an_executive_summary_is_cited_from_its_own_document_and_saved(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Full-time staff get twenty five days of annual leave [1].")
    events = summary(client, admin, library["leave"], "executive")
    result = done(events)
    assert result["kind"] == "executive" and result["cached"] is False
    content = result["content"]
    assert content["text"] == "Full-time staff get twenty five days of annual leave [1]."
    (citation,) = content["citations"]
    assert citation["document_name"] == "leave.md" and "twenty five days" in citation["text"]
    assert citation["document_id"] == library["leave"]["id"]
    assert events[0][0] == "status"
    assert insight_count(db) == 1 and len(fake_llm.calls) == 1


def test_a_saved_summary_costs_no_second_model_call_until_refreshed(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("First version [1].").reply("Second version [1].")
    first = done(summary(client, admin, library["leave"], "executive"))
    again = done(summary(client, admin, library["leave"], "executive"))
    assert again["cached"] is True and again["content"] == first["content"]
    assert len(fake_llm.calls) == 1

    fresh = done(summary(client, admin, library["leave"], "executive", refresh=True))
    assert fresh["cached"] is False and fresh["content"]["text"] == "Second version [1]."
    assert len(fake_llm.calls) == 2
    assert (
        done(summary(client, admin, library["leave"], "executive"))["content"]["text"]
        == "Second version [1]."
    )


@pytest.mark.parametrize(
    ("kind", "reply", "expected"),
    [
        ("chapters", {"items": [{"title": "Annual Leave", "summary": "25 days [1]."}]}, "summary"),
        (
            "faq",
            {"items": [{"question": "How many days?", "answer": "Twenty five [1]."}]},
            "answer",
        ),
        (
            "glossary",
            {"items": [{"term": "Annual leave", "definition": "Paid time off [1]."}]},
            "definition",
        ),
        (
            "key_changes",
            {"items": [{"change": "Leave is 25 days [1].", "impact": "More rest [1]."}]},
            "change",
        ),
    ],
)
def test_every_summary_kind_returns_cited_items(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    kind: str,
    reply: dict,
    expected: str,
) -> None:
    json_reply(fake_llm, reply)
    content = done(summary(client, admin, library["leave"], kind))["content"]
    assert (
        content["kind"] == kind
        and len(content["items"]) == 1
        and "[1]" in content["items"][0][expected]
    )
    assert content["citations"][0]["document_id"] == library["leave"]["id"]


def test_different_kinds_are_saved_separately_and_listed(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Executive [1].")
    json_reply(fake_llm, {"items": [{"question": "Q?", "answer": "A [1]."}]})
    done(summary(client, admin, library["leave"], "executive"))
    done(summary(client, admin, library["leave"], "faq"))
    rows = client.get(f"/api/documents/{library['leave']['id']}/insights", headers=admin).json()
    assert [(r["kind"], r["status"]) for r in rows] == [("executive", "ready"), ("faq", "ready")]
    assert rows[1]["content"]["items"][0]["question"] == "Q?"
    assert (
        client.get(f"/api/documents/{library['gateway']['id']}/insights", headers=admin).json()
        == []
    )


def test_a_failed_summary_is_reported_saved_as_failed_and_can_be_retried(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("no citation").reply("still no citation")
    events = summary(client, admin, library["leave"], "executive")
    assert events[-1][0] == "error" and events[-1][1]["code"] == "failed"
    rows = client.get(f"/api/documents/{library['leave']['id']}/insights", headers=admin).json()
    assert rows[0]["status"] == "failed" and rows[0]["content"] is None and rows[0]["error_message"]

    fake_llm.reply("Now it works [1].")
    assert (
        done(summary(client, admin, library["leave"], "executive"))["content"]["text"]
        == "Now it works [1]."
    )


def test_a_blocked_reply_ends_with_an_error_event_not_a_broken_stream(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.block()
    events = summary(client, admin, library["leave"], "faq")
    assert events[-1][0] == "error" and events[-1][1]["code"] == "blocked"


UNAVAILABLE = RuntimeError("503 UNAVAILABLE. The model is overloaded.")


def test_an_analysis_outlasts_more_transient_errors_than_a_chat_turn_would(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    settings = get_settings()
    assert settings.analysis_max_retries > settings.interactive_max_retries
    for _ in range(settings.interactive_max_retries + 1):  # one more than chat survives
        fake_llm.fail(UNAVAILABLE)
    fake_llm.reply("It works after all [1].")
    events = summary(client, admin, library["leave"], "executive")
    assert done(events)["content"]["text"] == "It works after all [1]."
    assert len(fake_llm.slept) == settings.interactive_max_retries + 1
    assert [d["stage"] for n, d in events if n == "status"].count("rate_limited") == len(
        fake_llm.slept
    )


def test_an_analysis_that_never_gets_through_ends_with_a_friendly_error_after_its_own_limit(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    settings = get_settings()
    for _ in range(settings.analysis_max_retries + 3):
        fake_llm.fail(UNAVAILABLE)
    events = summary(client, admin, library["leave"], "faq")
    assert events[-1][0] == "error" and events[-1][1]["code"] == "unavailable"
    assert len(fake_llm.calls) == 1 + settings.analysis_max_retries


# --- Access ---------------------------------------------------------------------------------------------


def test_a_document_above_the_users_level_does_not_exist_for_analysis(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    salary = {"document_id": library["salary"]["id"]}
    for path, body in [
        ("/api/analysis/summary", {**salary, "kind": "executive"}),
        ("/api/analysis/extract", {**salary, "fields": ["dates"]}),
    ]:
        assert client.post(path, headers=user, json=body).status_code == 404
    assert (
        client.get(f"/api/documents/{library['salary']['id']}/insights", headers=user).status_code
        == 404
    )
    compare = {
        "question": "salary?",
        "document_ids": [library["salary"]["id"], library["leave"]["id"]],
    }
    assert client.post("/api/analysis/compare", headers=user, json=compare).status_code == 404
    assert fake_llm.calls == []


def test_unknown_and_not_yet_indexed_documents_are_refused_before_any_model_call(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    ghost = {"document_id": str(uuid.uuid4()), "kind": "executive"}
    assert client.post("/api/analysis/summary", headers=admin, json=ghost).status_code == 404

    queued = Document(
        name="queued.pdf", stored_path="q.pdf", mime_type="application/pdf", size_bytes=1,
        status="queued", chunk_count=0, department="", document_type="",
        creation_date=date(2026, 1, 1), access_level="Internal",
    )  # fmt: skip
    db.add(queued)
    db.commit()
    response = client.post(
        "/api/analysis/summary", headers=admin, json={"document_id": str(queued.id), "kind": "faq"}
    )
    assert response.status_code == 409 and "not indexed" in response.json()["detail"]
    assert fake_llm.calls == []


def test_requests_are_validated(client: TestClient, admin: Headers, library: dict) -> None:
    leave = library["leave"]["id"]
    bad = [
        {"document_id": leave, "kind": "poem"},
        {"document_id": leave, "kind": "faq", "compare_to": library["gateway"]["id"]},
        {"document_id": leave, "kind": "key_changes", "compare_to": leave},
        {"document_id": leave, "kind": "faq", "extra": 1},
        {"kind": "faq"},
    ]
    for body in bad:
        assert client.post("/api/analysis/summary", headers=admin, json=body).status_code == 422, (
            body
        )


# --- Extraction --------------------------------------------------------------------------------------------


def test_extraction_returns_cited_values_for_presets_and_custom_fields_and_is_saved(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    json_reply(
        fake_llm,
        {
            "fields": [
                {
                    "field": "Amounts",
                    "values": [{"value": "25 days", "source": 1, "quote": "twenty five days"}],
                },
                {"field": "Governing law", "values": []},
            ]
        },
    )
    body = {"document_id": library["leave"]["id"], "fields": ["amounts", "Governing law"]}
    result = done(post_stream(client, admin, "/api/analysis/extract", body))
    content = result["content"]
    assert content["kind"] == "extraction" and result["cached"] is False
    assert content["fields"][0] == {
        "field": "Amounts",
        "values": [{"value": "25 days", "n": 1, "quote": "twenty five days"}],
    }
    assert content["fields"][1] == {"field": "Governing law", "values": []}
    assert content["citations"][0]["document_name"] == "leave.md"
    task = str(fake_llm.calls[0][-1].content)
    assert "- Amounts: money" in task and "- Governing law" in task

    # The same fields in another order and case are the same extraction: no second call.
    again = done(
        post_stream(
            client, admin, "/api/analysis/extract", {**body, "fields": ["GOVERNING LAW", "Amounts"]}
        )
    )
    assert again["cached"] is True and len(fake_llm.calls) == 1 and insight_count(db) == 1

    # Different fields are a different extraction.
    json_reply(fake_llm, {"fields": []})
    other = done(post_stream(client, admin, "/api/analysis/extract", {**body, "fields": ["dates"]}))
    assert other["cached"] is False and len(fake_llm.calls) == 2 and insight_count(db) == 2


def test_extraction_field_rules_are_enforced(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "extraction_max_fields", 2)
    leave = library["leave"]["id"]
    for fields in (["a", "b", "c"], ["   "], ["x" * 61]):
        response = client.post(
            "/api/analysis/extract", headers=admin, json={"document_id": leave, "fields": fields}
        )
        assert response.status_code == 422, fields
    assert (
        client.post(
            "/api/analysis/extract", headers=admin, json={"document_id": leave, "fields": []}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/analysis/extract",
            headers=admin,
            json={"document_id": leave, "fields": ["a\x00b"]},
        ).status_code
        == 422
    )
    assert fake_llm.calls == []


# --- Comparison ---------------------------------------------------------------------------------------------


def compare(
    client: TestClient, headers: Headers, question: str, *docs: dict
) -> list[tuple[str, dict]]:
    return post_stream(
        client,
        headers,
        "/api/analysis/compare",
        {"question": question, "document_ids": [d["id"] for d in docs]},
    )


def test_comparison_answers_each_document_from_its_own_text_and_reports_the_disagreement(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("The leave policy gives twenty five days [1].")
    fake_llm.reply("The handbook gives thirty days [1].")
    json_reply(
        fake_llm,
        {
            "conflicts": [
                {
                    "topic": "Annual leave allowance",
                    "claims": [
                        {"source": 1, "claim": "25 days"},
                        {"source": 2, "claim": "30 days"},
                    ],
                }
            ]
        },
    )
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    columns = {data["document_id"]: data for name, data in events if name == "column"}
    assert set(columns) == {library["leave"]["id"], handbook["id"]}
    for doc_id, column in columns.items():
        assert column["covered"] is True
        assert {c["document_id"] for c in column["citations"]} == {doc_id}, (
            "a column cites only its own document"
        )
    assert "twenty five days" in columns[library["leave"]["id"]]["citations"][0]["text"]
    assert "thirty days" in columns[handbook["id"]]["citations"][0]["text"]
    assert any(n == "token" and d["document_id"] == handbook["id"] for n, d in events), (
        "answers stream per document"
    )

    (conflicts,) = [d["conflicts"] for n, d in events if n == "conflicts"]
    assert len(conflicts) == 1 and conflicts[0]["topic"] == "Annual leave allowance"
    claims = {c["document_id"]: c for c in conflicts[0]["claims"]}
    assert (
        claims[library["leave"]["id"]]["claim"] == "25 days"
        and claims[handbook["id"]]["claim"] == "30 days"
    )
    assert {c["n"] for c in conflicts[0]["claims"]} == {1}, (
        "claim numbers are the column's own citation numbers"
    )
    assert done(events)["conflicts"] == 1 and len(fake_llm.calls) == 3
    conflict_call = str(fake_llm.calls[2][-1].content)
    assert 'document="leave.md"' in conflict_call and 'document="handbook.md"' in conflict_call


def test_a_document_that_does_not_cover_the_question_is_said_so_without_a_model_call(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("The handbook gives thirty days [1].")
    events = compare(client, admin, LEAVE_Q, library["gateway"], handbook)
    columns = {d["document_id"]: d for n, d in events if n == "column"}
    assert (
        columns[library["gateway"]["id"]]["covered"] is False
        and columns[library["gateway"]["id"]]["answer"] == ""
    )
    assert columns[handbook["id"]]["covered"] is True
    assert len(fake_llm.calls) == 1, (
        "no answer call for the uncovered document and no conflict check for one source"
    )
    assert [d["conflicts"] for n, d in events if n == "conflicts"] == [[]]


def test_a_model_that_refuses_for_one_document_is_reported_as_not_covered(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(REFUSAL_MESSAGE).reply("The handbook gives thirty days [1].")
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    columns = {d["document_id"]: d for n, d in events if n == "column"}
    assert columns[library["leave"]["id"]]["covered"] is False
    assert columns[handbook["id"]]["covered"] is True and len(fake_llm.calls) == 2


def test_a_failing_conflict_check_still_delivers_both_columns(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Leave [1].").reply("Handbook [1].").reply("not json").reply("still not json")
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    assert [d["covered"] for n, d in events if n == "column"] == [True, True]
    assert [d["conflicts"] for n, d in events if n == "conflicts"] == [[]]
    assert events[-1][0] == "done"


def test_a_comparison_column_outlasts_more_transient_errors_than_a_chat_turn_would(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
) -> None:
    settings = get_settings()
    for _ in range(settings.interactive_max_retries + 1):  # one more than chat survives
        fake_llm.fail(UNAVAILABLE)
    fake_llm.reply("Leave [1].").reply("Handbook [1].")
    json_reply(fake_llm, {"conflicts": []})
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    assert [d["covered"] for n, d in events if n == "column"] == [True, True]
    assert events[-1][0] == "done" and not [1 for n, _ in events if n == "error"]


def test_the_extra_conflict_check_gives_up_as_fast_as_chat_does(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
) -> None:
    settings = get_settings()
    fake_llm.reply("Leave [1].").reply("Handbook [1].")
    for _ in range(settings.analysis_max_retries + 3):
        fake_llm.fail(UNAVAILABLE)
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    assert [d["covered"] for n, d in events if n == "column"] == [True, True]
    assert events[-1][0] == "done"
    # Two answers, then the check with chat's patience: extra work never waits longer than chat.
    assert len(fake_llm.calls) == 2 + 1 + settings.interactive_max_retries


def test_conflict_detection_can_be_switched_off(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "conflict_detection", False)
    fake_llm.reply("Leave [1].").reply("Handbook [1].")
    events = compare(client, admin, LEAVE_Q, library["leave"], handbook)
    assert len(fake_llm.calls) == 2 and done(events)["conflicts"] == 0


def test_comparison_needs_two_to_the_configured_maximum_distinct_documents(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leave = library["leave"]
    assert (
        client.post(
            "/api/analysis/compare",
            headers=admin,
            json={"question": "q?", "document_ids": [leave["id"]]},
        ).status_code
        == 422
    )
    twice = client.post(
        "/api/analysis/compare",
        headers=admin,
        json={"question": "q?", "document_ids": [leave["id"], leave["id"]]},
    )
    assert twice.status_code == 422 and "at least two different" in twice.json()["detail"]
    monkeypatch.setattr(get_settings(), "compare_max_documents", 2)
    three = client.post(
        "/api/analysis/compare", headers=admin,
        json={"question": "q?", "document_ids": [leave["id"], handbook["id"], library["gateway"]["id"]]},
    )  # fmt: skip
    assert three.status_code == 422 and "at most 2" in three.json()["detail"]
    for bad in ("", "   ", "a\x00b"):
        assert (
            client.post(
                "/api/analysis/compare",
                headers=admin,
                json={"question": bad, "document_ids": [leave["id"], handbook["id"]]},
            ).status_code
            == 422
        )
    assert fake_llm.calls == []


# --- Deleting a document removes what was generated from it ----------------------------------------------------


def test_deleting_a_document_removes_its_saved_analyses_and_those_that_cite_it(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Leave [1].")
    json_reply(fake_llm, {"items": [{"change": "Was [2], now [1].", "impact": "x"}]})
    done(summary(client, admin, library["leave"], "executive"))
    done(
        summary(client, admin, library["gateway"], "key_changes", compare_to=library["leave"]["id"])
    )
    assert insight_count(db) == 2

    assert (
        client.delete(f"/api/documents/{library['leave']['id']}", headers=admin).status_code == 204
    )
    assert insight_count(db) == 0, "its own summary is gone, and so is the report that quoted it"
    assert (
        client.get(f"/api/documents/{library['leave']['id']}/insights", headers=admin).status_code
        == 404
    )
    assert (
        client.get(f"/api/documents/{library['gateway']['id']}/insights", headers=admin).json()
        == []
    )


def test_a_demoted_reader_no_longer_sees_passages_of_a_document_above_their_level(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    json_reply(
        fake_llm, {"items": [{"change": "Salary band appears [2]; leave is [1].", "impact": "x"}]}
    )
    done(
        summary(client, admin, library["leave"], "key_changes", compare_to=library["salary"]["id"])
    )
    rows = client.get(f"/api/documents/{library['leave']['id']}/insights", headers=admin).json()
    assert {c["n"]: c.get("unavailable", False) for c in rows[0]["content"]["citations"]} == {
        1: False,
        2: False,
    }

    db.execute(update(User).where(User.email == "boss@example.com").values(role="user"))
    db.commit()
    rows = client.get(f"/api/documents/{library['leave']['id']}/insights", headers=admin).json()
    citations = {c["n"]: c for c in rows[0]["content"]["citations"]}
    assert citations[1].get("unavailable") is not True and "twenty five" in citations[1]["text"]
    assert (
        citations[2]["unavailable"] is True
        and citations[2]["text"] == ""
        and citations[2]["document_name"] == ""
    )
    assert "SALARY_X9" not in json.dumps(rows)


# --- The executive summary after upload ---------------------------------------------------------------------------


@pytest.fixture
def auto_summary(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turn the after-upload hook on (startup is neutralized in tests) and stop it afterwards."""
    monkeypatch.setattr(ingestion, "_indexed_hooks", [])
    ingestion.on_indexed(analysis_service.submit_auto_summary)
    yield
    analysis_service.shutdown()


def wait_for(db: Session, document_id: str, status: str, timeout: float = 10.0) -> DocumentInsight:
    end = time.time() + timeout
    while time.time() < end:
        db.expire_all()
        row = db.scalar(
            select(DocumentInsight).where(DocumentInsight.document_id == uuid.UUID(document_id))
        )
        if row is not None and row.status == status:
            return row
        time.sleep(0.05)
    raise AssertionError(f"no insight reached {status!r}")


def test_a_new_document_gets_an_executive_summary_by_itself(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel, db: Session, auto_summary: Any
) -> None:
    fake_llm.reply("This note explains the retry policy [1].")
    doc = one(
        client,
        admin,
        "retry.md",
        b"# Retry Policy\n\nThe retry limit for API timeouts is three attempts.\n",
    )
    row = wait_for(db, doc["id"], "ready")
    assert (
        row.kind == "executive"
        and row.content["text"] == "This note explains the retry policy [1]."
    )
    assert row.content["citations"][0]["document_id"] == doc["id"]

    listed = client.get("/api/documents", headers=admin).json()["items"]
    assert [d["executive_summary"] for d in listed if d["id"] == doc["id"]] == ["ready"]
    # Opening it later costs nothing.
    again = done(summary(client, admin, doc, "executive"))
    assert again["cached"] is True and len(fake_llm.calls) == 1


def test_a_failed_automatic_summary_does_not_fail_the_upload(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel, db: Session, auto_summary: Any
) -> None:
    fake_llm.block()
    doc = one(
        client,
        admin,
        "retry.md",
        b"# Retry Policy\n\nThe retry limit for API timeouts is three attempts.\n",
    )
    row = wait_for(db, doc["id"], "failed")
    assert row.error_message and row.content is None
    listed = client.get(f"/api/documents/{doc['id']}", headers=admin).json()
    assert listed["status"] == "indexed", "the document itself is fine"


def test_the_automatic_summary_can_be_switched_off(
    client: TestClient,
    admin: Headers,
    fake_llm: FakeChatModel,
    db: Session,
    auto_summary: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "auto_executive_summary", False)
    one(
        client,
        admin,
        "retry.md",
        b"# Retry Policy\n\nThe retry limit for API timeouts is three attempts.\n",
    )
    time.sleep(0.3)
    assert insight_count(db) == 0 and fake_llm.calls == []


def test_a_request_for_a_summary_that_is_being_generated_waits_for_it_instead_of_paying_twice(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_service, "POLL_SECONDS", 0.1)
    leave_id = uuid.UUID(library["leave"]["id"])
    analysis_service._begin(leave_id, "executive", "")  # someone else started it
    finished = {"text": "Generated elsewhere [1].", "kind": "executive", "citations": []}
    threading.Timer(
        0.4,
        analysis_service._finish,
        args=(leave_id, "executive", ""),
        kwargs={"content": finished},
    ).start()
    result = done(summary(client, admin, library["leave"], "executive"))
    assert result["cached"] is True and result["content"]["text"] == "Generated elsewhere [1]."
    assert fake_llm.calls == []


def test_summaries_a_restart_cut_off_are_failed_and_a_dead_pending_row_is_regenerated(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    leave_id = uuid.UUID(library["leave"]["id"])
    analysis_service._begin(leave_id, "faq", "")
    assert analysis_service.recover_interrupted(db) == 1
    row = db.scalar(select(DocumentInsight))
    db.refresh(row)
    assert row.status == "failed" and "restart" in row.error_message

    # A pending row that nobody is working on any more (older than the freshness window) is rerun.
    db.execute(
        update(DocumentInsight).values(
            status="pending", updated_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 30)
        )
    )
    db.commit()
    json_reply(fake_llm, {"items": [{"question": "Q?", "answer": "A [1]."}]})
    assert done(summary(client, admin, library["leave"], "faq"))["cached"] is False


# --- Conflicts in a normal chat answer ---------------------------------------------------------------------------------


def ask_two(
    client: TestClient, headers: Headers, a: dict, b: dict, **extra: Any
) -> list[tuple[str, dict]]:
    return ask(client, headers, LEAVE_Q, filters={"document_ids": [a["id"], b["id"]]}, **extra)


def test_an_answer_citing_two_documents_that_disagree_shows_both_claims_with_citations(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    fake_llm.reply("One source says 25 days [1] while another says 30 days [2].")
    json_reply(
        fake_llm,
        {
            "conflicts": [
                {
                    "topic": "Annual leave",
                    "claims": [
                        {"source": 1, "claim": "25 days"},
                        {"source": 2, "claim": "30 days"},
                    ],
                }
            ]
        },
    )
    events = ask_two(client, admin, library["leave"], handbook)
    names = [n for n, _ in events]
    assert "checking_sources" in [d.get("stage") for n, d in events if n == "status"]
    assert names[-3:] == ["citations", "conflicts", "done"]

    (payload,) = of(events, "conflicts")
    (conflict,) = payload["conflicts"]
    assert conflict["topic"] == "Annual leave" and len(conflict["claims"]) == 2
    cited = {c["n"]: c["document_id"] for c in of(events, "citations")[0]["citations"]}
    for claim in conflict["claims"]:
        assert cited[claim["n"]] == claim["document_id"], (
            "every claim points at a real citation of the answer"
        )
    assert {c["claim"] for c in conflict["claims"]} == {"25 days", "30 days"}
    assert len(fake_llm.calls) == 2

    stored = db.scalars(select(Message.conflicts).where(Message.role == "assistant")).one()
    assert stored and stored[0]["topic"] == "Annual leave"
    cid = of(events, "done")[0]["conversation_id"]
    reopened = client.get(f"/api/conversations/{cid}", headers=admin).json()["messages"][1]
    assert reopened["conflicts"][0]["claims"][0]["claim"] in {"25 days", "30 days"}


def test_an_answer_from_one_document_never_pays_for_a_conflict_check(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Twenty five days [1].")
    events = ask(client, admin, LEAVE_Q, filters={"document_ids": [library["leave"]["id"]]})
    assert of(events, "conflicts") == [] and len(fake_llm.calls) == 1
    assert "checking_sources" not in [d.get("stage") for n, d in events if n == "status"]


def test_documents_that_agree_produce_no_conflict_event(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    fake_llm.reply("Leave is paid [1][2].")
    json_reply(fake_llm, {"conflicts": []})
    events = ask_two(client, admin, library["leave"], handbook)
    assert of(events, "conflicts") == [] and events[-1][0] == "done" and len(fake_llm.calls) == 2
    assert db.scalars(select(Message.conflicts).where(Message.role == "assistant")).one() is None


def test_a_failing_conflict_check_never_costs_the_user_the_answer(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Leave is paid [1][2].").reply("garbage").reply("more garbage")
    events = ask_two(client, admin, library["leave"], handbook)
    assert of(events, "error") == [] and of(events, "conflicts") == []
    assert of(events, "done")[0]["answer"] == "Leave is paid [1][2]."


def test_conflict_detection_in_chat_can_be_switched_off(
    client: TestClient,
    admin: Headers,
    library: dict,
    handbook: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "conflict_detection", False)
    fake_llm.reply("Leave is paid [1][2].")
    events = ask_two(client, admin, library["leave"], handbook)
    assert len(fake_llm.calls) == 1 and of(events, "conflicts") == []


def test_a_saved_conflict_loses_claims_whose_source_the_reader_may_no_longer_see(
    client: TestClient, admin: Headers, library: dict, handbook: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("One source says 25 days [1] while another says 30 days [2].")
    json_reply(
        fake_llm,
        {
            "conflicts": [
                {
                    "topic": "Annual leave",
                    "claims": [
                        {"source": 1, "claim": "25 days"},
                        {"source": 2, "claim": "30 days"},
                    ],
                }
            ]
        },
    )
    cid = of(ask_two(client, admin, library["leave"], handbook), "done")[0]["conversation_id"]
    assert client.delete(f"/api/documents/{handbook['id']}", headers=admin).status_code == 204
    message = client.get(f"/api/conversations/{cid}", headers=admin).json()["messages"][1]
    assert message["conflicts"] is None, "with one side gone there is nothing left to compare"
    assert any(c.get("unavailable") for c in message["citations"])
    assert "30 days" not in json.dumps(message["conflicts"])
