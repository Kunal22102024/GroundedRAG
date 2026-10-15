"""Question suggestions: questions a document answers, questions others asked, follow-ups after an
answer (real Postgres and Chroma, fake embeddings, scripted model).

What matters: a suggestion always names a passage that answers it; one person's question is never
shown to someone else; a suggestion never reveals a document the reader may not see; follow-ups
are an extra that can fail without costing the answer.
"""

# The fixtures are imported from test_chat and used by name as test arguments.
# ruff: noqa: F401, F811

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DocumentInsight, Message, QueryEvent, User
from app.services import analysis_service
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, ask, client, library, of, user

LEAVE_Q = "How many days of paid annual leave do full-time employees receive?"


@pytest.fixture(autouse=True)
def insights_thread() -> Iterator[None]:
    """Document questions are written on the insights thread; stop it after each test."""
    yield
    analysis_service.shutdown()


def suggestions(client: TestClient, headers: Headers, *docs: dict) -> dict[str, Any]:
    params = {"document_ids": ",".join(d["id"] for d in docs)} if docs else {}
    response = client.get("/api/suggestions", headers=headers, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def settled(
    client: TestClient, headers: Headers, *docs: dict, timeout: float = 10.0
) -> dict[str, Any]:
    """Ask until nothing is pending any more (the questions are written in the background)."""
    end = time.time() + timeout
    while time.time() < end:
        result = suggestions(client, headers, *docs)
        if not result["pending"]:
            return result
        time.sleep(0.05)
    raise AssertionError("suggested questions never arrived")


def questions_reply(fake: FakeChatModel, *items: tuple[str, int]) -> None:
    fake.reply(json.dumps({"questions": [{"question": q, "source": s} for q, s in items]}))


# --- Questions a document answers ------------------------------------------------------------------


def test_a_documents_questions_are_written_once_saved_and_reused(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    questions_reply(
        fake_llm,
        ("How many days of annual leave do full-time employees get?", 1),
        ("What does the policy say about the weather on Mars?", 9),  # no such source: dropped
        ("not a question", 1),  # not shaped like a question: dropped
        ("How many days of annual leave do full-time employees get?", 1),  # a repeat: dropped
    )
    first = suggestions(client, admin, library["leave"])
    assert first["pending"] == [library["leave"]["id"]] and first["documents"] == []

    result = settled(client, admin, library["leave"])
    assert result["documents"] == [
        {
            "question": "How many days of annual leave do full-time employees get?",
            "document_id": library["leave"]["id"],
            "document_name": "leave.md",
        }
    ]
    assert len(fake_llm.calls) == 1
    again = suggestions(client, admin, library["leave"])
    assert again["documents"] == result["documents"] and len(fake_llm.calls) == 1, (
        "saved: no new call"
    )

    db.expire_all()
    row = db.scalars(select(DocumentInsight).where(DocumentInsight.kind == "questions")).one()
    assert row.content["citations"][0]["document_id"] == library["leave"]["id"]
    event = db.scalars(select(QueryEvent).where(QueryEvent.kind == "suggest")).one()
    assert event.outcome == "answered" and event.total_tokens == 132


def test_questions_of_several_documents_take_turns(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    questions_reply(
        fake_llm, ("How long is annual leave for staff?", 1), ("Who approves annual leave?", 1)
    )
    questions_reply(fake_llm, ("What does ERR_404_NET mean at the gateway?", 1))
    result = settled(client, admin, library["leave"], library["gateway"])
    assert [q["document_name"] for q in result["documents"]][:2] in (
        ["leave.md", "gateway.md"],
        ["gateway.md", "leave.md"],
    )
    assert len(result["documents"]) == 3


def test_without_a_selection_only_saved_questions_are_offered_and_nothing_is_generated(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    assert suggestions(client, admin) == {"documents": [], "popular": [], "pending": []}
    assert fake_llm.calls == []
    questions_reply(fake_llm, ("What is the L5 salary band in euros?", 1))
    settled(client, admin, library["salary"])
    assert [q["document_name"] for q in suggestions(client, admin)["documents"]] == ["salary.md"]
    assert suggestions(client, user)["documents"] == [], (
        "a Restricted document's questions stay hidden"
    )


def test_a_failed_set_is_not_retried_at_once_and_does_not_block_the_page(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.block()
    assert settled(client, admin, library["leave"])["documents"] == []
    assert (
        suggestions(client, admin, library["leave"])["pending"] == [] and len(fake_llm.calls) == 1
    )


def test_a_document_the_reader_may_not_see_is_not_found(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    response = client.get(
        "/api/suggestions", headers=user, params={"document_ids": library["salary"]["id"]}
    )
    assert response.status_code == 404 and fake_llm.calls == []
    assert (
        client.get("/api/suggestions", headers=user, params={"document_ids": "nope"}).status_code
        == 422
    )
    assert client.get("/api/suggestions").status_code == 401


def test_deleting_a_document_removes_its_questions(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    questions_reply(fake_llm, ("How many days of annual leave do staff get?", 1))
    settled(client, admin, library["leave"])
    assert (
        client.delete(f"/api/documents/{library['leave']['id']}", headers=admin).status_code == 204
    )
    db.expire_all()
    assert (
        db.scalars(select(DocumentInsight).where(DocumentInsight.kind == "questions")).all() == []
    )


# --- Questions other people asked -----------------------------------------------------------------------


def seed_question(db: Session, email: str, question: str, cited: list[str], **fields: Any) -> None:
    user_id = db.scalar(select(User.id).where(User.email == email))
    row = {
        "kind": "chat", "outcome": "answered", "model": "m", "user_id": user_id,
        "question": question, "cited": len(cited), "cited_documents": len(cited),
        "cited_document_ids": cited, **fields,
    }  # fmt: skip
    db.add(QueryEvent(**row))
    db.commit()


def test_a_question_is_offered_to_others_only_once_two_people_asked_it(
    client: TestClient, admin: Headers, user: Headers, library: dict, db: Session
) -> None:
    leave = [library["leave"]["id"]]
    seed_question(db, "boss@example.com", "How long is annual leave?", leave)
    assert suggestions(client, user)["popular"] == [], "one person's question stays theirs"

    seed_question(db, "worker@example.com", "how long is  annual leave", leave)  # same, other words
    popular = suggestions(client, user)["popular"]
    assert len(popular) == 1 and popular[0]["asked_by"] == 2
    assert popular[0]["question"].lower().startswith("how long is")
    assert suggestions(client, user, library["leave"])["popular"] == popular
    assert suggestions(client, user, library["gateway"])["popular"] == [], "not about this document"


def test_a_question_citing_a_document_the_reader_may_not_see_is_never_offered(
    client: TestClient, admin: Headers, user: Headers, library: dict, db: Session
) -> None:
    both = [library["leave"]["id"], library["salary"]["id"]]
    for email in ("boss@example.com", "worker@example.com"):
        seed_question(db, email, "Leave and salary for grade L5?", both)
    assert suggestions(client, user)["popular"] == []
    assert suggestions(client, admin)["popular"][0]["asked_by"] == 2

    assert (
        client.delete(f"/api/documents/{library['salary']['id']}", headers=admin).status_code == 204
    )
    assert suggestions(client, admin)["popular"] == [], (
        "a deleted document takes the question with it"
    )


def test_failed_refused_and_flagged_questions_are_never_offered(
    client: TestClient, admin: Headers, user: Headers, library: dict, db: Session
) -> None:
    leave = [library["leave"]["id"]]
    variants: list[dict[str, Any]] = [
        {"outcome": "failed"},
        {"outcome": "refused"},
        {"citations_removed": 1},
        {"uncited": True},
        {"partial": True},
    ]
    for i, extra in enumerate(variants):
        for email in ("boss@example.com", "worker@example.com"):
            seed_question(db, email, f"Bad question {i}?", leave, **extra)
    assert suggestions(client, admin)["popular"] == []


# --- Follow-ups after an answer -----------------------------------------------------------------------------


@pytest.fixture
def followups_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "followup_suggestions", True)


def followup_reply(fake: FakeChatModel, *items: tuple[str, str, int]) -> None:
    fake.reply(
        json.dumps({"suggestions": [{"type": t, "question": q, "source": s} for t, q, s in items]})
    )


def test_an_answer_comes_with_checked_follow_up_questions_that_are_saved(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    followups_on: None,
) -> None:
    fake_llm.reply("Full-time staff get twenty five days [1].")
    followup_reply(
        fake_llm,
        ("follow_up", "Can unused annual leave days be carried over?", 1),
        ("related", "What happens when the gateway returns ERR_404_NET?", 2),
        ("sub_question", "Who counts as a full-time employee?", 1),
        ("related", "Is there a bonus for cyclists?", 42),  # names no passage: dropped
        ("musing", "Why is leave important?", 1),  # unknown type: dropped
        ("follow_up", LEAVE_Q, 1),  # the question itself: dropped
    )
    events = ask(client, admin, LEAVE_Q)
    assert "suggesting" in [d.get("stage") for n, d in events if n == "status"]
    (offered,) = of(events, "suggestions")
    assert [(s["type"], s["question"]) for s in offered["suggestions"]] == [
        ("follow_up", "Can unused annual leave days be carried over?"),
        ("related", "What happens when the gateway returns ERR_404_NET?"),
        ("sub_question", "Who counts as a full-time employee?"),
    ]
    assert [n for n, _ in events][-2:] == ["suggestions", "done"]
    assert of(events, "done")[0]["answer"] == "Full-time staff get twenty five days [1]."

    stored = db.scalars(select(Message).where(Message.role == "assistant")).one()
    assert len(stored.suggestions) == 3 and all(s["chunk_id"] for s in stored.suggestions)
    cid = of(events, "done")[0]["conversation_id"]
    reopened = client.get(f"/api/conversations/{cid}", headers=admin).json()["messages"][1]
    assert [s["question"] for s in reopened["suggestions"]] == [
        s["question"] for s in offered["suggestions"]
    ]
    assert "chunk_id" not in reopened["suggestions"][0]

    event = db.scalars(select(QueryEvent)).one()
    assert event.suggestions == 3 and event.total_tokens == 264, "the extra call is counted"


def test_a_refusal_never_pays_for_follow_ups(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, followups_on: None
) -> None:
    ask(client, admin, "Who won the football world cup in Brazil?")
    assert fake_llm.calls == []
    fake_llm.reply("Twenty five days [1].")
    fake_llm.reply("{}")
    events = ask(client, admin, LEAVE_Q)
    assert of(events, "suggestions") == [] and len(fake_llm.calls) == 2


def test_follow_ups_can_be_switched_off(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Twenty five days [1].")
    events = ask(client, admin, LEAVE_Q)
    assert of(events, "suggestions") == [] and len(fake_llm.calls) == 1


def test_failing_follow_ups_never_cost_the_answer(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, followups_on: None
) -> None:
    fake_llm.reply("Twenty five days [1].").reply("not json").reply("still not json")
    events = ask(client, admin, LEAVE_Q)
    assert of(events, "error") == [] and of(events, "suggestions") == []
    assert of(events, "done")[0]["answer"] == "Twenty five days [1]."


def test_follow_ups_from_a_deleted_document_disappear(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    followups_on: None,
) -> None:
    fake_llm.reply("Twenty five days [1].")
    followup_reply(
        fake_llm,
        ("follow_up", "Can unused annual leave days be carried over?", 1),
        ("related", "What happens when the gateway returns ERR_404_NET?", 2),
    )
    events = ask(client, admin, LEAVE_Q)
    offered = of(events, "suggestions")[0]["suggestions"]
    gateway = [s for s in offered if s["document_id"] == library["gateway"]["id"]]
    assert gateway, "the fake retrieval put the gateway runbook second"
    assert (
        client.delete(f"/api/documents/{library['gateway']['id']}", headers=admin).status_code
        == 204
    )

    db.expire_all()
    stored = db.scalars(select(Message).where(Message.role == "assistant")).one()
    assert all(s["document_id"] != library["gateway"]["id"] for s in stored.suggestions or [])
    cid = of(events, "done")[0]["conversation_id"]
    reopened = client.get(f"/api/conversations/{cid}", headers=admin).json()["messages"][1]
    assert all("ERR_404_NET" not in s["question"] for s in reopened["suggestions"] or [])


def test_a_suggestion_can_be_asked_and_is_recorded_as_one(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Twenty five days [1].")
    ask(client, admin, LEAVE_Q, source="suggestion")
    assert db.scalars(select(QueryEvent.source)).one() == "suggestion"
