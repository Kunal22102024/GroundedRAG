"""Conversations, feedback and what a saved conversation may still show.

Also: every route except the public ones refuses a request without a valid token, and no text
field accepts a NUL character (PostgreSQL cannot store one; it used to surface as a 500).
"""

# The fixtures below are imported from test_chat and used by name as test arguments, which is how
# pytest shares them between modules; ruff reads that as an unused import that is then redefined.
# ruff: noqa: F401, F811

from __future__ import annotations

import re
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.constants import REFUSAL_MESSAGE
from app.core.filters import build_filter
from app.db.models import Feedback, Message, User
from app.main import app
from app.services import chat_service
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, ask, client, library, of, user
from tests.test_documents_api import auth

LEAVE_Q = "How many days of paid annual leave do full-time employees receive?"
SALARY_Q = "What is the grade L5 salary band in euros?"


@pytest.fixture
def other(client: TestClient, user: Headers) -> Headers:
    return auth(client, "other@example.com")


def converse(
    client: TestClient, headers: Headers, fake_llm: FakeChatModel, question: str, reply: str
) -> dict[str, Any]:
    fake_llm.reply(reply)
    return of(ask(client, headers, question), "done")[0]


def conversation(client: TestClient, headers: Headers, conversation_id: str) -> dict[str, Any]:
    response = client.get(f"/api/conversations/{conversation_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


# --- Ownership: every conversation route -----------------------------------------------------------


def test_conversations_are_listed_read_renamed_and_deleted_by_their_owner_only(
    client: TestClient,
    admin: Headers,
    user: Headers,
    library: dict,
    fake_llm: FakeChatModel,
) -> None:
    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    cid = done["conversation_id"]

    assert [c["id"] for c in client.get("/api/conversations", headers=admin).json()] == [cid]
    assert client.get("/api/conversations", headers=user).json() == []

    # Somebody else's conversation does not exist, whichever way it is addressed.
    path = f"/api/conversations/{cid}"
    assert client.get(path, headers=user).status_code == 404
    assert client.patch(path, headers=user, json={"title": "mine now"}).status_code == 404
    assert client.delete(path, headers=user).status_code == 404
    assert conversation(client, admin, cid)["title"].startswith("How many days")  # untouched

    renamed = client.patch(path, headers=admin, json={"title": "  Leave   policy  "})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Leave policy"
    assert client.delete(path, headers=admin).status_code == 204
    assert client.get(path, headers=admin).status_code == 404
    assert client.get("/api/conversations", headers=admin).json() == []


def test_a_conversation_shows_its_messages_in_order_with_citations(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    detail = conversation(client, admin, done["conversation_id"])
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    (citation,) = detail["messages"][1]["citations"]
    assert citation["document_name"] == "leave.md" and "twenty five days" in citation["text"]
    assert not citation.get("unavailable")


def test_rename_validates_the_title(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    cid = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")["conversation_id"]
    path = f"/api/conversations/{cid}"
    for bad in ({"title": ""}, {"title": "x" * 201}, {"title": "a\x00b"}, {"nope": 1}):
        assert client.patch(path, headers=admin, json=bad).status_code == 422


# --- Feedback ----------------------------------------------------------------------------------------


def test_a_rating_belongs_to_one_user_and_can_be_changed(
    client: TestClient,
    admin: Headers,
    user: Headers,
    other: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    message_id, cid = done["message_id"], done["conversation_id"]
    url = f"/api/messages/{message_id}/feedback"

    assert client.post(url, headers=admin, json={"rating": "up"}).status_code == 204
    assert conversation(client, admin, cid)["messages"][1]["feedback"] == "up"

    assert (
        client.post(url, headers=admin, json={"rating": "down", "comment": "wrong"}).status_code
        == 204
    )
    assert conversation(client, admin, cid)["messages"][1]["feedback"] == "down"
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Feedback)) == 1  # changed, not duplicated
    assert db.scalars(select(Feedback.comment)).one() == "wrong"

    # Nobody else can rate it, and they see no rating of their own on it.
    assert client.post(url, headers=user, json={"rating": "up"}).status_code == 404
    assert client.post(url, headers=other, json={"rating": "up"}).status_code == 404
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Feedback)) == 1


def test_only_assistant_messages_can_be_rated_and_the_rating_is_validated(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    question_id = db.scalars(select(Message.id).where(Message.role == "user")).one()
    assert (
        client.post(
            f"/api/messages/{question_id}/feedback", headers=admin, json={"rating": "up"}
        ).status_code
        == 404
    )
    url = f"/api/messages/{done['message_id']}/feedback"
    for bad in (
        {"rating": "meh"},
        {"rating": "up", "comment": "a\x00b"},
        {"rating": "up", "comment": "x" * 2001},
        {},
    ):
        assert client.post(url, headers=admin, json=bad).status_code == 422
    unknown = f"/api/messages/{uuid.uuid4()}/feedback"
    assert client.post(unknown, headers=admin, json={"rating": "up"}).status_code == 404


def test_two_first_ratings_at_once_cannot_collide(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    """The rating is one INSERT ... ON CONFLICT, not a read followed by an insert."""
    from concurrent.futures import ThreadPoolExecutor

    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    url = f"/api/messages/{done['message_id']}/feedback"
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(
            pool.map(
                lambda n: (
                    client.post(
                        url, headers=admin, json={"rating": "up" if n % 2 else "down"}
                    ).status_code
                ),
                range(12),
            )
        )
    assert set(codes) == {204}
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Feedback)) == 1


# --- Saved citations must not outlive the right to read them ----------------------------------------------


def test_deleting_a_document_blanks_its_passages_in_saved_conversations(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    done = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    cid = done["conversation_id"]
    assert (
        "twenty five days"
        in conversation(client, admin, cid)["messages"][1]["citations"][0]["text"]
    )

    assert (
        client.delete(f"/api/documents/{library['leave']['id']}", headers=admin).status_code == 204
    )

    # In the database itself, not only in what the API returns.
    db.expire_all()
    stored = db.scalars(select(Message.citations).where(Message.role == "assistant")).one()
    assert stored[0]["text"] == "" and stored[0]["document_id"] == library["leave"]["id"]
    assert "twenty five days" not in str(stored)

    message = conversation(client, admin, cid)["messages"][1]
    (citation,) = message["citations"]
    assert citation["unavailable"] is True
    assert citation["text"] == "" and citation["document_name"] == "" and citation["n"] == 1
    assert message["content"] == "Twenty five days [1]."  # the answer stays as a record


def test_a_demoted_administrator_no_longer_reads_restricted_passages_in_old_conversations(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    restricted = converse(
        client, admin, fake_llm, SALARY_Q, "The L5 band is ninety thousand euros [1]."
    )
    internal = converse(client, admin, fake_llm, LEAVE_Q, "Twenty five days [1].")
    before = conversation(client, admin, restricted["conversation_id"])["messages"][1]["citations"][
        0
    ]
    assert before["document_name"] == "salary.md" and "ninety thousand" in before["text"]

    db.execute(update(User).where(User.email == "boss@example.com").values(role="user"))
    db.commit()  # the same token now belongs to an ordinary user

    after = conversation(client, admin, restricted["conversation_id"])["messages"][1]["citations"][
        0
    ]
    assert after["unavailable"] is True
    assert "ninety thousand" not in str(after) and "salary" not in str(after).lower()
    still = conversation(client, admin, internal["conversation_id"])["messages"][1]["citations"][0]
    assert not still.get("unavailable") and "twenty five days" in still["text"]


def test_a_conversation_deleted_while_its_answer_is_written_ends_with_a_clear_error(
    db: Session,
) -> None:
    user_row = User(email="u@example.com", password_hash="x", role="user")
    db.add(user_row)
    db.commit()
    plan = chat_service.ChatPlan(
        user_id=user_row.id,
        question="anything",
        conversation_id=uuid.uuid4(),  # no such conversation: it was deleted in the meantime
        new_conversation=False,
        history=[],
        search_filter=build_filter(user_row),
    )
    with pytest.raises(chat_service.ConversationNotFound):
        chat_service._persist(plan, "answer", [], None, {}, False)

    event = chat_service.error_event(chat_service.ConversationNotFound())
    assert event.data["code"] == "conversation_deleted" and "deleted" in event.data["message"]


# --- Authentication on every route -------------------------------------------------------------------

PUBLIC = {
    ("POST", "/api/auth/signup"),
    ("POST", "/api/auth/login"),
    ("GET", "/api/health"),
    ("GET", "/api/ready"),
}


def protected_routes() -> list[tuple[str, str]]:
    """Every (method, concrete path) the API serves except the public ones.

    Read from the OpenAPI schema: it lists the real routes whatever FastAPI does internally with
    included routers, so a route added later is covered without touching this test.
    """
    found = []
    for template, operations in app.openapi()["paths"].items():
        path = re.sub(r"\{[^}]+\}", lambda _m: str(uuid.uuid4()), template)
        for method in sorted(m.upper() for m in operations):
            if (method, template) not in PUBLIC:
                found.append((method, path))
    return found


def test_the_route_list_is_not_empty_and_covers_every_router() -> None:
    routes = protected_routes()
    paths = {path.split("/")[2] for _, path in routes}
    assert {"documents", "chat", "conversations", "messages", "auth"} <= paths
    assert len(routes) >= 12


@pytest.mark.parametrize(
    "method_and_path", protected_routes(), ids=lambda mp: f"{mp[0]} {mp[1][:28]}"
)
def test_every_route_but_the_public_ones_refuses_a_request_without_a_valid_token(
    client: TestClient, method_and_path: tuple[str, str]
) -> None:
    method, path = method_and_path
    assert client.request(method, path).status_code == 401
    bad = {"Authorization": "Bearer not.a.token"}
    assert client.request(method, path, headers=bad).status_code == 401
    wrong_scheme = {"Authorization": "Basic Zm9vOmJhcg=="}
    assert client.request(method, path, headers=wrong_scheme).status_code == 401


# --- NUL characters are refused, not stored or turned into a 500 -------------------------------------------


def test_no_text_field_accepts_a_nul_character(
    client: TestClient, admin: Headers, library: dict
) -> None:
    assert client.post("/api/chat", headers=admin, json={"question": "a\x00b"}).status_code == 422
    filters = {"question": "ok", "filters": {"department": "H\x00R"}}
    assert client.post("/api/chat", headers=admin, json=filters).status_code == 422
    assert client.get("/api/documents", headers=admin, params={"q": "a\x00b"}).status_code == 422
    assert (
        client.get("/api/documents", headers=admin, params={"department": "\x00"}).status_code
        == 422
    )
    signup = {"email": "nu\x00ll@example.com", "password": "correct horse battery"}
    assert client.post("/api/auth/signup", json=signup).status_code == 422
    assert client.get("/api/documents", headers=admin).status_code == 200  # the server is fine


def test_an_off_topic_question_still_refuses_after_all_of_this(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    done = of(ask(client, admin, "Who won the football world cup in Brazil?"), "done")[0]
    assert done["refused"] and done["answer"] == REFUSAL_MESSAGE and len(fake_llm.calls) == 0


# --- Database connections are not held while a stream is open ------------------------------------------------


def test_a_chat_stream_does_not_hold_a_database_transaction_open(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticating a request read the user in a session that stayed open until the response
    ended, so every streaming answer kept one pooled connection "idle in transaction" for as long
    as the model took. Enough concurrent chats then starved everything else of connections."""
    import asyncio
    import threading

    from sqlalchemy import text

    reached, gate = threading.Event(), threading.Event()
    original = fake_llm.astream

    async def slow(messages: list[object]):  # noqa: ANN202
        reached.set()
        await asyncio.to_thread(gate.wait, 15)
        async for chunk in original(messages):
            yield chunk

    monkeypatch.setattr(fake_llm, "astream", slow)
    fake_llm.reply("Twenty five days [1].")
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome["events"] = ask(client, admin, LEAVE_Q)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert reached.wait(15), "the stream never reached the model"
        states = dict(
            db.execute(
                text(
                    "select state, count(*) from pg_stat_activity "
                    "where datname = current_database() and pid <> pg_backend_pid() group by state"
                )
            ).all()
        )
    finally:
        gate.set()
        thread.join(20)
    assert states.get("idle in transaction", 0) == 0, states
    assert of(outcome["events"], "done")[0]["refused"] is False


def test_a_login_does_not_hold_a_database_connection_during_the_password_check(
    client: TestClient, admin: Headers, monkeypatch: pytest.MonkeyPatch, db: Session
) -> None:
    """bcrypt takes about 250 ms; forty logins at once used to hold every pooled connection."""
    import threading

    from sqlalchemy import text

    from app.services import auth_service

    reached, gate = threading.Event(), threading.Event()
    real = auth_service.verify_password

    def slow(password: str, password_hash: str) -> bool:
        reached.set()
        gate.wait(15)
        return real(password, password_hash)

    monkeypatch.setattr(auth_service, "verify_password", slow)
    outcome: dict[str, int] = {}

    def run() -> None:
        body = {"email": "boss@example.com", "password": "correct horse battery"}
        outcome["status"] = client.post("/api/auth/login", json=body).status_code

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert reached.wait(15)
        states = dict(
            db.execute(
                text(
                    "select state, count(*) from pg_stat_activity "
                    "where datname = current_database() and pid <> pg_backend_pid() group by state"
                )
            ).all()
        )
    finally:
        gate.set()
        thread.join(20)
    assert states.get("idle in transaction", 0) == 0, states
    assert outcome["status"] == 200  # and the login itself still works after releasing the session
