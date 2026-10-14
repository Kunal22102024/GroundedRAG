"""Chat end to end: real Postgres and Chroma, fake embeddings, scripted LLM (no network, no key).

The guarantees under test: answers carry verified citations; below the threshold the refusal
sentence is returned WITHOUT calling the LLM; a user can never retrieve Restricted documents, in
either search channel; a broken index is an error, never a silent "not found"; rate limits and
model failures end the stream with a friendly ``error`` event.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import REFUSAL_MESSAGE
from app.core import vectorstore
from app.db.models import Conversation, Message
from app.main import app
from tests.conftest import FakeChatModel, FakeEmbeddings
from tests.test_documents_api import auth, one

LEAVE = b"# Annual Leave\n\nFull-time employees receive twenty five days of paid annual leave each year.\n"
GATEWAY = (
    b"# Gateway Errors\n\nWhen the gateway returns ERR_404_NET the upstream service could not be "
    b"reached, so check the load balancer first.\n"
)
SALARY = (
    b"# Salary Bands\n\nThe confidential SALARY_X9 band for grade L5 is ninety thousand euros.\n"
)
Headers = dict[str, str]
Events = list[tuple[str, dict[str, Any]]]


@pytest.fixture
def client(
    db: Session,
    upload_dir: Path,
    vectors: Any,
    fake_embeddings: FakeEmbeddings,
    inline_ingestion: list,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    # The fake bag-of-words embedding scores a good paraphrase around 0.4-0.7, not Gemini's range.
    monkeypatch.setattr(get_settings(), "relevance_threshold", 0.30)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin(client: TestClient) -> Headers:
    return auth(client, "boss@example.com")


@pytest.fixture
def user(client: TestClient, admin: Headers) -> Headers:
    return auth(client, "worker@example.com")


@pytest.fixture
def library(client: TestClient, admin: Headers) -> dict[str, dict[str, Any]]:
    """Three indexed documents: HR leave, IT gateway runbook, and an HR Restricted salary sheet."""
    return {
        "leave": one(client, admin, "leave.md", LEAVE, department="HR", access_level="Internal"),
        "gateway": one(
            client, admin, "gateway.md", GATEWAY, department="IT", access_level="Internal"
        ),
        "salary": one(
            client, admin, "salary.md", SALARY, department="HR", access_level="Restricted"
        ),
    }


def ask(client: TestClient, headers: Headers, question: str, **body: Any) -> Events:
    with client.stream(
        "POST", "/api/chat", headers=headers, json={"question": question, **body}
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())
    events: Events = []
    for block in raw.strip().split("\n\n"):
        name, data = "", {}
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events


def of(events: Events, name: str) -> list[dict[str, Any]]:
    return [data for event, data in events if event == name]


def answer_text(events: Events) -> str:
    return "".join(d["text"] for d in of(events, "token"))


# --- A cited answer ------------------------------------------------------------------------------


def test_a_question_gets_a_streamed_cited_answer_with_timings(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Full-time staff get ", "twenty five days of annual leave [1].")
    events = ask(
        client, admin, "How many days of paid annual leave do full-time employees receive?"
    )

    assert [name for name, _ in events][0] == "status"
    assert [name for name, _ in events][-2:] == ["citations", "done"]
    assert answer_text(events) == "Full-time staff get twenty five days of annual leave [1]."

    (citation,) = of(events, "citations")[0]["citations"]
    assert citation["n"] == 1 and citation["document_name"] == "leave.md"
    assert citation["page_number"] == 1 and "twenty five days" in citation["text"]

    done = of(events, "done")[0]
    assert done["answer"] == answer_text(events) and not done["refused"]
    assert done["guard"]["passed"] and done["guard"]["reason"] == "relevant"
    assert done["guard"]["score"] >= done["guard"]["threshold"]
    for stage in ("embed_ms", "dense_ms", "keyword_ms", "fusion_ms", "generate_ms", "total_ms"):
        assert done["timings"][stage] >= 0
    assert done["usage"]["input_tokens"] == 120 and done["usage"]["reasoning_tokens"] == 0
    assert done["flags"] == {
        "partial": False,
        "citations_removed": 0,
        "uncited": False,
        "retried": False,
        "rewritten": False,
    }

    messages = db.scalars(select(Message).order_by(Message.created_at)).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].citations[0]["document_name"] == "leave.md" and not messages[1].flagged
    assert messages[1].guard_score == pytest.approx(done["guard"]["score"], abs=1e-3)
    assert str(messages[1].conversation_id) == done["conversation_id"]


# --- Refusal without calling the LLM -------------------------------------------------------------------


def test_off_topic_question_is_refused_without_calling_the_llm(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    events = ask(client, admin, "Who won the football world cup in Brazil?")

    assert answer_text(events) == REFUSAL_MESSAGE
    done = of(events, "done")[0]
    assert done["refused"] and done["guard"]["reason"] == "below_threshold"
    assert of(events, "citations")[0]["citations"] == []
    assert len(fake_llm.calls) == 0  # the call counter: the model was never contacted
    assistant = db.scalars(select(Message).where(Message.role == "assistant")).one()
    assert assistant.content == REFUSAL_MESSAGE and assistant.citations == []


def test_an_empty_library_refuses_without_calling_the_llm(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel
) -> None:
    events = ask(client, admin, "What is the leave policy?")
    assert answer_text(events) == REFUSAL_MESSAGE
    assert of(events, "done")[0]["guard"]["reason"] == "no_candidates"
    assert len(fake_llm.calls) == 0


# --- Exact tokens: the lexical override ------------------------------------------------------------------


def test_an_awkwardly_phrased_exact_token_question_is_not_refused(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A threshold nothing can reach and no floor below it: only the override can let it through.
    monkeypatch.setattr(get_settings(), "relevance_threshold", 0.99)
    monkeypatch.setattr(get_settings(), "lexical_override_margin", 1.0)
    fake_llm.reply("It means the upstream service could not be reached [1].")
    events = ask(client, admin, "so um whatever, ERR_404_NET?? keeps popping up, any idea lol")

    done = of(events, "done")[0]
    assert done["guard"]["reason"] == "lexical_override" and not done["refused"]
    assert of(events, "citations")[0]["citations"][0]["document_name"] == "gateway.md"
    assert len(fake_llm.calls) == 1


def test_an_unknown_token_is_still_refused(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "relevance_threshold", 0.99)
    monkeypatch.setattr(get_settings(), "lexical_override_margin", 1.0)
    events = ask(client, admin, "what does ERR_999_XYZ mean?")
    assert answer_text(events) == REFUSAL_MESSAGE and len(fake_llm.calls) == 0


def test_the_override_cannot_rescue_a_question_far_below_the_threshold(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact token is in the best keyword hit, but the question is nowhere near the threshold."""
    monkeypatch.setattr(get_settings(), "relevance_threshold", 0.99)  # default margin 0.15
    events = ask(client, admin, "so um whatever, ERR_404_NET?? keeps popping up, any idea lol")
    done = of(events, "done")[0]
    assert done["refused"] and done["guard"]["reason"] == "below_threshold"
    assert len(fake_llm.calls) == 0


# --- Access control: both channels ----------------------------------------------------------------------


def test_a_user_cannot_retrieve_a_restricted_document_but_an_admin_can(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    question = "What is the grade L5 salary band in euros?"

    denied = ask(client, user, question)
    assert answer_text(denied) == REFUSAL_MESSAGE and len(fake_llm.calls) == 0

    fake_llm.reply("The L5 band is ninety thousand euros [1].")
    allowed = ask(client, admin, question)
    assert of(allowed, "citations")[0]["citations"][0]["document_name"] == "salary.md"


def test_the_keyword_channel_cannot_leak_a_restricted_exact_token(
    client: TestClient,
    admin: Headers,
    user: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "relevance_threshold", 0.99)  # only the override could pass
    monkeypatch.setattr(get_settings(), "lexical_override_margin", 1.0)
    events = ask(client, user, "SALARY_X9")
    assert answer_text(events) == REFUSAL_MESSAGE and len(fake_llm.calls) == 0

    fake_llm.reply("The SALARY_X9 band is for grade L5 [1].")
    assert not of(ask(client, admin, "SALARY_X9"), "done")[0]["refused"]


def test_a_user_naming_a_restricted_document_id_still_gets_nothing(
    client: TestClient, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    events = ask(
        client,
        user,
        "What is the grade L5 salary band in euros?",
        filters={"document_ids": [library["salary"]["id"]]},
    )
    assert answer_text(events) == REFUSAL_MESSAGE and len(fake_llm.calls) == 0


def test_a_department_filter_excludes_other_departments(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    question = "Tell me about the gateway load balancer upstream service errors"
    fake_llm.reply("Check the load balancer first [1].")
    inside = ask(client, admin, question, filters={"department": "IT"})
    assert of(inside, "citations")[0]["citations"][0]["document_name"] == "gateway.md"

    outside = ask(client, admin, question, filters={"department": "HR"})
    assert answer_text(outside) == REFUSAL_MESSAGE and len(fake_llm.calls) == 1  # only the first


# --- Citation verification, finish reasons and failures -----------------------------------------------------


def test_an_invalid_citation_is_retried_once_then_stripped_and_flagged(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel, db: Session
) -> None:
    fake_llm.reply("Twenty five days [7].").reply("Twenty five days [8] indeed [1].")
    events = ask(
        client, admin, "How many days of paid annual leave do full-time employees receive?"
    )

    statuses = of(events, "status")
    assert {"stage": "retrying", "reset": True} in statuses
    assert len(fake_llm.calls) == 2
    done = of(events, "done")[0]
    assert "[8]" not in done["answer"] and "[7]" not in done["answer"] and "[1]" in done["answer"]
    assert done["flags"]["citations_removed"] == 1 and done["flags"]["retried"]
    assert db.scalars(select(Message).where(Message.role == "assistant")).one().flagged


def test_max_tokens_returns_the_partial_answer_flagged(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Full-time staff get twenty five days [1] and also", finish="MAX_TOKENS")
    done = of(
        ask(client, admin, "How many days of annual leave do full-time employees get?"), "done"
    )[0]
    assert done["flags"]["partial"] and done["answer"].startswith("Full-time staff get")


@pytest.mark.parametrize(
    ("script", "code"),
    [("safety", "blocked"), ("block", "blocked"), ("empty", "empty")],
)
def test_model_blocks_and_empty_replies_become_error_events(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    script: str,
    code: str,
) -> None:
    if script == "safety":
        fake_llm.reply("partial", finish="SAFETY")
    elif script == "block":
        fake_llm.block()
    else:
        fake_llm.reply()
    events = ask(client, admin, "How many days of annual leave do full-time employees get?")
    (error,) = of(events, "error")
    assert error["code"] == code and error["message"]
    assert of(events, "done") == []
    assert db.scalars(select(Message)).all() == []  # nothing half-saved


def test_a_rate_limit_is_retried_briefly_with_the_servers_hint(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.fail(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 3s."))
    fake_llm.reply("Twenty five days [1].")
    events = ask(client, admin, "How many days of annual leave do full-time employees get?")
    assert {"stage": "rate_limited", "retry_in": 4} in of(events, "status") or any(
        s.get("stage") == "rate_limited" for s in of(events, "status")
    )
    assert fake_llm.slept and 3.9 <= fake_llm.slept[0] <= 4.5  # hint + 1 s, with a little jitter
    assert of(events, "done")[0]["answer"] == "Twenty five days [1]."


def test_a_rate_limit_that_persists_ends_with_a_friendly_error_not_a_hang(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    for _ in range(5):
        fake_llm.fail(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2s."))
    events = ask(client, admin, "How many days of annual leave do full-time employees get?")
    (error,) = of(events, "error")
    assert error["code"] == "rate_limited"
    assert error["message"] == "Rate limit reached, please try again in a minute."
    assert len(fake_llm.calls) == 1 + get_settings().interactive_max_retries


def test_a_long_retry_hint_is_not_waited_out(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.fail(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 120s."))
    (error,) = of(
        ask(client, admin, "How many days of annual leave do full-time employees get?"), "error"
    )
    assert error["code"] == "rate_limited" and fake_llm.slept == [] and len(fake_llm.calls) == 1


def test_embedding_rate_limit_is_also_a_friendly_error(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    fake_embeddings: FakeEmbeddings,
) -> None:
    from app.core import embeddings

    fake_embeddings.error = embeddings.EmbeddingRateLimitError("slow down")
    (error,) = of(ask(client, admin, "How many days of annual leave do employees get?"), "error")
    assert error["code"] == "rate_limited" and "try again in a minute" in error["message"]
    assert len(fake_llm.calls) == 0


# --- Follow-ups and conversations ---------------------------------------------------------------------------------


def test_follow_up_is_rewritten_but_the_first_question_is_not(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    fake_embeddings: FakeEmbeddings,
    db: Session,
) -> None:
    fake_llm.reply("Twenty five days [1].")
    first = ask(client, admin, "How many days of paid annual leave do full-time employees receive?")
    assert len(fake_llm.calls) == 1  # no rewrite call on the first turn
    conversation_id = of(first, "done")[0]["conversation_id"]

    fake_llm.reply("How many days of annual leave do contractors receive?")  # the rewrite
    fake_llm.reply("Twenty five days [1].")
    second = ask(client, admin, "what about contractors?", conversation_id=conversation_id)

    assert len(fake_llm.calls) == 3
    assert "LAST USER QUESTION" in str(fake_llm.calls[1][-1].content)
    assert fake_embeddings.queries[-1] == "How many days of annual leave do contractors receive?"
    done = of(second, "done")[0]
    assert done["flags"]["rewritten"] and done["conversation_id"] == conversation_id
    assert [m.role for m in db.scalars(select(Message).order_by(Message.created_at))] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert db.scalars(select(Conversation)).one().title.startswith("How many days of paid")


def test_another_users_conversation_is_not_found(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("Twenty five days [1].")
    owned = of(ask(client, admin, "How many days of annual leave do employees get?"), "done")[0]
    response = client.post(
        "/api/chat",
        headers=user,
        json={"question": "and for managers?", "conversation_id": owned["conversation_id"]},
    )
    assert response.status_code == 404


def test_chat_requires_a_login_and_a_real_question(client: TestClient, admin: Headers) -> None:
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 401
    assert client.post("/api/chat", headers=admin, json={"question": "   "}).status_code == 422
    assert client.post("/api/chat", headers=admin, json={"question": "x" * 2001}).status_code == 422


# --- Index health and deletion -------------------------------------------------------------------------------------------


def test_a_broken_index_is_an_error_not_a_silent_not_found(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    vectorstore.drop_collection()  # Postgres still lists indexed chunks, Chroma has none
    vectorstore.get_collection()
    events = ask(client, admin, "How many days of annual leave do employees get?")
    (error,) = of(events, "error")
    assert error["code"] == "index_unavailable"
    assert of(events, "token") == [] and len(fake_llm.calls) == 0


def test_a_deleted_document_never_appears_in_answers_again(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    question = "How many days of paid annual leave do full-time employees receive?"
    fake_llm.reply("Twenty five days [1].")
    assert not of(ask(client, admin, question), "done")[0]["refused"]

    assert (
        client.delete(f"/api/documents/{library['leave']['id']}", headers=admin).status_code == 204
    )
    after = ask(client, admin, question)
    assert answer_text(after) == REFUSAL_MESSAGE and len(fake_llm.calls) == 1
