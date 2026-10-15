"""The semantic answer cache: a repeated or near-identical question is answered without searching
or calling the model, and a saved answer never outlives the documents behind it.

The first half tests the cache on its own (no services). The second half is chat end to end on the
real stack with the scripted model: what is served, to whom, and what removes an entry.
"""

# ruff: noqa: F401, F811  (the chat fixtures are imported, then named as test arguments)

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import REFUSAL_MESSAGE
from app.core import answer_cache
from app.core.bm25 import ChunkMeta
from app.core.filters import SearchFilter
from app.db.models import Message, QueryEvent
from app.services import cache_service, ingestion
from tests.conftest import FakeChatModel
from tests.test_chat import (
    Headers,
    admin,
    answer_text,
    ask,
    client,
    library,
    of,
    user,
)
from tests.test_documents_api import one

LEVELS = ("Public", "Internal")
QUESTION = "How many days of paid annual leave do full-time employees receive?"
ANSWER = "Full-time staff get twenty five days of annual leave [1]."


def vec(*values: float) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    return (array / np.linalg.norm(array)).tolist()


def entry(
    scope: str = "s",
    vector: list[float] | None = None,
    *,
    search_filter: SearchFilter | None = None,
    documents: tuple[str, ...] = ("d1",),
    created_at: float | None = None,
) -> answer_cache.CacheEntry:
    made = answer_cache.CacheEntry(
        scope=scope,
        vector=np.asarray(vector or vec(1, 0, 0), dtype=np.float32),
        search_filter=search_filter or SearchFilter(access_levels=LEVELS),
        answer="a [1]",
        citations=[{"chunk_id": "d1_chunk_0000", "document_id": documents[0]}],
        conflicts=[],
        suggestions=[],
        guard={"passed": True, "reason": "relevant", "score": 0.9, "threshold": 0.6},
        cited_chunk_ids=("d1_chunk_0000",),
        cited_document_ids=frozenset(documents),
        saved_tokens=100,
    )
    if created_at is not None:
        made.created_at = created_at
    return made


def meta(document_id: str = "new", department: str = "HR", level: str = "Internal") -> ChunkMeta:
    return ChunkMeta("", document_id, department, "Policy", 20250101, level)


@pytest.fixture
def cache() -> answer_cache.AnswerCache:
    return answer_cache.get_cache()  # emptied around every test by the autouse fixture


# --- The cache on its own ------------------------------------------------------------------------


def test_a_question_at_the_threshold_hits_and_one_below_it_does_not(
    cache: answer_cache.AnswerCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache_threshold", 0.95)
    assert cache.store(entry(), cache.generation) is not None

    # cos = 0.96 and 0.94 against (1, 0, 0)
    close = [0.96, float(np.sqrt(1 - 0.96**2)), 0.0]
    far = [0.94, float(np.sqrt(1 - 0.94**2)), 0.0]
    hit = cache.lookup("s", close)
    assert hit is not None and hit.similarity == pytest.approx(0.96, abs=1e-4)
    assert cache.lookup("s", far) is None


def test_the_closest_of_several_entries_wins(cache: answer_cache.AnswerCache) -> None:
    cache.store(entry(vector=vec(1, 0.1, 0)), cache.generation)
    nearest = cache.store(entry(vector=vec(1, 0.02, 0)), cache.generation)
    hit = cache.lookup("s", vec(1, 0, 0))
    assert hit is not None and hit.entry_id == nearest


def test_an_entry_is_never_served_to_another_scope(cache: answer_cache.AnswerCache) -> None:
    cache.store(entry(scope="admin"), cache.generation)
    assert cache.lookup("user", vec(1, 0, 0)) is None
    assert cache.lookup("admin", vec(1, 0, 0)) is not None


def test_the_scope_depends_on_levels_filters_and_selection_but_not_their_order() -> None:
    base = SearchFilter(access_levels=("Public", "Internal"))
    assert answer_cache.scope_key(base) == answer_cache.scope_key(
        SearchFilter(access_levels=("Internal", "Public"))
    )
    assert answer_cache.scope_key(base) != answer_cache.scope_key(
        SearchFilter(access_levels=("Public", "Internal", "Restricted"))
    )
    assert answer_cache.scope_key(base) != answer_cache.scope_key(
        SearchFilter(access_levels=("Public", "Internal"), department="HR")
    )
    a = SearchFilter(access_levels=("Public",), document_ids=("1", "2"))
    b = SearchFilter(access_levels=("Public",), document_ids=("2", "1"))
    assert answer_cache.scope_key(a) == answer_cache.scope_key(b)
    assert answer_cache.scope_key(a) != answer_cache.scope_key(
        SearchFilter(access_levels=("Public",))
    )


def test_a_changed_model_or_prompt_version_changes_every_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = SearchFilter(access_levels=LEVELS)
    before = answer_cache.scope_key(base)
    monkeypatch.setattr(get_settings(), "llm_model", "another-model")
    assert answer_cache.scope_key(base) != before
    monkeypatch.undo()
    monkeypatch.setattr(answer_cache.prompts, "PROMPT_VERSION", "2")
    assert answer_cache.scope_key(base) != before


def test_an_expired_entry_is_gone(
    cache: answer_cache.AnswerCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache_ttl_seconds", 60)
    cache.store(entry(created_at=time.monotonic() - 61), cache.generation)
    assert cache.lookup("s", vec(1, 0, 0)) is None
    assert len(cache) == 0


def test_the_least_recently_used_entry_is_dropped_first(
    cache: answer_cache.AnswerCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache_max_entries", 2)
    cache.store(entry(vector=vec(1, 0, 0)), cache.generation)
    cache.store(entry(vector=vec(0, 1, 0)), cache.generation)
    assert cache.lookup("s", vec(1, 0, 0)) is not None  # touched: now the newest
    cache.store(entry(vector=vec(0, 0, 1)), cache.generation)
    assert cache.lookup("s", vec(0, 1, 0)) is None
    assert cache.lookup("s", vec(1, 0, 0)) is not None


def test_invalidating_a_document_drops_the_entries_it_can_change(
    cache: answer_cache.AnswerCache,
) -> None:
    hr = SearchFilter(access_levels=LEVELS, department="HR")
    it = SearchFilter(access_levels=LEVELS, department="IT")
    restricted_only = SearchFilter(access_levels=("Public",))
    cache.store(entry(vector=vec(1, 0, 0), search_filter=hr), cache.generation)
    cache.store(entry(vector=vec(0, 1, 0), search_filter=it), cache.generation)
    cache.store(entry(vector=vec(0, 0, 1), search_filter=restricted_only), cache.generation)

    # A new Internal HR document: only the HR entry could now be answered differently.
    assert cache.invalidate_document("new", meta()) == 1
    assert cache.lookup("s", vec(1, 0, 0)) is None
    assert cache.lookup("s", vec(0, 1, 0)) is not None
    assert cache.lookup("s", vec(0, 0, 1)) is not None


def test_an_entry_that_cites_a_document_goes_with_it_whatever_its_filter(
    cache: answer_cache.AnswerCache,
) -> None:
    cache.store(entry(documents=("gone",)), cache.generation)
    assert (
        cache.invalidate_document("gone", meta("gone", department="Legal", level="Restricted")) == 1
    )


def test_without_the_documents_metadata_everything_is_dropped(
    cache: answer_cache.AnswerCache,
) -> None:
    cache.store(entry(vector=vec(1, 0, 0)), cache.generation)
    cache.store(entry(vector=vec(0, 1, 0)), cache.generation)
    assert cache.invalidate_document("x", None) == 2


def test_an_answer_written_before_an_invalidation_is_not_stored_after_it(
    cache: answer_cache.AnswerCache,
) -> None:
    seen = cache.generation  # a turn starts searching here ...
    cache.invalidate_document("new", meta())  # ... a document is indexed meanwhile ...
    assert cache.store(entry(), seen) is None  # ... so its answer may be stale and is dropped
    assert cache.store(entry(), cache.generation) is not None


def test_a_message_can_be_evicted_by_id(cache: answer_cache.AnswerCache) -> None:
    entry_id = cache.store(entry(), cache.generation)
    assert entry_id is not None
    cache.register_message(entry_id, "m1")
    assert cache.evict_message("other") == 0 and len(cache) == 1
    assert cache.evict_message("m1") == 1 and len(cache) == 0


# --- Chat end to end -------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cache_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache is off for the rest of the suite (conftest); here it is on, with the same
    after-indexing hook the application registers at startup."""
    monkeypatch.setattr(get_settings(), "semantic_cache", True)
    ingestion.on_indexed(cache_service.on_document_indexed)


def stored(cache: answer_cache.AnswerCache) -> int:
    return len(cache)


def test_the_same_question_twice_calls_the_model_once(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    cache: answer_cache.AnswerCache,
) -> None:
    fake_llm.reply(ANSWER)
    first = ask(client, admin, QUESTION)
    assert not of(first, "done")[0]["cached"] and stored(cache) == 1

    second = ask(client, admin, QUESTION)
    assert len(fake_llm.calls) == 1  # no second model call
    assert ("status", {"stage": "cache_hit"}) in second
    assert answer_text(second) == answer_text(first) == ANSWER
    assert of(second, "citations") == of(first, "citations")
    done = of(second, "done")[0]
    assert done["cached"] and not done["refused"] and done["usage"] == {}
    assert done["timings"]["cache"] == "hit" and done["timings"]["cache_similarity"] >= 0.95

    # A hit is a normal message in the conversation, and a normal measurement.
    assistants = db.scalars(
        select(Message).where(Message.role == "assistant").order_by(Message.created_at)
    ).all()
    assert [m.content for m in assistants] == [ANSWER, ANSWER]
    assert assistants[1].citations[0]["document_name"] == "leave.md"
    assert str(assistants[1].id) == done["message_id"]
    events = db.scalars(select(QueryEvent).order_by(QueryEvent.created_at)).all()
    assert len(events) == 2 and events[1].timings["cache"] == "hit"
    assert events[1].total_tokens in (0, None) and events[1].cited == 1

    conversation = client.get(f"/api/conversations/{done['conversation_id']}", headers=admin).json()
    assert [m["cached"] for m in conversation["messages"] if m["role"] == "assistant"] == [True]


def test_a_reworded_question_is_a_hit_and_a_different_one_is_not(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache_threshold", 0.9)
    fake_llm.reply(ANSWER).reply("Check the load balancer first [1].")
    ask(client, admin, QUESTION)

    reworded = ask(client, admin, "How many days of paid annual leave do employees get, full-time?")
    assert of(reworded, "done")[0]["cached"] and len(fake_llm.calls) == 1

    other = ask(client, admin, "Tell me about the gateway load balancer upstream service errors")
    assert not of(other, "done")[0]["cached"] and len(fake_llm.calls) == 2


def test_two_roles_never_share_an_answer(
    client: TestClient, admin: Headers, user: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    question = "What is the grade L5 salary band in euros?"
    fake_llm.reply("The L5 band is ninety thousand euros [1].")
    allowed = ask(client, admin, question)
    assert of(allowed, "citations")[0]["citations"][0]["document_name"] == "salary.md"

    denied = ask(client, user, question)  # same words, but a user may not read Restricted
    assert answer_text(denied) == REFUSAL_MESSAGE
    assert not of(denied, "done")[0]["cached"] and "ninety thousand" not in answer_text(denied)
    assert len(fake_llm.calls) == 1


def test_a_different_document_selection_is_a_different_question(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER).reply(ANSWER)
    ask(client, admin, QUESTION)
    narrowed = ask(client, admin, QUESTION, filters={"document_ids": [library["leave"]["id"]]})
    assert not of(narrowed, "done")[0]["cached"] and len(fake_llm.calls) == 2

    again = ask(client, admin, QUESTION, filters={"document_ids": [library["leave"]["id"]]})
    assert of(again, "done")[0]["cached"] and len(fake_llm.calls) == 2


def test_uploading_a_document_in_scope_stops_the_saved_answer(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER).reply("Leave was updated [1].")
    ask(client, admin, QUESTION)

    one(
        client,
        admin,
        "leave2.md",
        b"# Leave Update\n\nA new leave rule applies.\n",
        department="HR",
    )
    after = ask(client, admin, QUESTION)
    assert not of(after, "done")[0]["cached"] and len(fake_llm.calls) == 2


def test_a_document_outside_the_scope_leaves_the_answer_alone(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER)
    ask(client, admin, QUESTION, filters={"department": "HR"})
    one(client, admin, "ops.md", b"# Ops\n\nRunbook text.\n", department="IT")
    again = ask(client, admin, QUESTION, filters={"department": "HR"})
    assert of(again, "done")[0]["cached"] and len(fake_llm.calls) == 1


def test_deleting_the_cited_document_removes_its_saved_answer(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER)
    ask(client, admin, QUESTION)

    deleted = client.delete(f"/api/documents/{library['leave']['id']}", headers=admin)
    assert deleted.status_code == 204
    after = ask(client, admin, QUESTION)
    assert answer_text(after) == REFUSAL_MESSAGE  # not the old answer, and its text is gone
    assert not of(after, "done")[0]["cached"] and len(fake_llm.calls) == 1


def test_a_hit_is_checked_against_the_database_even_if_invalidation_was_missed(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
    cache: answer_cache.AnswerCache,
) -> None:
    fake_llm.reply(ANSWER)
    ask(client, admin, QUESTION)
    assert stored(cache) == 1

    # The document stops being searchable without the cache being told.
    db.execute(
        text("UPDATE documents SET status = 'failed' WHERE id = CAST(:id AS uuid)"),
        {"id": library["leave"]["id"]},
    )
    db.commit()
    after = ask(client, admin, QUESTION)
    assert answer_text(after) == REFUSAL_MESSAGE and not of(after, "done")[0]["cached"]
    assert len(fake_llm.calls) == 1 and stored(cache) == 0


def test_a_thumbs_down_removes_the_saved_answer_and_a_thumbs_up_keeps_it(
    client: TestClient, admin: Headers, library: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER).reply(ANSWER)
    first = of(ask(client, admin, QUESTION), "done")[0]

    up = client.post(
        f"/api/messages/{first['message_id']}/feedback", headers=admin, json={"rating": "up"}
    )
    assert up.status_code == 204
    served = of(ask(client, admin, QUESTION), "done")[0]
    assert served["cached"]

    # A rating on the message that was *served* from the cache removes the entry as well.
    down = client.post(
        f"/api/messages/{served['message_id']}/feedback", headers=admin, json={"rating": "down"}
    )
    assert down.status_code == 204
    assert not of(ask(client, admin, QUESTION), "done")[0]["cached"]
    assert len(fake_llm.calls) == 2


def test_refusals_and_unclean_answers_are_not_saved(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    cache: answer_cache.AnswerCache,
) -> None:
    ask(client, admin, "Who won the football world cup in Brazil?")  # guard refusal
    assert stored(cache) == 0

    fake_llm.reply(REFUSAL_MESSAGE)  # the model itself says the documents do not contain it
    ask(client, admin, QUESTION)
    assert stored(cache) == 0

    fake_llm.reply("Leave is 25 days.").reply("Leave is still 25 days.")  # never cites: flagged
    ask(client, admin, QUESTION)
    assert stored(cache) == 0


def test_a_failed_extra_keeps_the_answer_out_of_the_cache(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    cache: answer_cache.AnswerCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A follow-up suggestion call that failed would be missing from the saved answer for good."""
    monkeypatch.setattr(get_settings(), "followup_suggestions", True)
    fake_llm.reply(ANSWER).fail(RuntimeError("boom"))
    done = of(ask(client, admin, QUESTION), "done")[0]
    assert not done["refused"] and stored(cache) == 0


def test_a_follow_up_the_rewrite_did_not_change_is_not_cached(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    cache: answer_cache.AnswerCache,
) -> None:
    fake_llm.reply(ANSWER)
    first = of(ask(client, admin, QUESTION), "done")[0]
    assert stored(cache) == 1

    follow_up = "How many days of annual leave is that for full-time employees?"
    fake_llm.reply(follow_up)  # the rewrite hands the question back unchanged ...
    fake_llm.reply(ANSWER)
    second = of(ask(client, admin, follow_up, conversation_id=first["conversation_id"]), "done")[0]
    assert not second["cached"] and stored(cache) == 1  # ... so it is neither served nor saved


def test_the_cache_can_be_switched_off(
    client: TestClient,
    admin: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    cache: answer_cache.AnswerCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache", False)
    fake_llm.reply(ANSWER).reply(ANSWER)
    ask(client, admin, QUESTION)
    again = ask(client, admin, QUESTION)
    assert not of(again, "done")[0]["cached"] and len(fake_llm.calls) == 2 and stored(cache) == 0
