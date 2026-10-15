"""GraphRAG: passages connected through the knowledge graph are added to an answer's context.

Real Postgres and Chroma, fake embeddings, scripted model. ``RETRIEVAL_TOP_K`` is 1 in these tests,
so plain search finds one passage and anything else the model is given came through the graph.
"""

# ruff: noqa: F401, F811  (the chat fixtures are imported, then named as test arguments)

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.constants import REFUSAL_MESSAGE
from app.core import answer_cache, graph_retrieval
from app.core.filters import SearchFilter
from app.db.models import Chunk
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, answer_text, ask, client, of, user
from tests.test_documents_api import one
from tests.test_graph_api import _expose_fake, build, ent, rel

TRANSFERS = b"# Transfers\n\nThe Registrar approves the Credit Transfer Policy for every student.\n"
LIMITS = (
    b"# Limits\n\nThe Credit Transfer Policy sets the Credit Limit: at most thirty credits may be "
    b"transferred.\n"
)
QUESTION = "Who approves the Credit Transfer Policy, the Registrar?"
ANSWER = "The Registrar approves it [1]. At most thirty credits may be transferred [2]."

TRANSFERS_GRAPH = (
    [ent("Registrar", "role"), ent("Credit Transfer Policy", "policy")],
    [rel("Registrar", "Credit Transfer Policy", "approves")],
)
LIMITS_GRAPH = (
    [ent("Credit Transfer Policy", "policy"), ent("Credit Limit", "concept")],
    [
        rel(
            "Credit Transfer Policy",
            "Credit Limit",
            "sets",
            "The Credit Transfer Policy sets the Credit Limit",
        )
    ],
)


@pytest.fixture(autouse=True)
def one_passage_from_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "retrieval_top_k", 1)


def answer_prompts(fake: FakeChatModel) -> list[str]:
    """What the model was asked to answer (graph builds use the same model; their prompts have no
    QUESTION block)."""
    texts = [str(call[-1].content) for call in fake.calls]
    return [text for text in texts if "\n\nQUESTION\n" in text]


def prompt(fake: FakeChatModel, call: int = -1) -> str:
    return answer_prompts(fake)[call]


def sources(fake: FakeChatModel, call: int = -1) -> list[str]:
    """The documents of the numbered sources the model was given, in order."""
    return re.findall(r'<source id="\d+" document="([^"]*)"', prompt(fake, call))


@pytest.fixture
def pair(client: TestClient, admin: Headers) -> dict[str, dict[str, Any]]:
    """Two Internal documents that share the entity 'Credit Transfer Policy', graphs built."""
    docs = {
        "transfers": one(client, admin, "transfers.md", TRANSFERS, department="Academic"),
        "limits": one(client, admin, "limits.md", LIMITS, department="Academic"),
    }
    build(client, admin, docs["transfers"], TRANSFERS_GRAPH)
    build(client, admin, docs["limits"], LIMITS_GRAPH)
    return docs


# --- The graph widens an answer -----------------------------------------------------------------------


def test_a_passage_of_another_document_is_pulled_in_through_a_shared_entity(
    client: TestClient, admin: Headers, pair: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER)
    events = ask(client, admin, QUESTION)

    assert sources(fake_llm) == [
        "transfers.md",
        "limits.md",
    ]  # one found by search, one by the graph
    done = of(events, "done")[0]
    assert done["timings"]["graph_added"] == 1 and done["timings"]["graph_ms"] >= 0
    cited = of(events, "citations")[0]["citations"]
    assert [c["document_name"] for c in cited] == ["transfers.md", "limits.md"]  # both can be cited
    assert not done["flags"]["citations_removed"]
    assert "thirty credits" in cited[1]["text"]


def test_without_the_graph_the_second_document_is_not_found(
    client: TestClient,
    admin: Headers,
    pair: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "graph_retrieval", False)
    fake_llm.reply("The Registrar approves it [1].")
    done = of(ask(client, admin, QUESTION), "done")[0]
    assert sources(fake_llm) == ["transfers.md"]
    assert "graph_added" not in done["timings"]


def test_turning_the_graph_feature_off_turns_the_widening_off_too(
    client: TestClient,
    admin: Headers,
    pair: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "knowledge_graph", False)
    fake_llm.reply("The Registrar approves it [1].")
    ask(client, admin, QUESTION)
    assert sources(fake_llm) == ["transfers.md"]


def test_the_guard_never_sees_the_added_passages(
    client: TestClient,
    admin: Headers,
    pair: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm.reply(ANSWER).reply("The Registrar approves it [1].")
    widened = of(ask(client, admin, QUESTION), "done")[0]
    monkeypatch.setattr(get_settings(), "graph_retrieval", False)
    plain = of(ask(client, admin, QUESTION), "done")[0]
    assert widened["guard"] == plain["guard"]  # same score, same reason: decided on the seeds alone


def test_a_refused_question_is_never_widened(
    client: TestClient, admin: Headers, pair: dict, fake_llm: FakeChatModel
) -> None:
    events = ask(client, admin, "Who won the football world cup in Brazil?")
    assert answer_text(events) == REFUSAL_MESSAGE and answer_prompts(fake_llm) == []
    assert "graph_added" not in of(events, "done")[0]["timings"]


def test_the_extra_passages_are_stored_with_the_answer_like_any_citation(
    client: TestClient, admin: Headers, pair: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply(ANSWER)
    done = of(ask(client, admin, QUESTION), "done")[0]
    conversation = client.get(f"/api/conversations/{done['conversation_id']}", headers=admin).json()
    (assistant,) = [m for m in conversation["messages"] if m["role"] == "assistant"]
    assert [c["document_name"] for c in assistant["citations"]] == ["transfers.md", "limits.md"]


# --- What it must never add ---------------------------------------------------------------------------------


def test_a_restricted_passage_is_never_added_for_a_user_but_is_for_an_admin(
    client: TestClient, admin: Headers, user: Headers, fake_llm: FakeChatModel
) -> None:
    transfers = one(
        client, admin, "transfers.md", TRANSFERS, department="Academic", access_level="Internal"
    )
    limits = one(
        client, admin, "limits.md", LIMITS, department="Academic", access_level="Restricted"
    )
    build(client, admin, transfers, TRANSFERS_GRAPH)
    build(client, admin, limits, LIMITS_GRAPH)

    fake_llm.reply("The Registrar approves it [1].")
    as_user = of(ask(client, user, QUESTION), "done")[0]
    assert sources(fake_llm, 0) == ["transfers.md"]
    assert as_user["timings"]["graph_added"] == 0
    assert "thirty credits" not in prompt(fake_llm, 0)

    fake_llm.reply(ANSWER)
    of(ask(client, admin, QUESTION), "done")
    assert sources(fake_llm, 1) == ["transfers.md", "limits.md"]


def test_a_document_outside_the_chosen_selection_is_never_added(
    client: TestClient, admin: Headers, pair: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("The Registrar approves it [1].")
    ask(client, admin, QUESTION, filters={"document_ids": [pair["transfers"]["id"]]})
    assert sources(fake_llm) == ["transfers.md"]


def test_a_deleted_document_is_never_added(
    client: TestClient, admin: Headers, pair: dict, fake_llm: FakeChatModel
) -> None:
    assert client.delete(f"/api/documents/{pair['limits']['id']}", headers=admin).status_code == 204
    fake_llm.reply("The Registrar approves it [1].")
    done = of(ask(client, admin, QUESTION), "done")[0]
    assert sources(fake_llm) == ["transfers.md"] and done["timings"]["graph_added"] == 0
    assert "thirty credits" not in prompt(fake_llm)


def test_a_very_common_entity_is_not_followed(
    client: TestClient,
    admin: Headers,
    pair: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        get_settings(), "kg_max_entity_chunks", 1
    )  # 'Credit Transfer Policy' is in 2
    fake_llm.reply("The Registrar approves it [1].")
    done = of(ask(client, admin, QUESTION), "done")[0]
    assert sources(fake_llm) == ["transfers.md"] and done["timings"]["graph_added"] == 0


def test_an_expansion_that_fails_costs_nothing_but_the_extra_context(
    client: TestClient,
    admin: Headers,
    pair: dict,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(graph_retrieval, "expand", broken)
    fake_llm.reply("The Registrar approves it [1].")
    events = ask(client, admin, QUESTION)
    assert not of(events, "done")[0]["refused"] and answer_text(events).startswith("The Registrar")


def test_nothing_is_added_when_no_graph_has_been_built(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel
) -> None:
    one(client, admin, "transfers.md", TRANSFERS, department="Academic")
    one(client, admin, "limits.md", LIMITS, department="Academic")
    fake_llm.reply("The Registrar approves it [1].")
    done = of(ask(client, admin, QUESTION), "done")[0]
    assert sources(fake_llm) == ["transfers.md"] and done["timings"]["graph_added"] == 0


# --- Hops ---------------------------------------------------------------------------------------------------


CHAIN = [
    (
        b"# One\n\nAlpha Rules govern the Beta Process.\n",
        [ent("Alpha Rules"), ent("Beta Process")],
        [rel("Alpha Rules", "Beta Process", "governs")],
    ),
    (
        b"# Two\n\nThe Beta Process requires the Gamma Form.\n",
        [ent("Beta Process"), ent("Gamma Form")],
        [rel("Beta Process", "Gamma Form", "requires")],
    ),
    (
        b"# Three\n\nThe Gamma Form is filed with the Delta Office.\n",
        [ent("Gamma Form"), ent("Delta Office")],
        [rel("Gamma Form", "Delta Office", "filed_with")],
    ),
    (
        b"# Four\n\nThe Delta Office reports to the Epsilon Board.\n",
        [ent("Delta Office"), ent("Epsilon Board")],
        [rel("Delta Office", "Epsilon Board", "reports_to")],
    ),
]
CHAIN_QUESTION = "Which Alpha Rules govern the Beta Process?"


@pytest.fixture
def chain(client: TestClient, admin: Headers) -> None:
    for number, (text, entities, relations) in enumerate(CHAIN, start=1):
        document = one(client, admin, f"d{number}.md", text, department="Academic")
        build(client, admin, document, (entities, relations))


def test_each_extra_hop_reaches_one_document_further(
    client: TestClient,
    admin: Headers,
    chain: None,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "graph_expansion_max_chunks", 5)
    reached = {}
    for hops in (1, 2, 3):
        monkeypatch.setattr(get_settings(), "graph_hops", hops)
        fake_llm.reply("Alpha Rules govern it [1].")
        ask(client, admin, CHAIN_QUESTION)
        reached[hops] = sources(fake_llm)
    assert reached[1][0] == "d1.md" and reached[1][1] == "d2.md"  # nearest first
    assert "d4.md" not in reached[1] and "d4.md" not in reached[2]
    assert reached[3] == ["d1.md", "d2.md", "d3.md", "d4.md"]


def test_only_the_configured_number_of_passages_is_added_nearest_first(
    client: TestClient,
    admin: Headers,
    chain: None,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "graph_expansion_max_chunks", 1)
    monkeypatch.setattr(get_settings(), "graph_hops", 3)
    fake_llm.reply("Alpha Rules govern it [1].")
    done = of(ask(client, admin, CHAIN_QUESTION), "done")[0]
    assert sources(fake_llm) == ["d1.md", "d2.md"] and done["timings"]["graph_added"] == 1


# --- Saved answers --------------------------------------------------------------------------------------------


def test_a_saved_answer_is_dropped_when_a_graph_is_built_that_could_change_it(
    client: TestClient,
    admin: Headers,
    fake_llm: FakeChatModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "semantic_cache", True)
    transfers = one(client, admin, "transfers.md", TRANSFERS, department="Academic")
    limits = one(client, admin, "limits.md", LIMITS, department="Academic")
    build(client, admin, transfers, TRANSFERS_GRAPH)

    fake_llm.reply("The Registrar approves it [1].")
    ask(client, admin, QUESTION)
    assert len(answer_cache.get_cache()) == 1
    assert of(ask(client, admin, QUESTION), "done")[0]["cached"]  # served from the cache

    build(client, admin, limits, LIMITS_GRAPH)  # the limits document is now connected
    assert len(answer_cache.get_cache()) == 0
    fake_llm.reply(ANSWER)
    after = of(ask(client, admin, QUESTION), "done")[0]
    assert not after["cached"] and "limits.md" in sources(fake_llm)


def test_the_cache_scope_depends_on_whether_the_graph_widens_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = SearchFilter(access_levels=("Public", "Internal"))
    widened = answer_cache.scope_key(base)
    monkeypatch.setattr(get_settings(), "graph_hops", 1)
    assert answer_cache.scope_key(base) != widened
    monkeypatch.undo()
    monkeypatch.setattr(get_settings(), "graph_retrieval", False)
    assert answer_cache.scope_key(base) != widened
