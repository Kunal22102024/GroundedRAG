"""The knowledge graph end to end: build from a document, read it back, and who may see what.

Real Postgres and Chroma, fake embeddings, scripted model. The build runs on the insights thread
like in production, so tests wait for the status instead of assuming it.
"""

# ruff: noqa: F401, F811  (the chat fixtures are imported, then named as test arguments)

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import InsightKind
from app.db.models import KgEntity, KgMention, KgRelation, QueryEvent
from app.services import analysis_service, graph_service, ingestion
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, client, of, user
from tests.test_documents_api import one

HANDBOOK = (
    b"# Credit Transfer\n\nThe Registrar approves the Credit Transfer Policy. Students must "
    b"complete Calculus I before they enrol in Calculus II.\n"
)
COURSES = (
    b"# Courses\n\nCalculus II requires Calculus I as a prerequisite. Professor Ada Lovelace "
    b"teaches Calculus II. The Registrar publishes the course catalogue.\n"
)
BONUS = b"# Pay\n\nThe Compensation Committee sets the Executive Bonus Plan every spring.\n"


def ent(name: str, type_: str = "concept", description: str = "") -> dict:
    return {"name": name, "type": type_, "description": description, "sources": [1]}


def rel(subject: str, object_: str, label: str, quote: str = "") -> dict:
    return {"subject": subject, "object": object_, "label": label, "source": 1, "quote": quote}


def script(fake: FakeChatModel, entities: list[dict], relations: list[dict]) -> None:
    fake.reply(json.dumps({"entities": entities, "relations": relations}))


HANDBOOK_GRAPH = (
    [
        ent("Registrar", "role", "Approves credit transfers."),
        ent("Credit Transfer Policy", "policy"),
        ent("Calculus I", "course"),
        ent("Calculus II", "course"),
    ],
    [
        rel(
            "Registrar",
            "Credit Transfer Policy",
            "approves",
            "The Registrar approves the Credit Transfer Policy",
        ),
        rel("Calculus I", "Calculus II", "prerequisite_of"),
    ],
)
COURSES_GRAPH = (
    [
        ent("Calculus II", "course"),
        ent("Calculus I", "course"),
        ent("Ada Lovelace", "person", "Teaches Calculus II."),
        ent("Registrar", "role"),
    ],
    [rel("Calculus II", "Calculus I", "requires"), rel("Ada Lovelace", "Calculus II", "teaches")],
)
BONUS_GRAPH = (
    [ent("Compensation Committee", "organization"), ent("Executive Bonus Plan", "policy")],
    [rel("Compensation Committee", "Executive Bonus Plan", "governs")],
)


def get_graph(client: TestClient, headers: Headers, **params: str) -> Any:
    return client.get("/api/graph", headers=headers, params=params)


def wait_built(client: TestClient, headers: Headers, document_id: str, timeout: float = 15) -> Any:
    """The graph view of one document once its build is no longer pending."""
    deadline = time.monotonic() + timeout
    while True:
        response = get_graph(client, headers, document_ids=document_id)
        assert response.status_code == 200, response.text
        view = response.json()
        if view["documents"][0]["status"] not in ("pending", "none"):
            return view
        assert time.monotonic() < deadline, "the graph was not built in time"
        time.sleep(0.05)


def build(client: TestClient, headers: Headers, document: dict, graph: tuple) -> Any:
    """Script the model, queue a build, wait for it."""
    fake: FakeChatModel = client.app.state.fake_llm  # set by the fixture below
    script(fake, *graph)
    queued = client.post(f"/api/documents/{document['id']}/graph", headers=headers)
    assert queued.status_code == 202 and queued.json() == {"status": "pending"}
    return wait_built(client, headers, document["id"])


@pytest.fixture(autouse=True)
def _expose_fake(client: TestClient, fake_llm: FakeChatModel) -> None:
    client.app.state.fake_llm = fake_llm


@pytest.fixture
def docs(client: TestClient, admin: Headers) -> dict[str, dict[str, Any]]:
    return {
        "handbook": one(
            client, admin, "handbook.md", HANDBOOK, department="Academic", access_level="Internal"
        ),
        "courses": one(
            client, admin, "courses.md", COURSES, department="Academic", access_level="Internal"
        ),
        "bonus": one(client, admin, "bonus.md", BONUS, department="HR", access_level="Restricted"),
    }


def nodes(view: Any) -> dict[str, dict]:
    return {n["id"]: n for n in view["nodes"]}


# --- Building and reading ---------------------------------------------------------------------------


def test_a_built_graph_has_typed_nodes_and_labelled_edges(
    client: TestClient, admin: Headers, docs: dict, fake_llm: FakeChatModel
) -> None:
    view = build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    assert len(fake_llm.calls) == 1  # one model call for a short document

    (status,) = view["documents"]
    assert status["status"] == "ready" and status["entities"] == 4 and status["relations"] == 2
    assert status["coverage"] == 1.0
    by_id = nodes(view)
    assert set(by_id) == {"registrar", "credit transfer policy", "calculus i", "calculus ii"}
    assert (
        by_id["registrar"]["type"] == "role"
        and by_id["registrar"]["description"] == "Approves credit transfers."
    )
    assert by_id["calculus i"]["degree"] == 1 and by_id["calculus i"]["mentions"] == 1
    edges = {(e["source"], e["target"], e["label"]): e["count"] for e in view["edges"]}
    assert edges == {
        ("registrar", "credit transfer policy", "approves"): 1,
        ("calculus i", "calculus ii", "prerequisite_of"): 1,
    }
    assert view["total_nodes"] == 4 and view["truncated"] is False


def test_an_unbuilt_document_says_so(client: TestClient, admin: Headers, docs: dict) -> None:
    view = get_graph(client, admin, document_ids=docs["handbook"]["id"]).json()
    assert view["documents"][0]["status"] == "none" and view["nodes"] == []


def test_the_same_entity_in_two_documents_is_one_node_that_links_them(
    client: TestClient, admin: Headers, docs: dict
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    build(client, admin, docs["courses"], COURSES_GRAPH)
    view = get_graph(
        client, admin, document_ids=f"{docs['handbook']['id']},{docs['courses']['id']}"
    ).json()

    by_id = nodes(view)
    both = {docs["handbook"]["id"], docs["courses"]["id"]}
    for shared in ("registrar", "calculus i", "calculus ii"):
        assert set(by_id[shared]["document_ids"]) == both
    assert by_id["ada lovelace"]["document_ids"] == [docs["courses"]["id"]]
    labels = {(e["source"], e["target"], e["label"]) for e in view["edges"]}
    assert ("ada lovelace", "calculus ii", "teaches") in labels
    assert ("calculus ii", "calculus i", "requires") in labels
    # Without naming documents, every readable document counts (the same view).
    assert set(nodes(get_graph(client, admin).json())) >= set(by_id)


def test_an_entity_links_back_to_the_passages_it_comes_from(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    build(client, admin, docs["courses"], COURSES_GRAPH)

    detail = client.get("/api/graph/entity", headers=admin, params={"key": "calculus ii"})
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["name"] == "Calculus II" and body["type"] == "course"
    assert {d["name"] for d in body["documents"]} == {"handbook.md", "courses.md"}

    passages = {m["document_name"]: m for m in body["mentions"]}
    assert set(passages) == {"handbook.md", "courses.md"}
    for mention in body["mentions"]:  # shaped like a chat citation, so the drawer opens it
        assert {"chunk_id", "document_id", "page_number", "section_title", "text"} <= set(mention)
        assert "Calculus II" in mention["text"]

    relations = {(r["direction"], r["label"], r["other"]["id"]): r for r in body["relations"]}
    assert ("in", "prerequisite_of", "calculus i") in relations  # Calculus I -> Calculus II
    assert ("out", "requires", "calculus i") in relations  # Calculus II -> Calculus I
    assert ("in", "teaches", "ada lovelace") in relations
    evidence = relations[("out", "requires", "calculus i")]["evidence"]
    assert evidence["document_name"] == "courses.md" and "requires Calculus I" in evidence["text"]
    assert db.scalar(select(func.count()).select_from(KgMention)) > 0


def test_a_relations_quote_and_evidence_are_the_text_it_was_read_from(
    client: TestClient, admin: Headers, docs: dict
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    body = client.get("/api/graph/entity", headers=admin, params={"key": "registrar"}).json()
    (approves,) = [r for r in body["relations"] if r["label"] == "approves"]
    assert approves["quote"] == "The Registrar approves the Credit Transfer Policy"
    assert approves["quote"] in approves["evidence"]["text"]


def test_an_unknown_entity_is_a_404(client: TestClient, admin: Headers, docs: dict) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    assert (
        client.get("/api/graph/entity", headers=admin, params={"key": "nobody"}).status_code == 404
    )


def test_the_view_can_be_filtered_by_type_and_name_and_is_capped(
    client: TestClient, admin: Headers, docs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)

    courses = get_graph(client, admin, types="course").json()
    assert set(nodes(courses)) == {"calculus i", "calculus ii"}
    assert [(e["source"], e["target"]) for e in courses["edges"]] == [("calculus i", "calculus ii")]

    found = get_graph(client, admin, q="registrar").json()  # the match and its neighbour
    assert set(nodes(found)) == {"registrar", "credit transfer policy"}

    monkeypatch.setattr(get_settings(), "kg_view_max_nodes", 2)
    capped = get_graph(client, admin).json()
    assert len(capped["nodes"]) == 2 and capped["total_nodes"] == 4 and capped["truncated"] is True
    kept = {n["id"] for n in capped["nodes"]}
    assert all(e["source"] in kept and e["target"] in kept for e in capped["edges"])


# --- Rebuilding, failing, queueing ------------------------------------------------------------------------


def test_a_rebuild_replaces_the_graph_instead_of_adding_to_it(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    view = build(client, admin, docs["handbook"], ([ent("Registrar", "role")], []))
    assert set(nodes(view)) == {"registrar"} and view["edges"] == []
    assert db.scalar(select(func.count()).select_from(KgEntity)) == 1
    assert db.scalar(select(func.count()).select_from(KgRelation)) == 0


def test_a_failed_build_says_why_and_can_be_retried(
    client: TestClient, admin: Headers, docs: dict, fake_llm: FakeChatModel
) -> None:
    fake_llm.reply("not json").reply("still not json")
    client.post(f"/api/documents/{docs['handbook']['id']}/graph", headers=admin)
    view = wait_built(client, admin, docs["handbook"]["id"])
    (status,) = view["documents"]
    assert status["status"] == "failed" and status["error"] and view["nodes"] == []

    retried = build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    assert retried["documents"][0]["status"] == "ready" and len(retried["nodes"]) == 4


def test_a_build_is_measured_once(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    (event,) = db.scalars(select(QueryEvent).where(QueryEvent.kind == "graph")).all()
    assert event.outcome == "answered" and event.total_tokens > 0
    assert event.timings["entities"] == 4 and event.document_ids == [docs["handbook"]["id"]]


def test_a_build_already_under_way_is_not_queued_twice(
    client: TestClient, admin: Headers, docs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued: list[Any] = []
    monkeypatch.setattr(
        analysis_service, "submit_background", lambda job, *args: queued.append(args)
    )
    url = f"/api/documents/{docs['handbook']['id']}/graph"
    assert client.post(url, headers=admin).status_code == 202
    assert client.post(url, headers=admin).status_code == 202
    assert len(queued) == 1
    view = get_graph(client, admin, document_ids=docs["handbook"]["id"]).json()
    assert view["documents"][0]["status"] == "pending"


def test_a_build_a_restart_cut_off_is_failed(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    analysis_service.begin_insight(uuid.UUID(docs["handbook"]["id"]), InsightKind.GRAPH.value, "")
    analysis_service.recover_interrupted(db)
    view = get_graph(client, admin, document_ids=docs["handbook"]["id"]).json()
    assert view["documents"][0]["status"] == "failed"


def test_the_graph_is_built_after_upload_when_that_is_switched_on(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "auto_knowledge_graph", True)
    ingestion.on_indexed(graph_service.submit_graph)
    script(fake_llm, *HANDBOOK_GRAPH)
    document = one(client, admin, "handbook.md", HANDBOOK, department="Academic")
    view = wait_built(client, admin, document["id"])
    assert view["documents"][0]["status"] == "ready" and len(view["nodes"]) == 4


def test_nothing_is_built_after_upload_when_that_is_switched_off(
    client: TestClient, admin: Headers, fake_llm: FakeChatModel
) -> None:
    ingestion.on_indexed(graph_service.submit_graph)  # AUTO_KNOWLEDGE_GRAPH is false in tests
    document = one(client, admin, "handbook.md", HANDBOOK, department="Academic")
    assert (
        get_graph(client, admin, document_ids=document["id"]).json()["documents"][0]["status"]
        == "none"
    )
    assert fake_llm.calls == []


def test_a_document_that_is_not_indexed_cannot_be_built(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    db.execute(
        text("UPDATE documents SET status = 'queued' WHERE id = CAST(:id AS uuid)"),
        {"id": docs["handbook"]["id"]},
    )
    db.commit()
    assert (
        client.post(f"/api/documents/{docs['handbook']['id']}/graph", headers=admin).status_code
        == 409
    )
    view = get_graph(client, admin).json()  # and it contributes nothing to the whole-library view
    assert docs["handbook"]["id"] not in {d["id"] for d in view["documents"]}


def test_the_whole_feature_can_be_switched_off(
    client: TestClient, admin: Headers, docs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "knowledge_graph", False)
    assert get_graph(client, admin).status_code == 404
    assert client.get("/api/graph/entity", headers=admin, params={"key": "x"}).status_code == 404
    assert (
        client.post(f"/api/documents/{docs['handbook']['id']}/graph", headers=admin).status_code
        == 404
    )


# --- Who may see what (CLAUDE.md rules 6, 24, 32) --------------------------------------------------------


def test_a_user_never_sees_the_graph_of_a_restricted_document(
    client: TestClient, admin: Headers, user: Headers, docs: dict
) -> None:
    build(client, admin, docs["bonus"], BONUS_GRAPH)
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    restricted = docs["bonus"]["id"]

    assert "compensation committee" in nodes(get_graph(client, admin).json())

    everything = get_graph(client, user).json()  # no documents named: only what a user may read
    assert "compensation committee" not in nodes(everything)
    assert restricted not in {d["id"] for d in everything["documents"]}
    assert "registrar" in nodes(everything)

    assert get_graph(client, user, document_ids=restricted).status_code == 404
    mixed = get_graph(client, user, document_ids=f"{docs['handbook']['id']},{restricted}")
    assert mixed.status_code == 404  # one unreadable document is a 404, not a partial answer

    leak = client.get("/api/graph/entity", headers=user, params={"key": "compensation committee"})
    assert leak.status_code == 404
    assert (
        client.get(
            "/api/graph/entity",
            headers=user,
            params={"key": "compensation committee", "document_ids": restricted},
        ).status_code
        == 404
    )
    assert client.post(f"/api/documents/{restricted}/graph", headers=user).status_code == 404


def test_a_restricted_passage_never_reaches_a_user_through_a_shared_entity(
    client: TestClient, admin: Headers, user: Headers, docs: dict
) -> None:
    """Both documents name the Registrar; the user may read one of them only."""
    secret = one(
        client,
        admin,
        "memo.md",
        b"# Memo\n\nThe Registrar was told the Salary Freeze starts in May.\n",
        department="HR",
        access_level="Restricted",
    )
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    build(
        client,
        admin,
        secret,
        (
            [ent("Registrar", "role"), ent("Salary Freeze", "event")],
            [rel("Registrar", "Salary Freeze", "informed_about")],
        ),
    )

    body = client.get("/api/graph/entity", headers=user, params={"key": "registrar"}).json()
    assert {d["name"] for d in body["documents"]} == {"handbook.md"}
    texts = " ".join(m["text"] for m in body["mentions"]) + " ".join(
        r["evidence"]["text"] for r in body["relations"]
    )
    assert "Salary Freeze" not in texts
    assert {r["other"]["id"] for r in body["relations"]} == {"credit transfer policy"}

    as_admin = client.get("/api/graph/entity", headers=admin, params={"key": "registrar"}).json()
    assert {d["name"] for d in as_admin["documents"]} == {"handbook.md", "memo.md"}


def test_deleting_a_document_removes_its_part_of_the_graph(
    client: TestClient, admin: Headers, docs: dict, db: Session
) -> None:
    build(client, admin, docs["handbook"], HANDBOOK_GRAPH)
    build(client, admin, docs["courses"], COURSES_GRAPH)
    assert (
        client.delete(f"/api/documents/{docs['courses']['id']}", headers=admin).status_code == 204
    )

    view = get_graph(client, admin).json()
    assert "ada lovelace" not in nodes(view)  # only in the deleted document
    assert set(nodes(view)["registrar"]["document_ids"]) == {docs["handbook"]["id"]}
    assert (
        db.scalar(select(func.count()).select_from(KgEntity).where(KgEntity.name == "Ada Lovelace"))
        == 0
    )

    assert (
        client.delete(f"/api/documents/{docs['handbook']['id']}", headers=admin).status_code == 204
    )
    for model in (KgEntity, KgMention, KgRelation):
        assert db.scalar(select(func.count()).select_from(model)) == 0
