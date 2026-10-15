"""The knowledge graph: building it from a document, and reading it back for a reader.

* **Building** (``submit_graph`` after upload, ``request_build`` on demand) runs on the insights
  thread, one job at a time like the executive summary, so the model quota is shared and ingestion
  never waits. The status lives in ``document_insights`` (kind ``graph``: pending before the model
  is called, then ready or failed) and the graph itself in the ``kg_*`` tables, written in one short
  transaction after the model finished. No database session is held across a model call (rule 25a).
* **Reading** goes through the same gate as analysis (rule 32): only documents the caller may read
  and that are ``indexed`` contribute, a document above their level is a 404, and the passages an
  entity links to are filtered by the caller's access levels once more.
* **Deletion** (rules 24 and 33): every ``kg_*`` row cascades from its document and its chunks, so
  there is nothing to scrub.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import (
    DocumentStatus,
    EventKind,
    EventOutcome,
    InsightKind,
    InsightStatus,
    allowed_access_levels,
)
from app.core import graph, llm
from app.core.analysis import AnalysisError
from app.core.filters import build_filter
from app.db.database import get_sessionmaker
from app.db.models import (
    Chunk,
    Document,
    DocumentInsight,
    KgEntity,
    KgMention,
    KgRelation,
    User,
)
from app.services import analysis_service, cache_service, document_service, metrics_service
from app.services.analysis_service import DocumentNotReady

logger = logging.getLogger(__name__)

KIND = InsightKind.GRAPH.value
MAX_ENTITY_MENTIONS_SHOWN = 50
MAX_ENTITY_RELATIONS_SHOWN = 100
NOT_FOUND_STATUS = "none"  # a document whose graph was never requested


class GraphDisabled(Exception):
    """``KNOWLEDGE_GRAPH=false``: nothing is built or shown."""


class EntityNotFound(Exception):
    """No entity with that key in the documents the caller may read."""


# --- Building ---------------------------------------------------------------------------------------


def submit_graph(document_id: uuid.UUID) -> None:
    """After-indexing hook: queue the graph of a freshly indexed document (never raises)."""
    settings = get_settings()
    if not (settings.knowledge_graph and settings.auto_knowledge_graph):
        return
    try:
        _queue(document_id)
    except Exception as exc:  # never let this disturb the document that was just indexed
        logger.warning("could not queue the knowledge graph", extra={"error": type(exc).__name__})


def _queue(document_id: uuid.UUID) -> None:
    analysis_service.begin_insight(document_id, KIND, "")  # visible as "pending" at once
    analysis_service.submit_background(_run_graph, document_id)


def request_build(db: Session, user: User, document_id: uuid.UUID) -> str:
    """Queue (re)building the graph of a document the caller may read. Returns ``pending``.

    ``DocumentNotFound`` for a document above their level, ``DocumentNotReady`` when it is not
    indexed. A build already under way is not started a second time.
    """
    if not get_settings().knowledge_graph:
        raise GraphDisabled
    document = document_service.get_document(db, user, document_id)
    if document.status != DocumentStatus.INDEXED.value:
        raise DocumentNotReady
    saved = db.scalar(
        select(DocumentInsight).where(
            DocumentInsight.document_id == document.id,
            DocumentInsight.kind == KIND,
            DocumentInsight.params_key == "",
        )
    )
    if analysis_service.fresh_pending(saved):
        return InsightStatus.PENDING.value
    _queue(document.id)
    return InsightStatus.PENDING.value


def _store(document_id: uuid.UUID, result: graph.GraphResult) -> None:
    """Replace the document's graph in one transaction (its old rows cascade away first)."""
    with get_sessionmaker()() as db:
        document = db.get(Document, document_id)
        if document is None or document.status != DocumentStatus.INDEXED.value:
            raise _Gone
        db.execute(delete(KgEntity).where(KgEntity.document_id == document_id))
        ids = {entity.key: uuid.uuid4() for entity in result.graph.entities}
        if ids:
            db.execute(
                insert(KgEntity),
                [
                    {
                        "id": ids[e.key],
                        "document_id": document_id,
                        "name": e.name,
                        "key": e.key,
                        "type": e.type,
                        "description": e.description,
                    }
                    for e in result.graph.entities
                ],
            )
            db.execute(
                insert(KgMention),
                [
                    {"entity_id": ids[e.key], "chunk_id": chunk_id}
                    for e in result.graph.entities
                    for chunk_id in e.chunk_ids
                ],
            )
        if result.graph.relations:
            db.execute(
                insert(KgRelation),
                [
                    {
                        "document_id": document_id,
                        "source_id": ids[r.subject],
                        "target_id": ids[r.object],
                        "label": r.label,
                        "chunk_id": r.chunk_id,
                        "quote": r.quote,
                    }
                    for r in result.graph.relations
                ],
            )
        db.commit()


class _Gone(Exception):
    """The document was deleted (or taken out of search) while its graph was being built."""


def _run_graph(document_id: uuid.UUID) -> None:
    """Read a document with the model and store its graph (insights thread)."""
    event = analysis_service.new_event(EventKind.GRAPH, None, [document_id])
    started = time.perf_counter()
    try:
        with get_sessionmaker()() as db:
            document = db.get(Document, document_id)
            if document is None or document.status != DocumentStatus.INDEXED.value:
                return  # deleted or re-queued meanwhile
            sources = analysis_service.load_sources(db, document)
        result = llm.run_background(graph.extract_graph(sources))
        metrics_service.add_usage(event, result.usage)
        _store(document_id, result)
        stats = {
            "entities": len(result.graph.entities),
            "relations": len(result.graph.relations),
            "batches": result.batches,
            "coverage": result.coverage,
        }
        event["timings"].update(stats)
        if get_settings().graph_retrieval_active:
            # Answers are widened through the graph, so a new graph can change what they say. Done
            # before the graph is reported ready: nobody should see "ready" next to a stale answer.
            cache_service.invalidate_document(document_id)
        analysis_service.finish_insight(document_id, KIND, "", content=stats)
        logger.info("knowledge graph built", extra={"document_id": str(document_id), **stats})
    except _Gone:
        logger.info("graph dropped: document gone", extra={"document_id": str(document_id)})
        return
    except Exception as exc:
        event.update(
            outcome=EventOutcome.FAILED.value, error_code=analysis_service.failure_code(exc)
        )
        message = (
            exc.message
            if isinstance(exc, (llm.LlmError, AnalysisError))
            else "The knowledge graph could not be built."
        )
        logger.warning(
            "knowledge graph failed",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
        try:
            analysis_service.finish_insight(document_id, KIND, "", error=message)
        except Exception:  # the document may have been deleted: nothing to record
            logger.debug("no graph row to fail", extra={"document_id": str(document_id)})
    event["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    metrics_service.record(event)


# --- Reading ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scope:
    documents: list[Document]  # indexed and readable
    skipped: list[Document]  # readable but not indexed (yet)


def _scope(db: Session, user: User, document_ids: Sequence[uuid.UUID] | None) -> _Scope:
    """The documents a graph request covers. Named documents must be readable (``DocumentNotFound``);
    without names, every readable one."""
    if document_ids:
        found = [document_service.get_document(db, user, d) for d in dict.fromkeys(document_ids)]
    else:
        levels = [level.value for level in allowed_access_levels(user.role)]
        found = list(
            db.scalars(
                select(Document)
                .where(Document.access_level.in_(levels))
                .order_by(Document.created_at.desc())
            )
        )
    ready = [d for d in found if d.status == DocumentStatus.INDEXED.value]
    skipped = [d for d in found if d.status != DocumentStatus.INDEXED.value] if document_ids else []
    return _Scope(ready, skipped)


def _statuses(db: Session, documents: Sequence[Document]) -> list[dict[str, Any]]:
    """Per document: how its graph is doing (``none`` / ``pending`` / ``ready`` / ``failed``)."""
    rows = {
        row.document_id: row
        for row in db.scalars(
            select(DocumentInsight).where(
                DocumentInsight.document_id.in_([d.id for d in documents]),
                DocumentInsight.kind == KIND,
            )
        )
    }
    result = []
    for document in documents:
        row = rows.get(document.id)
        status = NOT_FOUND_STATUS if row is None else row.status
        if status == InsightStatus.PENDING.value and not analysis_service.fresh_pending(row):
            status = InsightStatus.FAILED.value  # its run died without saying so
        content = (row.content or {}) if row is not None else {}
        result.append(
            {
                "id": document.id,
                "name": document.name,
                "status": status,
                "entities": int(content.get("entities", 0)),
                "relations": int(content.get("relations", 0)),
                "coverage": content.get("coverage"),
                "error": row.error_message if row is not None else None,
            }
        )
    return result


def build_view(
    db: Session,
    user: User,
    document_ids: Sequence[uuid.UUID] | None,
    types: Sequence[str] = (),
    query: str = "",
) -> dict[str, Any]:
    """The graph of the chosen documents (or all the caller may read) as nodes and edges.

    Entities with the same normalised name in different documents are one node, which is what
    ties documents together. Nodes are ranked by connections and then by mentions and capped at
    ``KG_VIEW_MAX_NODES``; ``total_nodes`` says how many matched before the cap.
    """
    if not get_settings().knowledge_graph:
        raise GraphDisabled
    scope = _scope(db, user, document_ids)
    ids = [d.id for d in scope.documents]
    statuses = _statuses(db, scope.documents + scope.skipped)
    if not ids:
        return {
            "documents": statuses,
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "truncated": False,
        }

    # One node per key: the richest description, the most common type, the documents it is in.
    grouped: dict[str, dict[str, Any]] = {}
    rows = db.execute(
        select(
            KgEntity.key, KgEntity.name, KgEntity.type, KgEntity.description, KgEntity.document_id
        ).where(KgEntity.document_id.in_(ids))
    )
    for key, name, type_, description, document_id in rows:
        node = grouped.setdefault(
            key,
            {
                "id": key,
                "name": name,
                "description": "",
                "types": defaultdict(int),
                "documents": [],
            },
        )
        node["types"][type_] += 1
        if len(description) > len(node["description"]):
            node["description"], node["name"] = description, name
        if str(document_id) not in node["documents"]:
            node["documents"].append(str(document_id))
    mention_counts = dict(
        db.execute(
            select(KgEntity.key, func.count(KgMention.chunk_id))
            .join(KgMention, KgMention.entity_id == KgEntity.id)
            .where(KgEntity.document_id.in_(ids))
            .group_by(KgEntity.key)
        ).all()
    )

    source, target = KgEntity.__table__.alias("src"), KgEntity.__table__.alias("tgt")
    edge_rows = db.execute(
        select(source.c.key, target.c.key, KgRelation.label, func.count())
        .join(source, source.c.id == KgRelation.source_id)
        .join(target, target.c.id == KgRelation.target_id)
        .where(KgRelation.document_id.in_(ids))
        .group_by(source.c.key, target.c.key, KgRelation.label)
    ).all()

    wanted_types = {t for t in types if t}
    needle = graph.squash(query)
    nodes = {
        key: node
        for key, node in grouped.items()
        if (not wanted_types or max(node["types"], key=node["types"].get) in wanted_types)
    }
    edges = [e for e in edge_rows if e[0] in nodes and e[1] in nodes]
    if needle:
        matched = {key for key, node in nodes.items() if needle in graph.squash(node["name"])}
        nearby = {k for s, t, _, _ in edges for k in (s, t) if s in matched or t in matched}
        keep = matched | nearby
        nodes = {k: n for k, n in nodes.items() if k in keep}
        edges = [e for e in edges if e[0] in nodes and e[1] in nodes]

    degree: dict[str, int] = defaultdict(int)
    for s, t, _, count in edges:
        degree[s] += int(count)
        degree[t] += int(count)
    ranked = sorted(nodes, key=lambda k: (-degree[k], -int(mention_counts.get(k, 0)), k))
    limit = get_settings().kg_view_max_nodes
    kept = set(ranked[:limit])
    return {
        "documents": statuses,
        "nodes": [
            {
                "id": key,
                "name": nodes[key]["name"],
                "type": max(nodes[key]["types"], key=nodes[key]["types"].get),
                "description": nodes[key]["description"],
                "document_ids": nodes[key]["documents"],
                "mentions": int(mention_counts.get(key, 0)),
                "degree": degree[key],
            }
            for key in ranked[:limit]
        ],
        "edges": [
            {"source": s, "target": t, "label": label, "count": int(count)}
            for s, t, label, count in edges
            if s in kept and t in kept
        ],
        "total_nodes": len(nodes),
        "truncated": len(nodes) > limit,
    }


def _citation(n: int, row: Any) -> dict[str, Any]:
    """A passage in the shape the source drawer opens (the same as a chat citation)."""
    return {
        "n": n,
        "chunk_id": row.chunk_id,
        "document_id": str(row.document_id),
        "document_name": row.document_name,
        "page_number": row.page_number,
        "page_end": row.page_end,
        "section_title": row.section_title,
        "text": row.content,
    }


def entity_detail(
    db: Session, user: User, key: str, document_ids: Sequence[uuid.UUID] | None
) -> dict[str, Any]:
    """Everything the graph says about one entity: where it is mentioned and how it is related.

    Passages come with their text so they open in the source drawer; only indexed documents the
    caller may read are used (``EntityNotFound`` when none holds the entity).
    """
    if not get_settings().knowledge_graph:
        raise GraphDisabled
    scope = _scope(db, user, document_ids)
    ids = [d.id for d in scope.documents]
    entities = (
        db.scalars(select(KgEntity).where(KgEntity.document_id.in_(ids), KgEntity.key == key)).all()
        if ids
        else []
    )
    if not entities:
        raise EntityNotFound
    allowed = build_filter(user).sql_conditions(Chunk)  # the reader's access levels, once more
    names = {d.id: d.name for d in scope.documents}

    mentions = db.execute(
        select(
            Chunk.id.label("chunk_id"),
            Chunk.document_id,
            Document.name.label("document_name"),
            Chunk.page_number,
            Chunk.page_end,
            Chunk.section_title,
            Chunk.content,
        )
        .join(KgMention, KgMention.chunk_id == Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            KgMention.entity_id.in_([e.id for e in entities]),
            Document.status == DocumentStatus.INDEXED.value,
            *allowed,
        )
        .order_by(Document.name, Chunk.ordinal)
        .limit(MAX_ENTITY_MENTIONS_SHOWN)
    ).all()

    other = KgEntity.__table__.alias("other")
    own_ids = [e.id for e in entities]
    relations: list[dict[str, Any]] = []
    for direction, near, far in (
        ("out", KgRelation.source_id, KgRelation.target_id),
        ("in", KgRelation.target_id, KgRelation.source_id),
    ):
        rows = db.execute(
            select(
                KgRelation.label,
                KgRelation.quote,
                other.c.key,
                other.c.name,
                other.c.type,
                Chunk.id.label("chunk_id"),
                Chunk.document_id,
                Document.name.label("document_name"),
                Chunk.page_number,
                Chunk.page_end,
                Chunk.section_title,
                Chunk.content,
            )
            .join(other, other.c.id == far)
            .join(Chunk, Chunk.id == KgRelation.chunk_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                near.in_(own_ids),
                Document.status == DocumentStatus.INDEXED.value,
                *allowed,
            )
            .order_by(KgRelation.label, other.c.name)
            .limit(MAX_ENTITY_RELATIONS_SHOWN)
        ).all()
        for row in rows:
            relations.append(
                {
                    "label": row.label,
                    "direction": direction,
                    "quote": row.quote,
                    "other": {"id": row.key, "name": row.name, "type": row.type},
                    "evidence": _citation(0, row),
                }
            )
    best = max(entities, key=lambda e: len(e.description))
    types: dict[str, int] = defaultdict(int)
    for entity in entities:
        types[entity.type] += 1
    return {
        "id": key,
        "name": best.name,
        "type": max(types, key=types.get),
        "description": best.description,
        "documents": [
            {"id": e.document_id, "name": names.get(e.document_id, "")} for e in entities
        ],
        "mentions": [_citation(i, row) for i, row in enumerate(mentions, start=1)],
        "relations": [
            {**rel, "evidence": {**rel["evidence"], "n": i}}
            for i, rel in enumerate(relations, start=len(mentions) + 1)
        ],
    }
