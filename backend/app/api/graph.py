"""The knowledge graph: entities and relationships across documents, linked to their source chunks.

Every route goes through ``graph_service``, which applies the analysis access rule: only documents
the caller may read and that are indexed contribute (a document above their level is a 404, never
a 403), and the passages an entity links to are filtered by their access levels.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from app.config import get_settings
from app.deps import CurrentUser, DbSession
from app.schemas import EntityDetailOut, GraphBuildOut, GraphViewOut
from app.services import graph_service
from app.services.analysis_service import DocumentNotReady
from app.services.document_service import DocumentNotFound

router = APIRouter(tags=["graph"])

MAX_DOCUMENTS = 50
MAX_TYPES = 12
MAX_QUERY_CHARS = 100


def _parse_ids(raw: str | None) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    for part in (raw or "").split(","):
        if part.strip():
            try:
                ids.append(uuid.UUID(part.strip()))
            except ValueError:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, "document_ids must be document ids"
                ) from None
    if len(set(ids)) > MAX_DOCUMENTS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Choose at most {MAX_DOCUMENTS} documents."
        )
    return list(dict.fromkeys(ids))


def _parse_types(raw: str | None) -> list[str]:
    types = [t.strip().casefold() for t in (raw or "").split(",") if t.strip()]
    if len(types) > MAX_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Too many entity types.")
    return types


def _disabled() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "The knowledge graph is switched off.")


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")


@router.get("/graph", response_model=GraphViewOut, summary="Knowledge graph of documents")
def get_graph(
    user: CurrentUser,
    db: DbSession,
    document_ids: str | None = Query(default=None, description="Comma-separated document ids"),
    types: str | None = Query(default=None, description="Comma-separated entity types"),
    q: str = Query(default="", max_length=MAX_QUERY_CHARS, description="Entity name contains"),
) -> GraphViewOut:
    """Nodes (entities merged by name across the documents) and edges (relations) of the chosen
    documents, or of every document the caller may read."""
    ids = _parse_ids(document_ids)
    try:
        view = graph_service.build_view(db, user, ids, _parse_types(types), q)
    except graph_service.GraphDisabled:
        raise _disabled() from None
    except DocumentNotFound:
        raise _not_found() from None
    return GraphViewOut.model_validate(view)


@router.get(
    "/graph/entity", response_model=EntityDetailOut, summary="One entity of the knowledge graph"
)
def get_entity(
    user: CurrentUser,
    db: DbSession,
    key: str = Query(min_length=1, max_length=200, description="The entity's node id"),
    document_ids: str | None = Query(default=None, description="Comma-separated document ids"),
) -> EntityDetailOut:
    """Where an entity is mentioned (passages that open in the source drawer) and its relations."""
    ids = _parse_ids(document_ids)
    try:
        detail = graph_service.entity_detail(db, user, key, ids)
    except graph_service.GraphDisabled:
        raise _disabled() from None
    except DocumentNotFound:
        raise _not_found() from None
    except graph_service.EntityNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Entity not found") from None
    return EntityDetailOut.model_validate(detail)


@router.post(
    "/documents/{document_id}/graph",
    response_model=GraphBuildOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Build (or rebuild) a document's knowledge graph",
)
def build_graph(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> GraphBuildOut:
    """Queue the model run that reads the document and writes its graph. Costs about one model
    call per ``KG_BATCH_CHARS`` of text; poll ``GET /graph`` for its status."""
    if not get_settings().knowledge_graph:
        raise _disabled()
    try:
        graph_service.request_build(db, user, document_id)
    except DocumentNotFound:
        raise _not_found() from None
    except DocumentNotReady:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "The document is not indexed yet. Try again when it is ready."
        ) from None
    return GraphBuildOut(status="pending")
