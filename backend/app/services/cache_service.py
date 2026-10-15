"""Keeping the semantic answer cache honest when documents change (CLAUDE.md rules 24 and 38).

The cache (``core/answer_cache.py``) knows nothing about Postgres. This module looks up the
metadata of a document that was indexed, deleted or re-graphed and tells the cache to drop every
entry that change can make wrong. A failure here is logged and never fails the upload or delete
that triggered it: the cache is also re-validated on every hit (``chat_service``), and the
fallback when the metadata cannot be read is to drop everything.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from app.core import answer_cache
from app.core.bm25 import ChunkMeta
from app.db.database import get_sessionmaker
from app.db.models import Chunk

logger = logging.getLogger(__name__)


def _document_meta(document_id: uuid.UUID) -> ChunkMeta | None:
    """The filterable metadata of a document (every chunk carries the same copy)."""
    with get_sessionmaker()() as db:
        row = db.execute(
            select(
                Chunk.department,
                Chunk.document_type,
                Chunk.creation_date_int,
                Chunk.access_level,
            )
            .where(Chunk.document_id == document_id)
            .limit(1)
        ).first()
    if row is None:
        return None
    return ChunkMeta(
        chunk_id="",
        document_id=str(document_id),
        department=row.department,
        document_type=row.document_type,
        creation_date_int=row.creation_date_int,
        access_level=row.access_level,
    )


def invalidate_document(document_id: uuid.UUID) -> None:
    """Drop the cached answers a change to this document can make wrong. Never raises."""
    try:
        dropped = answer_cache.get_cache().invalidate_document(
            str(document_id), _document_meta(document_id)
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the cache's own safeguards
        # The metadata could not be read: drop everything rather than keep a possibly stale answer.
        logger.warning("answer cache invalidation fell back", extra={"error": type(exc).__name__})
        answer_cache.get_cache().clear()
        return
    if dropped:
        logger.info("cached answers dropped", extra={"count": dropped})


def on_document_indexed(document_id: uuid.UUID) -> None:
    """After-indexing hook (``ingestion.on_indexed``): a new document changes what is answerable."""
    invalidate_document(document_id)


def evict_message(message_id: uuid.UUID) -> None:
    """A message was rated down: stop serving the answer it carries. Never raises."""
    try:
        answer_cache.get_cache().evict_message(str(message_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer cache eviction failed", extra={"error": type(exc).__name__})
