"""Startup work that keeps Postgres, Chroma and the BM25 index consistent.

PostgreSQL is the source of truth; Chroma and BM25 are derived (CLAUDE.md rule 3).
``run_startup`` therefore:

1. opens the Chroma collection and verifies it matches the configured embedding model
   and dimension (a mismatch refuses to start, TDD 3.3);
2. fails documents whose ingestion a restart interrupted and re-queues the ones that
   never started;
3. refuses to start if Postgres holds indexed chunks but Chroma holds no vectors, which
   is what a changed ``EMBEDDING_MODEL``/``EMBEDDING_DIM`` or a lost Chroma volume looks
   like: serving questions from an empty index would silently return nothing. The
   remedy is ``python -m app.maintenance reembed``;
4. rebuilds the BM25 index from Postgres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.constants import DocumentStatus
from app.core import bm25, vectorstore
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document
from app.services import analysis_service, cache_service, ingestion, metrics_service

logger = logging.getLogger(__name__)

_LOAD_BATCH = 1000
# Verifying means one query per stored vector (batched). Beyond this many vectors startup
# skips it and leaves the check to `python -m app.maintenance verify-index`.
MAX_STARTUP_VERIFY_VECTORS = 20_000


class IndexInconsistentError(RuntimeError):
    """Postgres has indexed chunks that the vector store does not have."""


@dataclass(frozen=True)
class StartupReport:
    """What startup found and did (counts only)."""

    collection: str
    indexed_chunks: int
    vectors: int
    requeued: int
    repaired: int = 0


def indexed_chunk_count(db: Session) -> int:
    """Number of chunks belonging to fully indexed documents."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.INDEXED.value)
        )
        or 0
    )


def load_bm25(db: Session) -> int:
    """Rebuild the process-wide BM25 index from Postgres. Returns the chunk count."""
    rows = db.execute(
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.department,
            Chunk.document_type,
            Chunk.creation_date_int,
            Chunk.access_level,
            Chunk.section_title,
            Chunk.content,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == DocumentStatus.INDEXED.value)
        .execution_options(yield_per=_LOAD_BATCH)
    )
    bm25.get_index().rebuild(
        (
            bm25.ChunkMeta(
                chunk_id=row.id,
                document_id=str(row.document_id),
                department=row.department,
                document_type=row.document_type,
                creation_date_int=row.creation_date_int,
                access_level=row.access_level,
            ),
            bm25.keyword_text(row.section_title, row.content),
        )
        for row in rows
    )
    return len(bm25.get_index())


def run_startup() -> StartupReport:
    """Verify and repair the derived indexes, then start background ingestion."""
    from app.config import get_settings

    collection = get_settings().collection_name
    # First of all: opening the collection would create an empty one if a crash during a rebuild
    # had left the real collection missing and its verified copy beside it.
    vectorstore.recover_interrupted_rebuild()
    vectorstore.get_collection()  # raises VectorStoreMismatchError on a model/dimension mismatch

    with get_sessionmaker()() as db:
        queued = ingestion.recover_interrupted(db)
        analysis_service.recover_interrupted(db)  # summaries a restart cut off
        metrics_service.prune(db)  # dashboard measurements past METRICS_RETENTION_DAYS
        indexed = indexed_chunk_count(db)
        vectors = vectorstore.count()

        if indexed > 0 and vectors == 0:
            raise IndexInconsistentError(
                f"PostgreSQL holds {indexed} indexed chunks but the Chroma collection '{collection}' is "
                "empty. This happens after changing EMBEDDING_MODEL/EMBEDDING_DIM or losing the Chroma "
                "volume. Run `python -m app.maintenance reembed` (see README), or restore the old settings."
            )
        if vectors < indexed:
            # Some chunks have no vector, so those passages can only be found by keyword search.
            # Not a reason to refuse to start (the rest works), but never a quiet warning;
            # /api/ready reports it too.
            logger.error(
                "some indexed chunks have no vector: run `python -m app.maintenance reembed`",
                extra={"indexed_chunks": indexed, "vectors": vectors, "missing": indexed - vectors},
            )
        elif vectors > indexed:
            logger.warning(
                "more vectors than indexed chunks (leftovers of deleted documents)",
                extra={"indexed_chunks": indexed, "vectors": vectors},
            )
        loaded = load_bm25(db)
        repaired = 0
        if 0 < vectors <= MAX_STARTUP_VERIFY_VECTORS:
            unreachable = vectorstore.find_unreachable()
            if unreachable:
                logger.warning(
                    "vectors stored but unreachable by search; repairing",
                    extra={"unreachable": len(unreachable), "vectors": vectors},
                )
                repaired = vectorstore.repair_unreachable(unreachable)
        elif vectors > MAX_STARTUP_VERIFY_VECTORS:
            logger.info(
                "startup index verification skipped (large collection)", extra={"vectors": vectors}
            )

    ingestion.start_workers()
    # Registered first: a new document must stop stale saved answers before anything else runs.
    ingestion.on_indexed(cache_service.on_document_indexed)
    ingestion.on_indexed(analysis_service.submit_auto_summary)  # executive summary after upload
    for document_id in queued:
        ingestion.submit(document_id)

    report = StartupReport(collection, indexed, vectors, len(queued), repaired)
    logger.info(
        "index ready",
        extra={
            "collection": collection,
            "indexed_chunks": indexed,
            "vectors": vectors,
            "bm25_chunks": loaded,
            "requeued": len(queued),
            "repaired": repaired,
        },
    )
    return report
