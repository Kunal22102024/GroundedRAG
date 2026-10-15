"""Background ingestion: parse -> chunk -> metadata -> Postgres -> embed -> Chroma -> BM25.

State machine (TDD 3.4): ``queued -> processing -> indexed | failed``.

Rules this module enforces:

* Chunk rows are written to PostgreSQL **before** anything reaches Chroma (CLAUDE.md
  rule 3), so Postgres always knows what Chroma should contain.
* Any failure removes the document's partial chunks from Postgres, Chroma and the BM25
  index and marks it ``failed`` with a user-safe message: no half-indexed document is
  ever searchable.
* One document is processed at a time by default (``INGESTION_WORKERS``), which keeps
  embedding API usage predictable. A per-document lock stops ingestion and deletion of
  the same document from interleaving.
* Logs carry document ids, counts and timings only, never names or text (rule 21).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.constants import DocumentStatus
from app.core import bm25, embeddings, metadata, parsers, vectorstore
from app.core.chunking import ChunkDraft, StructureAwareChunker
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = (
    "Indexing was interrupted by a restart. Delete the document and upload it again."
)

# --- Per-document locks and the worker pool -----------------------------------------------

_locks: dict[uuid.UUID, threading.Lock] = {}
_locks_guard = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_executor_guard = threading.Lock()


def document_lock(document_id: uuid.UUID) -> threading.Lock:
    """The lock that serializes ingestion and deletion of one document."""
    with _locks_guard:
        return _locks.setdefault(document_id, threading.Lock())


def start_workers() -> None:
    """Create the worker pool (idempotent)."""
    global _executor
    with _executor_guard:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=get_settings().ingestion_workers, thread_name_prefix="ingest"
            )


def shutdown_workers() -> None:
    """Stop accepting work. A running ingestion is abandoned and recovered at next startup."""
    global _executor
    with _executor_guard:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None


_indexed_hooks: list[Callable[[uuid.UUID], None]] = []


def on_indexed(hook: Callable[[uuid.UUID], None]) -> None:
    """Call ``hook(document_id)`` after each document becomes ``indexed`` (idempotent).

    A hook runs on the ingestion thread, so it must only queue work, never do it.
    """
    if hook not in _indexed_hooks:
        _indexed_hooks.append(hook)


def submit(document_id: uuid.UUID) -> None:
    """Queue a document for background ingestion."""
    start_workers()
    assert _executor is not None
    _executor.submit(_run_safely, document_id)


def _trace(exc: BaseException) -> str:
    """Where an exception happened (file:line:function, innermost first) without its message.

    Library error messages can quote the text being processed, and rule 21 says document text is
    never logged; the call sites are enough to debug from.
    """
    frames = traceback.extract_tb(exc.__traceback__)[-6:]
    return " <- ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in reversed(frames))


def _run_safely(document_id: uuid.UUID) -> None:
    try:
        ingest_document(document_id)
    except Exception as exc:  # last resort: a worker thread must never die silently
        logger.error(
            "unexpected ingestion error",
            extra={
                "document_id": str(document_id),
                "error": type(exc).__name__,
                "trace": _trace(exc),
            },
        )


# --- Helpers ---------------------------------------------------------------------------------


def chunk_id_for(document_id: uuid.UUID, ordinal: int) -> str:
    """Stable chunk id, also used as the Chroma id (e.g. ``<uuid>_chunk_0014``)."""
    return f"{document_id}_chunk_{ordinal:04d}"


def chroma_metadata(
    document_id: uuid.UUID,
    document_name: str,
    resolved: metadata.ResolvedMetadata,
    draft: ChunkDraft,
) -> dict[str, str | int]:
    """Flat Chroma metadata for one chunk of a freshly resolved document."""
    return vectorstore.chunk_metadata(
        document_id=str(document_id),
        document_name=document_name,
        department=resolved.department,
        document_type=resolved.document_type,
        creation_date=resolved.creation_date,
        creation_date_int=resolved.creation_date_int,
        access_level=resolved.access_level,
        page_number=draft.page_number,
        page_end=draft.page_end,
        section_title=draft.section_title,
    )


def user_message(exc: BaseException) -> str:
    """A message safe to show the uploader; details stay in the logs."""
    if isinstance(exc, (parsers.ParseError, embeddings.EmbeddingError)):
        return str(exc)  # these classes are written for end users and never hold text
    if isinstance(exc, vectorstore.VectorStoreError):
        return "The search index is unavailable. Try uploading the document again later."
    return "Indexing failed unexpectedly. See the server logs for details."


# --- The job ----------------------------------------------------------------------------------


def ingest_document(document_id: uuid.UUID) -> None:
    """Index one queued document. Never raises for document-level problems: they mark it failed."""
    lock = document_lock(document_id)
    if not lock.acquire(blocking=False):
        logger.info("ingestion skipped: document is busy", extra={"document_id": str(document_id)})
        return
    try:
        _ingest(document_id)
    finally:
        lock.release()


def _ingest(document_id: uuid.UUID) -> None:
    settings = get_settings()
    sessions = get_sessionmaker()
    started = time.perf_counter()
    timings: dict[str, float] = {}

    with sessions() as db:
        document = db.get(Document, document_id)
        if document is None or document.status != DocumentStatus.QUEUED.value:
            return  # deleted, or already handled
        document.status = DocumentStatus.PROCESSING.value
        document.error_message = None
        db.commit()
        path = Path(settings.upload_dir) / document.stored_path
        extension = path.suffix
        name = document.name
        explicit = metadata.ExplicitMetadata(
            department=document.department or None,
            document_type=document.document_type or None,
            creation_date=document.creation_date,
            access_level=document.access_level,
        )

    try:
        stage = time.perf_counter()
        parsed = parsers.parse_document(path, extension)
        timings["parse_ms"] = _ms(stage)
        if sum(len(segment.text) for segment in parsed.segments) > settings.max_document_chars:
            raise parsers.ParseError(
                f"The document has more than {settings.max_document_chars:,} characters of text, "
                "which is more than can be indexed. Split it into smaller files."
            )

        stage = time.perf_counter()
        drafts = StructureAwareChunker(
            settings.chunk_target_chars, settings.chunk_overlap_chars
        ).chunk(parsed.segments)
        if not drafts:
            raise parsers.EmptyDocumentError("The document contains no extractable text.")
        resolved = metadata.resolve(explicit, name, parsed.properties, date.today())
        timings["chunk_ms"] = _ms(stage)

        stage = time.perf_counter()
        _store_chunks(document_id, resolved, drafts, sessions)  # Postgres first (rule 3)
        timings["db_ms"] = _ms(stage)

        stage = time.perf_counter()
        texts = [d.content for d in drafts]
        vectors = embeddings.embed_documents(texts)
        timings["embed_ms"] = _ms(stage)

        stage = time.perf_counter()
        chunk_ids = [chunk_id_for(document_id, d.ordinal) for d in drafts]
        vectorstore.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[chroma_metadata(document_id, name, resolved, d) for d in drafts],
        )
        bm25.get_index().add_document(
            (
                bm25.ChunkMeta(
                    chunk_id=chunk_id_for(document_id, d.ordinal),
                    document_id=str(document_id),
                    department=resolved.department,
                    document_type=resolved.document_type,
                    creation_date_int=resolved.creation_date_int,
                    access_level=resolved.access_level,
                ),
                bm25.keyword_text(d.section_title, d.content),
            )
            for d in drafts
        )
        timings["index_ms"] = _ms(stage)

        # Chroma can store a vector its index never returns (see core/vectorstore.py). Prove every
        # new chunk is searchable before calling the document indexed, repairing any that are not.
        stage = time.perf_counter()
        repaired = vectorstore.ensure_reachable(chunk_ids)
        timings["verify_ms"] = _ms(stage)

        with sessions() as db:
            updated = db.execute(
                update(Document)
                .where(
                    Document.id == document_id, Document.status == DocumentStatus.PROCESSING.value
                )
                .values(
                    status=DocumentStatus.INDEXED.value, chunk_count=len(drafts), error_message=None
                )
            )
            db.commit()
        if updated.rowcount != 1:
            raise RuntimeError("document disappeared while it was being indexed")
    except Exception as exc:
        _fail(document_id, exc, sessions)
        return

    timings["total_ms"] = _ms(started)
    logger.info(
        "document indexed",
        extra={
            "document_id": str(document_id),
            "pages": parsed.page_count,
            "chunks": len(drafts),
            "unique_texts": len(set(texts)),
            "repaired_vectors": repaired,
            **timings,
            **{f"parse_{k}": v for k, v in parsed.stats.items()},
        },
    )
    for hook in tuple(_indexed_hooks):  # extras (the executive summary) never disturb the upload
        try:
            hook(document_id)
        except Exception as exc:
            logger.warning("after-indexing hook failed", extra={"error": type(exc).__name__})


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _store_chunks(
    document_id: uuid.UUID,
    resolved: metadata.ResolvedMetadata,
    drafts: list[ChunkDraft],
    sessions: sessionmaker[Session],
) -> None:
    """Persist the resolved metadata and every chunk row in one transaction."""
    with sessions() as db:
        db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                department=resolved.department,
                document_type=resolved.document_type,
                creation_date=resolved.creation_date,
                access_level=resolved.access_level,
            )
        )
        db.execute(
            insert(Chunk),
            [
                {
                    "id": chunk_id_for(document_id, d.ordinal),
                    "document_id": document_id,
                    "ordinal": d.ordinal,
                    "content": d.content,
                    "page_number": d.page_number,
                    "page_end": d.page_end,
                    "section_title": d.section_title,
                    "department": resolved.department,
                    "document_type": resolved.document_type,
                    "creation_date_int": resolved.creation_date_int,
                    "access_level": resolved.access_level,
                }
                for d in drafts
            ],
        )
        db.commit()


def _fail(document_id: uuid.UUID, exc: BaseException, sessions: sessionmaker[Session]) -> None:
    """Mark a document failed and remove everything it left behind, in every store."""
    message = user_message(exc)
    expected = isinstance(
        exc, (parsers.ParseError, embeddings.EmbeddingError, vectorstore.VectorStoreError)
    )
    extra = {"document_id": str(document_id), "error": type(exc).__name__}
    if not expected:
        extra["trace"] = _trace(exc)
    logger.log(logging.WARNING if expected else logging.ERROR, "ingestion failed", extra=extra)
    cleanup_document(document_id, sessions)
    with sessions() as db:
        db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.FAILED.value, error_message=message, chunk_count=0)
        )
        db.commit()


def cleanup_document(document_id: uuid.UUID, sessions: sessionmaker[Session] | None = None) -> None:
    """Remove a document's chunks from Postgres, Chroma and the BM25 index (idempotent)."""
    factory = sessions or get_sessionmaker()
    bm25.get_index().remove_document(str(document_id))
    try:
        vectorstore.delete_document(str(document_id))
    except Exception as exc:  # Chroma may be down; Postgres below is the source of truth
        logger.error(
            "could not remove vectors",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
    with factory() as db:
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.commit()


# --- Startup recovery -------------------------------------------------------------------------


def recover_interrupted(db: Session) -> list[uuid.UUID]:
    """Fail documents whose ingestion a restart interrupted; return the ones still queued.

    ``processing`` documents are cleaned up and marked ``failed`` (re-queueing could loop
    forever on a document that crashes the process). ``queued`` ones never started and
    are simply returned so the caller can submit them again.
    """
    stuck = list(
        db.scalars(select(Document.id).where(Document.status == DocumentStatus.PROCESSING.value))
    )
    for document_id in stuck:
        cleanup_document(document_id)
        db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.FAILED.value, error_message=INTERRUPTED_MESSAGE, chunk_count=0
            )
        )
        logger.warning(
            "interrupted ingestion marked failed", extra={"document_id": str(document_id)}
        )
    db.commit()
    return list(
        db.scalars(select(Document.id).where(Document.status == DocumentStatus.QUEUED.value))
    )
