"""Hybrid retrieval: embed the question, search both channels under one filter, fuse, hydrate.

Dense (Chroma) and keyword (BM25) search run in parallel under the same ``SearchFilter`` and are
merged by RRF. Candidates are then loaded from PostgreSQL, the source of truth: a chunk only
counts if its document is ``indexed`` and still passes the filter, so a half-deleted document or a
stale vector can never surface (CLAUDE.md rules 3, 6 and 24).

Retrieval never answers "nothing found" because the index is broken: if PostgreSQL knows of
chunks the filter allows but a channel returns nothing, that is an ``IndexUnavailableError``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import DocumentStatus
from app.core import bm25, embeddings, fusion, vectorstore
from app.core.filters import SearchFilter
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)

T = TypeVar("T")

INDEX_UNAVAILABLE_MESSAGE = (
    "The search index is unavailable or out of date, so your question cannot be answered right "
    "now. Please try again later or contact your administrator."
)

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieve")


class IndexUnavailableError(Exception):
    """The vector or keyword index cannot be used. The message is safe to show to the user."""

    def __init__(self, message: str = INDEX_UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Candidate:
    """A retrieved chunk with its document details and scores."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int
    page_end: int
    section_title: str
    department: str
    document_type: str
    access_level: str
    # Dense cosine similarity; None when only the keyword channel found the chunk.
    similarity: float | None
    keyword_score: float | None
    rrf_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """The chunks to answer from, plus what the guard needs and the per-stage timings (ms)."""

    candidates: list[Candidate]
    best_similarity: float | None
    top_keyword_text: str | None
    timings: dict[str, float]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _timed(func_: Callable[..., T], *args: Any) -> tuple[T, float]:
    started = time.perf_counter()
    result = func_(*args)
    return result, _ms(started)


def _dense(vector: list[float], n: int, search_filter: SearchFilter) -> list[vectorstore.DenseHit]:
    try:
        return vectorstore.query(vector, n, search_filter.to_chroma())
    except vectorstore.VectorStoreError as exc:
        logger.warning("dense search failed", extra={"error": type(exc).__name__})
        raise IndexUnavailableError from exc


def _keyword(question: str, n: int, search_filter: SearchFilter) -> list[bm25.KeywordHit]:
    return bm25.get_index().search(question, n, allow=search_filter.allows)


def count_matching_chunks(db: Session, search_filter: SearchFilter) -> int:
    """How many chunks of indexed documents the filter allows, according to PostgreSQL."""
    return int(
        db.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.status == DocumentStatus.INDEXED.value,
                *search_filter.sql_conditions(Chunk),
            )
        )
        or 0
    )


def _load(db: Session, ids: list[str], search_filter: SearchFilter) -> dict[str, Any]:
    """Rows for the given chunk ids: indexed documents only, filter applied once more."""
    if not ids:
        return {}
    rows = db.execute(
        select(
            Chunk.id,
            Chunk.document_id,
            Document.name,
            Chunk.content,
            Chunk.page_number,
            Chunk.page_end,
            Chunk.section_title,
            Chunk.department,
            Chunk.document_type,
            Chunk.access_level,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.id.in_(ids),
            Document.status == DocumentStatus.INDEXED.value,
            *search_filter.sql_conditions(Chunk),
        )
    ).all()
    return {row.id: row for row in rows}


def retrieve(question: str, search_filter: SearchFilter) -> RetrievalResult:
    """Find the chunks to answer ``question`` from. Blocking; call it from a worker thread."""
    settings = get_settings()
    timings: dict[str, float] = {}
    if search_filter.matches_nothing:
        return RetrievalResult([], None, None, timings)

    started = time.perf_counter()
    vector = embeddings.embed_query(question, interactive=True)
    timings["embed_ms"] = _ms(started)

    n = settings.retrieval_candidates
    dense_future = _pool.submit(_timed, _dense, vector, n, search_filter)
    keyword_hits, timings["keyword_ms"] = _timed(_keyword, question, n, search_filter)
    dense_hits, timings["dense_ms"] = dense_future.result()

    with get_sessionmaker()() as db:
        # A channel that returned nothing is only legitimate if nothing is searchable at all.
        channel_empty = not dense_hits or len(bm25.get_index()) == 0
        if channel_empty and count_matching_chunks(db, search_filter) > 0:
            logger.error(
                "index unhealthy: chunks exist but a channel returned nothing",
                extra={"dense_hits": len(dense_hits), "bm25_chunks": len(bm25.get_index())},
            )
            raise IndexUnavailableError

        started = time.perf_counter()
        fused = fusion.rrf(
            [[h.chunk_id for h in dense_hits], [h.chunk_id for h in keyword_hits]], settings.rrf_k
        )
        timings["fusion_ms"] = _ms(started)

        started = time.perf_counter()
        rows = _load(db, [chunk_id for chunk_id, _ in fused], search_filter)
        timings["hydrate_ms"] = _ms(started)

    similarity = {h.chunk_id: h.similarity for h in dense_hits}
    keyword_score = {h.chunk_id: h.score for h in keyword_hits}
    candidates: list[Candidate] = []
    for chunk_id, score in fused:
        row = rows.get(chunk_id)
        if row is None:
            continue
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                document_id=str(row.document_id),
                document_name=row.name,
                text=row.content,
                page_number=row.page_number,
                page_end=row.page_end,
                section_title=row.section_title,
                department=row.department,
                document_type=row.document_type,
                access_level=row.access_level,
                similarity=similarity.get(chunk_id),
                keyword_score=keyword_score.get(chunk_id),
                rrf_score=score,
            )
        )
        if len(candidates) == settings.retrieval_top_k:
            break

    best = max((c.similarity for c in candidates if c.similarity is not None), default=None)
    top_keyword_text = next(
        (rows[h.chunk_id].content for h in keyword_hits if h.chunk_id in rows), None
    )
    return RetrievalResult(candidates, best, top_keyword_text, timings)
