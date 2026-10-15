"""Analysis workflows: saved summaries and extractions, comparison across documents.

* **Access.** A document is analysed only if the caller may read it (``get_document``: a document
  above their level is a 404, never a 403) and it is ``indexed``. Comparison checks every document.
* **Saved results.** A summary or an extraction is stored in ``document_insights`` and reused until
  the caller asks for a refresh, so a document costs one model call per kind, not one per visit.
  The row is written ``pending`` before the model is called, so a restart can fail what it cut off.
* **No connection across slow work** (CLAUDE.md rule 25a): every database step opens its own short
  session; the model is called with none open.
* **Deletion** (rule 24): a result that cites a deleted document is deleted with it
  (``document_service._scrub_citations``), and a result shown later hides passages the reader
  may no longer see.
* The streaming workflows yield ``SseEvent`` s like a chat turn, so one reader on the client and
  one error mapping serve all of them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
from app.core import analysis, generation, guard, llm, retrieval
from app.core.analysis import AnalysisError
from app.core.filters import build_filter
from app.core.retrieval import Candidate
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document, DocumentInsight, User
from app.services import conversation_service, document_service, metrics_service
from app.services.chat_service import SseEvent

logger = logging.getLogger(__name__)

T = TypeVar("T")

# A result still ``pending`` after this long belongs to a run that died; it is regenerated.
PENDING_FRESH = timedelta(minutes=10)
POLL_SECONDS = 1.5
WAIT_FOR_PENDING_SECONDS = 300
INTERRUPTED_MESSAGE = "Generation was interrupted by a restart. Generate it again."
CANCELLED_MESSAGE = "Generation was cancelled."


class DocumentNotReady(Exception):
    """The document exists but is not ``indexed`` (yet), so it cannot be analysed."""


@dataclass(frozen=True)
class DocRef:
    """A document as the workflows need it."""

    id: uuid.UUID
    name: str


# --- Loading what to analyse --------------------------------------------------------------------


def _sources(db: Session, document: Document) -> list[Candidate]:
    """Every chunk of a document, in reading order, as sources."""
    rows = db.scalars(
        select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)
    ).all()
    return [
        Candidate(
            chunk_id=row.id,
            document_id=str(document.id),
            document_name=document.name,
            text=row.content,
            page_number=row.page_number,
            page_end=row.page_end,
            section_title=row.section_title,
            department=row.department,
            document_type=row.document_type,
            access_level=row.access_level,
            similarity=None,
            keyword_score=None,
            rrf_score=0.0,
        )
        for row in rows
    ]


def _readable_document(db: Session, user: User, document_id: uuid.UUID) -> Document:
    """A document the user may read and that is indexed; ``DocumentNotFound`` / ``DocumentNotReady``."""
    document = document_service.get_document(db, user, document_id)
    if document.status != DocumentStatus.INDEXED.value:
        raise DocumentNotReady
    return document


# --- Saved results ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Saved:
    """A stored analysis as the workflows see it."""

    status: str
    content: dict[str, Any] | None
    error_message: str | None
    updated_at: datetime


def _row(db: Session, document_id: uuid.UUID, kind: str, params_key: str) -> DocumentInsight | None:
    return db.scalar(
        select(DocumentInsight).where(
            DocumentInsight.document_id == document_id,
            DocumentInsight.kind == kind,
            DocumentInsight.params_key == params_key,
        )
    )


def _begin(document_id: uuid.UUID, kind: str, params_key: str) -> None:
    """Mark an analysis ``pending`` (replacing any earlier result of the same kind)."""
    now = datetime.now(UTC)
    statement = pg_insert(DocumentInsight).values(
        document_id=document_id,
        kind=kind,
        params_key=params_key,
        status=InsightStatus.PENDING.value,
        content=None,
        error_message=None,
        model=get_settings().llm_model,
    )
    with get_sessionmaker()() as db:
        db.execute(
            statement.on_conflict_do_update(
                index_elements=["document_id", "kind", "params_key"],
                set_={
                    "status": statement.excluded.status,
                    "content": None,
                    "error_message": None,
                    "model": statement.excluded.model,
                    "updated_at": now,
                },
            )
        )
        db.commit()


def _finish(
    document_id: uuid.UUID,
    kind: str,
    params_key: str,
    *,
    content: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with get_sessionmaker()() as db:
        db.execute(
            update(DocumentInsight)
            .where(
                DocumentInsight.document_id == document_id,
                DocumentInsight.kind == kind,
                DocumentInsight.params_key == params_key,
            )
            .values(
                status=InsightStatus.FAILED.value if error else InsightStatus.READY.value,
                content=None if error else content,
                error_message=error,
                updated_at=datetime.now(UTC),
            )
        )
        db.commit()


def recover_interrupted(db: Session) -> int:
    """Fail analyses a restart cut off (startup); returns how many."""
    result = db.execute(
        update(DocumentInsight)
        .where(DocumentInsight.status == InsightStatus.PENDING.value)
        .values(status=InsightStatus.FAILED.value, error_message=INTERRUPTED_MESSAGE)
    )
    db.commit()
    return int(result.rowcount or 0)


def _fresh_pending(saved: DocumentInsight | None) -> bool:
    return (
        saved is not None
        and saved.status == InsightStatus.PENDING.value
        and datetime.now(UTC) - saved.updated_at < PENDING_FRESH
    )


def _scrub(db: Session, user: User, content: dict[str, Any] | None) -> dict[str, Any] | None:
    """A saved result with the passages this reader may no longer see made unavailable."""
    if not content:
        return content
    citations = content.get("citations") or []
    readable = conversation_service.readable_chunk_ids(
        db, user, {str(c["chunk_id"]) for c in citations if c.get("chunk_id")}
    )
    return {**content, "citations": conversation_service.hide_unreadable(citations, readable)}


def list_insights(db: Session, user: User, document_id: uuid.UUID) -> list[dict[str, Any]]:
    """The saved analyses of a document the user may read (``DocumentNotFound`` otherwise)."""
    document = document_service.get_document(db, user, document_id)
    rows = db.scalars(
        select(DocumentInsight)
        .where(DocumentInsight.document_id == document.id)
        .order_by(DocumentInsight.kind, DocumentInsight.updated_at.desc())
    ).all()
    return [
        {
            "id": row.id,
            "kind": row.kind,
            "params_key": row.params_key,
            "status": row.status,
            "content": _scrub(db, user, row.content),
            "error_message": row.error_message,
            "updated_at": row.updated_at,
        }
        for row in rows
    ]


def executive_statuses(db: Session, document_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
    """``none`` / ``pending`` / ``ready`` / ``failed`` for each document's executive summary."""
    if not document_ids:
        return {}
    rows = db.execute(
        select(DocumentInsight.document_id, DocumentInsight.status).where(
            DocumentInsight.document_id.in_(document_ids),
            DocumentInsight.kind == InsightKind.EXECUTIVE.value,
        )
    )
    return {document_id: status for document_id, status in rows}


# --- Running a model call while telling the client what is happening ----------------------------


@dataclass(frozen=True)
class _Done:
    value: Any


async def _drive(
    work: Callable[[analysis.OnWait, analysis.OnProgress], Awaitable[T]],
) -> AsyncIterator[SseEvent | _Done]:
    """Run ``work`` and yield the ``status`` events it reports, then its result as ``_Done``.

    The model call can wait out a rate limit or read a long document in parts; those become
    ``status`` events so the screen can say so. If the consumer goes away (the client closed the
    stream) the work is cancelled instead of running on unseen.
    """
    events: asyncio.Queue[SseEvent] = asyncio.Queue()

    def on_wait(seconds: float) -> None:
        events.put_nowait(SseEvent("status", {"stage": "rate_limited", "retry_in": round(seconds)}))

    def on_progress(detail: str) -> None:
        events.put_nowait(SseEvent("status", {"stage": "working", "detail": detail}))

    task = asyncio.ensure_future(work(on_wait, on_progress))
    try:
        while True:
            getter = asyncio.ensure_future(events.get())
            await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                yield getter.result()
                continue
            getter.cancel()
            break
        while not events.empty():
            yield events.get_nowait()
        yield _Done(task.result())
    finally:
        if not task.done():
            task.cancel()


# --- Measuring a run for the dashboard -----------------------------------------------------------


def _event(
    kind: EventKind, user: User | None, document_ids: Sequence[uuid.UUID], **fields: Any
) -> dict[str, Any]:
    return metrics_service.new_event(
        kind,
        None if user is None else user.id,
        document_ids=[str(d) for d in document_ids],
        **fields,
    )


def _failure_code(exc: BaseException) -> str:
    return str(getattr(exc, "code", "")) or "internal"


async def _measured(
    event: dict[str, Any], stream: AsyncIterator[SseEvent]
) -> AsyncIterator[SseEvent]:
    """Relay ``stream`` and record one measurement when it ends: answered (a saved result is
    marked ``cached`` and costs nothing), or failed with its error code. A client that went away
    (cancellation) is not recorded."""
    started = time.perf_counter()
    try:
        async for item in stream:
            if item.event == "done" and item.data.get("cached"):
                event["timings"]["cached"] = True
            yield item
    except Exception as exc:
        event.update(outcome=EventOutcome.FAILED.value, error_code=_failure_code(exc))
        event["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        await asyncio.to_thread(metrics_service.record, event)
        raise
    event["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    await asyncio.to_thread(metrics_service.record, event)


# --- Summaries ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryPlan:
    """Everything resolved before streaming starts (access, sources, what is already saved)."""

    document: DocRef
    kind: str
    params_key: str
    sources: list[Candidate]
    earlier: list[Candidate] | None
    refresh: bool
    saved: Saved | None


def _saved(db: Session, user: User, document_id: uuid.UUID, kind: str, key: str) -> Saved | None:
    row = _row(db, document_id, kind, key)
    if row is None:
        return None
    fresh_pending = _fresh_pending(row)
    status = row.status
    if status == InsightStatus.PENDING.value and not fresh_pending:
        status = InsightStatus.FAILED.value  # a run that died: treated as failed
    return Saved(status, _scrub(db, user, row.content), row.error_message, row.updated_at)


def prepare_summary(
    user: User, document_id: uuid.UUID, kind: str, compare_to: uuid.UUID | None, refresh: bool
) -> SummaryPlan:
    """Check access and load the sources. ``DocumentNotFound`` / ``DocumentNotReady`` / ``AnalysisError``."""
    if kind not in analysis.SUMMARY_KINDS:
        raise AnalysisError("bad_kind", "Unknown summary type.")
    with get_sessionmaker()() as db:
        document = _readable_document(db, user, document_id)
        ref = DocRef(document.id, document.name)
        sources = _sources(db, document)
        earlier = None
        key = ""
        if compare_to is not None and kind == InsightKind.KEY_CHANGES.value:
            if compare_to == document_id:
                raise AnalysisError("bad_request", "Choose a different document to compare with.")
            older = _readable_document(db, user, compare_to)
            earlier = _sources(db, older)
            key = str(older.id)
        return SummaryPlan(
            ref, kind, key, sources, earlier, refresh, _saved(db, user, ref.id, kind, key)
        )


async def _await_pending(
    plan_user: User, ref: DocRef, kind: str, key: str
) -> AsyncIterator[SseEvent | Saved]:
    """Wait (polling) for a run already in progress, e.g. the automatic summary after upload."""
    waited = 0.0
    while waited < WAIT_FOR_PENDING_SECONDS:
        yield SseEvent("status", {"stage": "working", "detail": "Already being generated"})
        await asyncio.sleep(POLL_SECONDS)
        waited += POLL_SECONDS

        def look() -> Saved | None:
            with get_sessionmaker()() as db:
                return _saved(db, plan_user, ref.id, kind, key)

        saved = await asyncio.to_thread(look)
        if saved is not None and saved.status != InsightStatus.PENDING.value:
            yield saved
            return
    yield Saved(
        InsightStatus.FAILED.value, None, "It is taking too long. Try again.", datetime.now(UTC)
    )


def run_summary(plan: SummaryPlan, user: User) -> AsyncIterator[SseEvent]:
    """Stream a summary: ``status`` events, then ``done`` with the content (or ``error``)."""
    ids = [plan.document.id] + ([uuid.UUID(plan.params_key)] if plan.params_key else [])
    event = _event(EventKind.SUMMARY, user, ids)
    event["timings"]["summary_kind"] = plan.kind
    return _measured(event, _summary_stream(plan, user, event))


async def _summary_stream(
    plan: SummaryPlan, user: User, event: dict[str, Any]
) -> AsyncIterator[SseEvent]:
    ref, kind, key = plan.document, plan.kind, plan.params_key
    saved = plan.saved
    if saved is not None and not plan.refresh:
        if saved.status == InsightStatus.READY.value and saved.content:
            yield _result(kind, saved.content, cached=True)
            return
        if saved.status == InsightStatus.PENDING.value:  # someone else is generating it
            async for item in _await_pending(user, ref, kind, key):
                if isinstance(item, SseEvent):
                    yield item
                elif item.status == InsightStatus.READY.value and item.content:
                    yield _result(kind, item.content, cached=True)
                    return
                else:
                    raise llm.LlmError("failed", item.error_message or "Generation failed.")
            return

    yield SseEvent("status", {"stage": "working", "detail": "Reading the document"})
    await asyncio.to_thread(_begin, ref.id, kind, key)

    async def work(on_wait: analysis.OnWait, on_progress: analysis.OnProgress) -> dict[str, Any]:
        content, usage = await analysis.summarize(
            kind, plan.sources, earlier=plan.earlier, on_wait=on_wait, on_progress=on_progress
        )
        metrics_service.add_usage(event, usage)
        logger.info("summary generated", extra={"kind": kind, "document_id": str(ref.id), **usage})
        return content

    async for item in _run_and_store(work, ref.id, kind, key):
        yield item


async def _run_and_store(
    work: Callable[[analysis.OnWait, analysis.OnProgress], Awaitable[dict[str, Any]]],
    document_id: uuid.UUID,
    kind: str,
    key: str,
) -> AsyncIterator[SseEvent]:
    """Run ``work``, relaying its status events; store the result (or the failure) and emit it."""
    try:
        result: dict[str, Any] | None = None
        async for item in _drive(work):
            if isinstance(item, _Done):
                result = item.value
            else:
                yield item
        assert result is not None
    except asyncio.CancelledError:
        _finish(
            document_id, kind, key, error=CANCELLED_MESSAGE
        )  # short and blocking: we are leaving
        raise
    except Exception as exc:
        message = (
            exc.message if isinstance(exc, (llm.LlmError, AnalysisError)) else "Generation failed."
        )
        _finish(document_id, kind, key, error=message)
        raise
    await asyncio.to_thread(_finish, document_id, kind, key, content=result)
    yield _result(kind, result, cached=False)


def _result(kind: str, content: dict[str, Any], *, cached: bool) -> SseEvent:
    return SseEvent("done", {"kind": kind, "content": content, "cached": cached})


# --- Automatic executive summary after upload ----------------------------------------------------

_executor: ThreadPoolExecutor | None = None


def _insight_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # One at a time: the model quota is shared with chat, and ingestion must never wait on this.
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insights")
    return _executor


def shutdown() -> None:
    """Stop accepting work (shutdown); a running job is abandoned and failed at the next start."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def submit_auto_summary(document_id: uuid.UUID) -> None:
    """Queue the executive summary of a freshly indexed document (called by ingestion)."""
    if not get_settings().auto_executive_summary:
        return
    try:
        _begin(document_id, InsightKind.EXECUTIVE.value, "")  # visible as "generating" at once
        _insight_executor().submit(_run_auto, document_id)
    except Exception as exc:  # never let this disturb the document that was just indexed
        logger.warning("could not queue the executive summary", extra={"error": type(exc).__name__})


def _run_auto(document_id: uuid.UUID) -> None:
    kind = InsightKind.EXECUTIVE.value
    event = _event(EventKind.SUMMARY, None, [document_id])
    event["timings"].update(summary_kind=kind, automatic=True)
    started = time.perf_counter()
    try:
        with get_sessionmaker()() as db:
            document = db.get(Document, document_id)
            if document is None or document.status != DocumentStatus.INDEXED.value:
                return  # deleted or re-queued meanwhile
            sources = _sources(db, document)
        content, usage = asyncio.run(analysis.summarize(kind, sources))
        metrics_service.add_usage(event, usage)
        _finish(document_id, kind, "", content=content)
        logger.info("executive summary generated", extra={"document_id": str(document_id), **usage})
    except Exception as exc:
        event.update(outcome=EventOutcome.FAILED.value, error_code=_failure_code(exc))
        message = (
            exc.message if isinstance(exc, (llm.LlmError, AnalysisError)) else "Generation failed."
        )
        logger.warning(
            "executive summary failed",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
        try:
            _finish(document_id, kind, "", error=message)
        except Exception:  # the document may have been deleted: nothing to record
            logger.debug("no summary row to fail", extra={"document_id": str(document_id)})
    event["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    metrics_service.record(event)


# --- Suggested questions for a document -----------------------------------------------------------

# A set that failed is not asked for again before this, so a broken model is not hammered.
RETRY_FAILED_AFTER = timedelta(minutes=10)
UNFILTERED_DOCUMENTS = 4  # without a selection: saved sets of this many recent documents


def _queue_questions(document_id: uuid.UUID) -> None:
    _begin(document_id, InsightKind.QUESTIONS.value, "")  # "pending" at once: no double queueing
    _insight_executor().submit(_run_questions, document_id)


def _run_questions(document_id: uuid.UUID) -> None:
    """Write and save the suggested questions of one document (insights thread)."""
    kind = InsightKind.QUESTIONS.value
    event = _event(EventKind.SUGGEST, None, [document_id])
    started = time.perf_counter()
    try:
        with get_sessionmaker()() as db:
            document = db.get(Document, document_id)
            if document is None or document.status != DocumentStatus.INDEXED.value:
                return
            sources = _sources(db, document)
        content, usage = asyncio.run(
            analysis.suggest_questions(sources, get_settings().suggestions_per_document)
        )
        metrics_service.add_usage(event, usage)
        _finish(document_id, kind, "", content=content)
    except Exception as exc:
        event.update(outcome=EventOutcome.FAILED.value, error_code=_failure_code(exc))
        logger.warning(
            "suggested questions failed",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
        try:
            _finish(document_id, kind, "", error="Suggested questions could not be written.")
        except Exception:
            logger.debug("no questions row to fail", extra={"document_id": str(document_id)})
    event["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    metrics_service.record(event)


def document_suggestions(
    user: User, document_ids: Sequence[uuid.UUID]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Suggested questions of the chosen documents: ``(questions, pending document ids)``.

    Each document must be readable (``DocumentNotFound`` otherwise); one that is not indexed is
    skipped. A document without a saved set has one written in the background (one model call,
    once) and is reported as pending. With no documents chosen, the saved sets of a few recently
    indexed readable documents are used and nothing is generated.
    """
    kind = InsightKind.QUESTIONS.value
    to_queue: list[uuid.UUID] = []
    pending: list[str] = []
    sets: list[tuple[Document, list[dict[str, Any]]]] = []
    with get_sessionmaker()() as db:
        if document_ids:
            documents = [
                document_service.get_document(db, user, d) for d in dict.fromkeys(document_ids)
            ]
            documents = [d for d in documents if d.status == DocumentStatus.INDEXED.value]
        else:
            levels = [level.value for level in allowed_access_levels(user.role)]
            documents = list(
                db.scalars(
                    select(Document)
                    .join(DocumentInsight, DocumentInsight.document_id == Document.id)
                    .where(
                        Document.status == DocumentStatus.INDEXED.value,
                        Document.access_level.in_(levels),
                        DocumentInsight.kind == kind,
                        DocumentInsight.status == InsightStatus.READY.value,
                    )
                    .order_by(Document.created_at.desc())
                    .limit(UNFILTERED_DOCUMENTS)
                )
            )
        now = datetime.now(UTC)
        for document in documents:
            row = _row(db, document.id, kind, "")
            if row is not None and row.status == InsightStatus.READY.value and row.content:
                sets.append((document, list(row.content.get("items") or [])))
            elif _fresh_pending(row):
                pending.append(str(document.id))
            elif document_ids and (
                row is None
                or now - row.updated_at >= RETRY_FAILED_AFTER
                or row.status == InsightStatus.PENDING.value  # a run that died
            ):
                to_queue.append(document.id)
                pending.append(str(document.id))
    for document_id in to_queue:
        try:
            _queue_questions(document_id)
        except Exception as exc:
            logger.warning(
                "could not queue suggested questions", extra={"error": type(exc).__name__}
            )
    # Take turns between documents so each one is represented near the top.
    questions: list[dict[str, Any]] = []
    for position in range(max((len(items) for _, items in sets), default=0)):
        for document, items in sets:
            if position < len(items):
                questions.append(
                    {
                        "question": items[position]["question"],
                        "document_id": str(document.id),
                        "document_name": document.name,
                    }
                )
    return questions, pending


# --- Structured extraction -----------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractPlan:
    document: DocRef
    specs: list[analysis.FieldSpec]
    params_key: str
    sources: list[Candidate]
    refresh: bool
    saved: Saved | None


def prepare_extract(
    user: User, document_id: uuid.UUID, fields: Sequence[str], refresh: bool
) -> ExtractPlan:
    """Check access, normalize the fields, load the sources."""
    specs = analysis.normalize_fields(fields)
    key = analysis.fields_key(specs)
    with get_sessionmaker()() as db:
        document = _readable_document(db, user, document_id)
        ref = DocRef(document.id, document.name)
        return ExtractPlan(
            ref, specs, key, _sources(db, document), refresh,
            _saved(db, user, ref.id, InsightKind.EXTRACTION.value, key),
        )  # fmt: skip


def run_extract(plan: ExtractPlan, user: User) -> AsyncIterator[SseEvent]:
    """Stream an extraction: ``status`` events, then ``done`` with the rows (or ``error``)."""
    event = _event(EventKind.EXTRACT, user, [plan.document.id])
    event["timings"]["fields"] = len(plan.specs)
    return _measured(event, _extract_stream(plan, user, event))


async def _extract_stream(
    plan: ExtractPlan, user: User, event: dict[str, Any]
) -> AsyncIterator[SseEvent]:
    ref, kind, key = plan.document, InsightKind.EXTRACTION.value, plan.params_key
    saved = plan.saved
    if saved is not None and not plan.refresh:
        if saved.status == InsightStatus.READY.value and saved.content:
            yield _result(kind, saved.content, cached=True)
            return
        if saved.status == InsightStatus.PENDING.value:
            async for item in _await_pending(user, ref, kind, key):
                if isinstance(item, SseEvent):
                    yield item
                elif item.status == InsightStatus.READY.value and item.content:
                    yield _result(kind, item.content, cached=True)
                    return
                else:
                    raise llm.LlmError("failed", item.error_message or "Extraction failed.")
            return

    yield SseEvent("status", {"stage": "working", "detail": "Reading the document"})
    await asyncio.to_thread(_begin, ref.id, kind, key)

    async def work(on_wait: analysis.OnWait, on_progress: analysis.OnProgress) -> dict[str, Any]:
        content, usage = await analysis.extract(
            plan.specs, plan.sources, on_wait=on_wait, on_progress=on_progress
        )
        metrics_service.add_usage(event, usage)
        logger.info(
            "extraction done",
            extra={"document_id": str(ref.id), "fields": len(plan.specs), **usage},
        )
        return content

    async for item in _run_and_store(work, ref.id, kind, key):
        yield item


# --- Comparing documents ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparePlan:
    question: str
    documents: list[DocRef]
    user: User


def prepare_compare(user: User, question: str, document_ids: Sequence[uuid.UUID]) -> ComparePlan:
    """Check access to every document. ``DocumentNotFound`` for any one the user may not read."""
    distinct = list(dict.fromkeys(document_ids))
    limit = get_settings().compare_max_documents
    if len(distinct) < 2:
        raise AnalysisError("bad_request", "Choose at least two different documents to compare.")
    if len(distinct) > limit:
        raise AnalysisError("bad_request", f"Compare at most {limit} documents at once.")
    with get_sessionmaker()() as db:
        documents = []
        for document_id in distinct:
            document = _readable_document(db, user, document_id)
            documents.append(DocRef(document.id, document.name))
    return ComparePlan(question.strip(), documents, user)


def run_compare(plan: ComparePlan) -> AsyncIterator[SseEvent]:
    """One cited answer per document, then the points on which they disagree.

    Each document is searched on its own (the search filter is narrowed to it), so a column can
    only cite its own text. A document that does not cover the question is reported as such
    without a model call. Columns run one after another: the quota is shared and a person is
    reading them in order anyway.
    """
    event = _event(
        EventKind.COMPARE, plan.user, [d.id for d in plan.documents], question=plan.question
    )
    return _measured(event, _compare_stream(plan, event))


async def _compare_stream(plan: ComparePlan, event: dict[str, Any]) -> AsyncIterator[SseEvent]:
    settings = get_settings()
    base = build_filter(plan.user, None)
    answered: list[tuple[DocRef, list[dict[str, Any]], list[Candidate]]] = []
    similarities: list[float | None] = []

    for document in plan.documents:
        doc_id = str(document.id)
        yield SseEvent("status", {"document_id": doc_id, "stage": "searching"})
        result = await asyncio.to_thread(
            retrieval.retrieve, plan.question, dataclasses.replace(base, document_ids=(doc_id,))
        )
        similarities += [c.similarity for c in result.candidates]
        metrics_service.add_retrieval(event, similarities)
        decision = guard.evaluate(
            plan.question,
            result.best_similarity,
            result.top_keyword_text,
            settings.relevance_threshold,
            settings.lexical_override_margin,
        )
        column: dict[str, Any] = {
            "document_id": doc_id,
            "document_name": document.name,
            "answer": "",
            "covered": False,
            "citations": [],
            "flags": {},
        }
        if decision.passed:
            yield SseEvent("status", {"document_id": doc_id, "stage": "generating"})
            final: generation.AnswerResult | None = None
            async for item in generation.generate_answer(
                plan.question, result.candidates, settings.analysis_max_retries
            ):
                if isinstance(item, llm.Wait):
                    yield SseEvent(
                        "status",
                        {
                            "document_id": doc_id,
                            "stage": "rate_limited",
                            "retry_in": round(item.seconds),
                        },
                    )
                elif isinstance(item, generation.Restart):
                    yield SseEvent(
                        "status", {"document_id": doc_id, "stage": "retrying", "reset": True}
                    )
                elif isinstance(item, generation.Token):
                    yield SseEvent("token", {"document_id": doc_id, "text": item.text})
                else:
                    final = item
            if final is None:
                raise llm.LlmError("empty", generation.EMPTY_MESSAGE)
            metrics_service.add_usage(event, final.usage)
            if not final.refused:
                cited = [retrieval.citation_dict(n, result.candidates[n - 1]) for n in final.cited]
                event["cited"] = int(event.get("cited") or 0) + len(cited)
                event["citations_removed"] = (
                    int(event.get("citations_removed") or 0) + final.invalid_removed
                )
                event["uncited"] = bool(event.get("uncited")) or final.uncited
                event["partial"] = bool(event.get("partial")) or final.truncated
                column.update(
                    answer=final.text,
                    covered=True,
                    citations=cited,
                    flags={
                        "partial": final.truncated,
                        "citations_removed": final.invalid_removed,
                        "uncited": final.uncited,
                    },
                )
                answered.append((document, cited, [result.candidates[n - 1] for n in final.cited]))
        yield SseEvent("column", column)

    conflicts: list[dict[str, Any]] = []
    cited_documents = {ref.id for ref, cited, _ in answered if cited}
    event["cited_documents"] = len(cited_documents)
    event["cited_document_ids"] = sorted(str(d) for d in cited_documents)
    if not answered:
        event["outcome"] = EventOutcome.REFUSED.value  # no document covers the question
    if settings.conflict_detection and len(cited_documents) >= 2:
        yield SseEvent("status", {"stage": "checking_sources"})
        # Number every cited passage once across the columns, ask, then map back to column numbers.
        passages: list[analysis.Passage] = []
        back: dict[int, tuple[str, int]] = {}
        for ref, cited, candidates in answered:
            for citation, candidate in zip(cited, candidates, strict=True):
                number = len(passages) + 1
                passages.append(analysis.Passage(number, candidate))
                back[number] = (str(ref.id), int(citation["n"]))
        try:
            found, used = await analysis.detect_conflicts(plan.question, passages)
            metrics_service.add_usage(event, used)
            for conflict in found:
                claims = [
                    {**claim, "n": back[claim["n"]][1]}
                    for claim in conflict["claims"]
                    if claim["n"] in back
                ]
                conflicts.append({"topic": conflict["topic"], "claims": claims})
        except Exception as exc:  # the columns are already delivered; this is extra
            logger.warning("conflict check failed", extra={"error": type(exc).__name__})
    event["conflicts"] = len(conflicts)
    yield SseEvent("conflicts", {"conflicts": conflicts})
    yield SseEvent(
        "done",
        {
            "question": plan.question,
            "documents": [str(d.id) for d in plan.documents],
            "conflicts": len(conflicts),
        },
    )
