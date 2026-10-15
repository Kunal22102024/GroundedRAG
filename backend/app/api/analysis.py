"""Document analysis: saved insights, summaries, field extraction and comparison.

The analyses run for seconds to minutes, so each one streams Server-Sent Events like chat:
``status`` while it works, then ``done`` with the result, or ``error``. Access problems are plain
HTTP errors before any streaming starts (404 for a document the caller may not read, 409 for one
that is not indexed yet).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.chat import _SSE_HEADERS, format_sse
from app.core.analysis import AnalysisError
from app.deps import CurrentUser, DbSession
from app.schemas import CompareRequest, ExtractRequest, InsightOut, SummaryRequest
from app.services import analysis_service, chat_service
from app.services.document_service import DocumentNotFound

router = APIRouter(tags=["analysis"])

_PRE_STREAM_ERRORS = (DocumentNotFound, analysis_service.DocumentNotReady, AnalysisError)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DocumentNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    if isinstance(exc, analysis_service.DocumentNotReady):
        return HTTPException(
            status.HTTP_409_CONFLICT, "The document is not indexed yet. Try again when it is ready."
        )
    if isinstance(exc, AnalysisError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.message)
    raise exc


async def _events(source: AsyncIterator[chat_service.SseEvent]) -> AsyncIterator[str]:
    """Format events; a failure becomes an ``error`` event instead of a broken stream."""
    try:
        async for event in source:
            yield format_sse(event)
    except Exception as exc:  # CancelledError (client went away) is not an Exception: it propagates
        yield format_sse(chat_service.error_event(exc))


def _stream(source: AsyncIterator[chat_service.SseEvent]) -> StreamingResponse:
    return StreamingResponse(_events(source), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get(
    "/documents/{document_id}/insights",
    response_model=list[InsightOut],
    summary="Saved summaries and extractions of a document",
)
def list_insights(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> list[InsightOut]:
    """What has already been generated for this document (passages the caller may no longer see
    are marked unavailable)."""
    try:
        rows = analysis_service.list_insights(db, user, document_id)
    except DocumentNotFound as exc:
        raise _http_error(exc) from None
    return [InsightOut.model_validate(row) for row in rows]


@router.post("/analysis/summary", response_class=StreamingResponse, summary="Summarize a document")
async def summarize(body: SummaryRequest, user: CurrentUser) -> StreamingResponse:
    """Executive summary, chapter-wise summary, FAQ, glossary or key-changes report.

    A saved result is returned at once unless ``refresh`` is set."""
    try:
        plan = await asyncio.to_thread(
            analysis_service.prepare_summary,
            user,
            body.document_id,
            body.kind,
            body.compare_to,
            body.refresh,
        )
    except _PRE_STREAM_ERRORS as exc:
        raise _http_error(exc) from None
    return _stream(analysis_service.run_summary(plan, user))


@router.post(
    "/analysis/extract", response_class=StreamingResponse, summary="Extract fields from a document"
)
async def extract(body: ExtractRequest, user: CurrentUser) -> StreamingResponse:
    """Dates, parties, amounts, requirements or any named fields, each value with its source."""
    try:
        plan = await asyncio.to_thread(
            analysis_service.prepare_extract, user, body.document_id, body.fields, body.refresh
        )
    except _PRE_STREAM_ERRORS as exc:
        raise _http_error(exc) from None
    return _stream(analysis_service.run_extract(plan, user))


@router.post("/analysis/compare", response_class=StreamingResponse, summary="Compare documents")
async def compare(body: CompareRequest, user: CurrentUser) -> StreamingResponse:
    """The same question answered from each chosen document separately, plus the points on which
    the documents disagree."""
    try:
        plan = await asyncio.to_thread(
            analysis_service.prepare_compare, user, body.question, body.document_ids
        )
    except _PRE_STREAM_ERRORS as exc:
        raise _http_error(exc) from None
    return _stream(analysis_service.run_compare(plan))
