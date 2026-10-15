"""Questions to suggest before the first question of a chat.

Two sources: questions the chosen documents answer (written once per document by the model and
saved), and questions other people asked and got a cleanly cited answer to (only when several
people asked them, and only about documents the caller may read). Follow-up questions after an
answer arrive with the answer itself (``suggestions`` chat event).
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Query, status

from app.config import get_settings
from app.db.database import get_sessionmaker
from app.deps import CurrentUser
from app.schemas import SuggestionsOut
from app.services import analysis_service, metrics_service
from app.services.document_service import DocumentNotFound

router = APIRouter(tags=["suggestions"])

POPULAR_LIMIT = 6
DOCUMENT_QUESTIONS_LIMIT = 8


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
    limit = get_settings().compare_max_documents
    if len(set(ids)) > limit:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Choose at most {limit} documents."
        )
    return list(dict.fromkeys(ids))


@router.get("/suggestions", response_model=SuggestionsOut, summary="Suggested questions")
async def suggestions(
    user: CurrentUser,
    document_ids: str | None = Query(default=None, description="Comma-separated document ids"),
) -> SuggestionsOut:
    """Questions from the chosen documents (or recent ones) and questions often asked about them."""
    ids = _parse_ids(document_ids)
    try:
        questions, pending = await asyncio.to_thread(
            analysis_service.document_suggestions, user, ids
        )
    except DocumentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found") from None

    def popular() -> list[dict[str, object]]:
        with get_sessionmaker()() as db:
            return metrics_service.popular_questions(
                db, user.role, [str(i) for i in ids] or None, POPULAR_LIMIT
            )

    asked = await asyncio.to_thread(popular)
    already = {q["question"].casefold() for q in questions}
    return SuggestionsOut.model_validate(
        {
            "documents": questions[:DOCUMENT_QUESTIONS_LIMIT],
            "popular": [p for p in asked if str(p["question"]).casefold() not in already],
            "pending": pending,
        }
    )
