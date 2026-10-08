"""Conversation history (list, read, rename, delete) and message feedback."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Response, status

from app.deps import CurrentUser, DbSession
from app.schemas import (
    ConversationDetail,
    ConversationOut,
    FeedbackRequest,
    MessageOut,
    RenameRequest,
)
from app.services import conversation_service as service

router = APIRouter(tags=["conversations"])


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Not found")


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: CurrentUser, db: DbSession) -> list[ConversationOut]:
    """The caller's conversations, newest activity first."""
    return [ConversationOut.model_validate(c) for c in service.list_conversations(db, user)]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> ConversationDetail:
    """A conversation with its messages, citations and the caller's ratings."""
    try:
        conversation = service.get_owned(db, user, conversation_id)
    except service.NotFound:
        raise _not_found() from None
    ratings = service.ratings_for(db, user, conversation.messages)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            # Built field by field: the ORM `feedback` relationship is a list, not a rating.
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=m.citations,
                guard_score=m.guard_score,
                flagged=m.flagged,
                created_at=m.created_at,
                feedback=ratings.get(m.id),  # type: ignore[arg-type]
            )
            for m in conversation.messages
        ],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: uuid.UUID, body: RenameRequest, user: CurrentUser, db: DbSession
) -> ConversationOut:
    """Rename a conversation."""
    try:
        return ConversationOut.model_validate(service.rename(db, user, conversation_id, body.title))
    except service.NotFound:
        raise _not_found() from None


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: uuid.UUID, user: CurrentUser, db: DbSession) -> Response:
    """Delete a conversation and its messages."""
    try:
        service.delete(db, user, conversation_id)
    except service.NotFound:
        raise _not_found() from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def rate_message(
    message_id: uuid.UUID, body: FeedbackRequest, user: CurrentUser, db: DbSession
) -> Response:
    """Thumbs up/down on an assistant message (one rating per user, overwritten on change)."""
    try:
        service.rate(db, user, message_id, body.rating.value, body.comment)
    except service.NotFound:
        raise _not_found() from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
