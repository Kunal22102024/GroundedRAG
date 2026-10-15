"""Conversation history and message feedback. Everything is scoped to the owning user."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.constants import DocumentStatus, MessageRole, allowed_access_levels
from app.db.models import Chunk, Conversation, Document, Feedback, Message, User
from app.services import cache_service


class NotFound(Exception):
    """The conversation or message does not exist or belongs to someone else."""


def list_conversations(db: Session, user: User) -> list[Conversation]:
    """The user's conversations, most recently active first."""
    return list(
        db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .limit(200)
        )
    )


def get_owned(db: Session, user: User, conversation_id: uuid.UUID) -> Conversation:
    """A conversation the user owns, or ``NotFound``."""
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user.id
        )
    )
    if conversation is None:
        raise NotFound
    return conversation


def ratings_for(db: Session, user: User, messages: list[Message]) -> dict[uuid.UUID, str]:
    """The user's thumbs up/down on the given messages."""
    if not messages:
        return {}
    rows = db.execute(
        select(Feedback.message_id, Feedback.rating).where(
            Feedback.user_id == user.id, Feedback.message_id.in_([m.id for m in messages])
        )
    )
    return {row.message_id: row.rating for row in rows}


def readable_chunk_ids(db: Session, user: User, chunk_ids: set[str]) -> set[str]:
    """Which of these chunks still exist in an indexed document the user may read."""
    if not chunk_ids:
        return set()
    levels = [level.value for level in allowed_access_levels(user.role)]
    return set(
        db.scalars(
            select(Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.id.in_(chunk_ids),
                Document.status == DocumentStatus.INDEXED.value,
                Document.access_level.in_(levels),
            )
        )
    )


def hide_unreadable(citations: list[dict[str, Any]], readable: set[str]) -> list[dict[str, Any]]:
    """The citations, with each one the reader may not see made ``unavailable``.

    A citation stores a copy of the cited passage so a result can be reopened later. That copy
    must not outlive the reader's right to read it: the document may have been deleted, or the
    reader's role lowered since. Such a citation keeps its number and page but loses the passage,
    the document name and the section, and the client shows "source no longer available".
    """
    return [
        citation
        if str(citation.get("chunk_id")) in readable
        else {
            **citation,
            "document_name": "",
            "section_title": "",
            "text": "",
            "unavailable": True,
        }
        for citation in citations
    ]


def readable_citations(
    db: Session, user: User, messages: list[Message]
) -> dict[uuid.UUID, list[dict[str, Any]] | None]:
    """Each message's citations, with the ones the user can no longer read made unavailable.

    The answer text itself is kept as a record of the exchange.
    """
    wanted = {str(c["chunk_id"]) for m in messages for c in m.citations or [] if c.get("chunk_id")}
    readable = readable_chunk_ids(db, user, wanted)
    return {
        m.id: None if m.citations is None else hide_unreadable(m.citations, readable)
        for m in messages
    }


def readable_conflicts(
    messages: list[Message], citations: dict[uuid.UUID, list[dict[str, Any]] | None]
) -> dict[uuid.UUID, list[dict[str, Any]] | None]:
    """Each message's conflicts without claims whose source the reader can no longer see.

    A conflict that is left with claims from fewer than two documents is no longer a conflict
    the reader can check, so it is dropped.
    """
    result: dict[uuid.UUID, list[dict[str, Any]] | None] = {}
    for message in messages:
        if not message.conflicts:
            result[message.id] = None
            continue
        gone = {c["n"] for c in citations.get(message.id) or [] if c.get("unavailable")}
        kept = []
        for conflict in message.conflicts:
            claims = [c for c in conflict.get("claims", []) if c.get("n") not in gone]
            if len({c.get("document_id") for c in claims}) >= 2:
                kept.append({**conflict, "claims": claims})
        result[message.id] = kept or None
    return result


def readable_suggestions(
    db: Session, user: User, messages: list[Message]
) -> dict[uuid.UUID, list[dict[str, Any]] | None]:
    """Each message's follow-up suggestions, without those whose passage the reader can no
    longer read (deleted document, lowered role): its question could quote that document."""
    wanted = {
        str(s["chunk_id"]) for m in messages for s in m.suggestions or [] if s.get("chunk_id")
    }
    readable = readable_chunk_ids(db, user, wanted)
    result: dict[uuid.UUID, list[dict[str, Any]] | None] = {}
    for message in messages:
        kept = [
            {"type": s["type"], "question": s["question"], "document_id": s.get("document_id")}
            for s in message.suggestions or []
            if str(s.get("chunk_id")) in readable
        ]
        result[message.id] = kept or None
    return result


def rename(db: Session, user: User, conversation_id: uuid.UUID, title: str) -> Conversation:
    conversation = get_owned(db, user, conversation_id)
    conversation.title = " ".join(title.split())[:200] or conversation.title
    db.commit()
    return conversation


def delete(db: Session, user: User, conversation_id: uuid.UUID) -> None:
    db.delete(get_owned(db, user, conversation_id))  # messages and feedback cascade
    db.commit()


def rate(db: Session, user: User, message_id: uuid.UUID, rating: str, comment: str | None) -> None:
    """Set the user's rating of an assistant message in one of their conversations."""
    message = db.scalar(
        select(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.id == message_id,
            Message.role == MessageRole.ASSISTANT.value,
            Conversation.user_id == user.id,
        )
    )
    if message is None:
        raise NotFound
    # One atomic statement: two first ratings arriving together cannot collide on the unique key.
    insert = pg_insert(Feedback).values(
        message_id=message_id, user_id=user.id, rating=rating, comment=comment
    )
    db.execute(
        insert.on_conflict_do_update(
            index_elements=[Feedback.message_id, Feedback.user_id],
            set_={"rating": insert.excluded.rating, "comment": insert.excluded.comment},
        )
    )
    db.commit()
    if rating == "down":  # a bad answer must not be served again from the saved-answer cache
        cache_service.evict_message(message_id)
