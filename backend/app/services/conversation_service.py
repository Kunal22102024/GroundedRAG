"""Conversation history and message feedback. Everything is scoped to the owning user."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.constants import DocumentStatus, MessageRole, allowed_access_levels
from app.db.models import Chunk, Conversation, Document, Feedback, Message, User


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


def readable_citations(
    db: Session, user: User, messages: list[Message]
) -> dict[uuid.UUID, list[dict[str, Any]] | None]:
    """Each message's citations, with the ones the user can no longer read made unavailable.

    A citation stores a copy of the cited passage so a conversation can be reopened later. That
    copy must not outlive the user's right to read it: the document may have been deleted, or
    the user's role lowered since. Such a citation keeps its number and page but loses the
    passage, the document name and the section (``unavailable`` is set), and the client shows
    "source no longer available". The answer text itself is kept as a record of the exchange.
    """
    wanted = {str(c["chunk_id"]) for m in messages for c in m.citations or [] if c.get("chunk_id")}
    readable: set[str] = set()
    if wanted:
        levels = [level.value for level in allowed_access_levels(user.role)]
        readable = set(
            db.scalars(
                select(Chunk.id)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    Chunk.id.in_(wanted),
                    Document.status == DocumentStatus.INDEXED.value,
                    Document.access_level.in_(levels),
                )
            )
        )
    result: dict[uuid.UUID, list[dict[str, Any]] | None] = {}
    for message in messages:
        if message.citations is None:
            result[message.id] = None
            continue
        result[message.id] = [
            citation
            if str(citation.get("chunk_id")) in readable
            else {
                **citation,
                "document_name": "",
                "section_title": "",
                "text": "",
                "unavailable": True,
            }
            for citation in message.citations
        ]
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
