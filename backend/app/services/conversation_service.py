"""Conversation history and message feedback. Everything is scoped to the owning user."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.constants import MessageRole
from app.db.models import Conversation, Feedback, Message, User


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
    feedback = db.scalar(
        select(Feedback).where(Feedback.message_id == message_id, Feedback.user_id == user.id)
    )
    if feedback is None:
        db.add(Feedback(message_id=message_id, user_id=user.id, rating=rating, comment=comment))
    else:
        feedback.rating, feedback.comment = rating, comment
    db.commit()
