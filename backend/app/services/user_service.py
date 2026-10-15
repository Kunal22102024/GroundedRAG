"""Account administration: list accounts, promote or demote, remove (administrators only).

There is always at least one administrator. Role changes and removals run under the same advisory
lock as signup (``auth_service.lock_accounts``), so two administrators demoting or removing each
other at the same moment cannot both pass the "is another admin left" check.

A role change applies at once, even to tokens already issued: ``deps.get_current_user`` re-reads
the role on every request. Removing an account cascades in the database: its conversations,
messages and feedback go; documents it uploaded stay (owner cleared); its measurements stay
(user cleared). Privacy (rule 21): log ids and roles, never emails.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.constants import Role
from app.db.models import Conversation, Document, User
from app.services import auth_service

logger = logging.getLogger(__name__)


class UserAdminError(Exception):
    """Base class; ``message`` is safe to show."""

    message = "The change was not made."

    def __str__(self) -> str:
        return self.message


class UserNotFound(UserAdminError):
    message = "User not found."


class LastAdmin(UserAdminError):
    message = (
        "There must always be at least one administrator. Make someone else an administrator first."
    )


class SelfDelete(UserAdminError):
    message = "You cannot delete your own account. Ask another administrator to do it."


@dataclass(frozen=True)
class UserRow:
    """An account with what it owns, for the administrators' list."""

    id: uuid.UUID
    email: str
    role: str
    created_at: datetime
    documents: int
    conversations: int


def list_users(db: Session) -> list[UserRow]:
    """Every account, oldest first, with how many documents it uploaded and conversations it has."""
    documents = (
        select(Document.owner_id, func.count().label("n")).group_by(Document.owner_id).subquery()
    )
    conversations = (
        select(Conversation.user_id, func.count().label("n"))
        .group_by(Conversation.user_id)
        .subquery()
    )
    rows = db.execute(
        select(User, documents.c.n, conversations.c.n)
        .outerjoin(documents, documents.c.owner_id == User.id)
        .outerjoin(conversations, conversations.c.user_id == User.id)
        .order_by(User.created_at, User.email)
    ).all()
    return [
        UserRow(u.id, u.email, u.role, u.created_at, int(d or 0), int(c or 0)) for u, d, c in rows
    ]


def _admin_count(db: Session) -> int:
    return int(
        db.scalar(select(func.count()).select_from(User).where(User.role == Role.ADMIN.value)) or 0
    )


def _locked_user(db: Session, user_id: uuid.UUID) -> User:
    """Take the accounts lock, then read the user (never the other way round)."""
    auth_service.lock_accounts(db)
    user = db.get(User, user_id)
    if user is None:
        db.rollback()
        raise UserNotFound
    return user


def set_role(db: Session, user_id: uuid.UUID, role: Role) -> User:
    """Promote or demote. ``UserNotFound``; ``LastAdmin`` if this would leave no administrator."""
    user = _locked_user(db, user_id)
    if user.role == role.value:
        db.rollback()
        return user
    if user.role == Role.ADMIN.value and _admin_count(db) <= 1:
        db.rollback()
        raise LastAdmin
    user.role = role.value
    db.commit()
    db.refresh(user)
    logger.info("user role changed", extra={"user_id": str(user.id), "role": user.role})
    return user


def delete_user(db: Session, actor_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Remove an account. ``UserNotFound``; ``SelfDelete``; ``LastAdmin``."""
    if actor_id == user_id:
        raise SelfDelete
    user = _locked_user(db, user_id)
    if user.role == Role.ADMIN.value and _admin_count(db) <= 1:
        db.rollback()
        raise LastAdmin
    # A statement, not db.delete(): the database cascades do the work (conversations, messages and
    # feedback are removed, documents keep existing with no owner, measurements lose the user).
    db.execute(delete(User).where(User.id == user_id))
    db.commit()
    logger.info("user deleted", extra={"user_id": str(user_id)})
