"""FastAPI dependencies. Auth is applied through these, never ad hoc."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.constants import Role
from app.db.database import session_scope
from app.db.models import User
from app.services import auth_service

# auto_error=False so a missing header becomes our own 401 (HTTPBearer's default is 403).
_bearer = HTTPBearer(auto_error=False, description="JWT from POST /api/auth/login or /signup")


def get_db() -> Iterator[Session]:
    """Provide a database session for one request."""
    yield from session_scope()


DbSession = Annotated[Session, Depends(get_db)]


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: DbSession,
) -> User:
    """Resolve the bearer token to a user, or respond 401."""
    if credentials is None:
        raise _unauthorized("Not authenticated")
    try:
        user_id = auth_service.decode_access_token(credentials.credentials)
    except auth_service.InvalidToken:
        raise _unauthorized("Invalid or expired token") from None
    user = auth_service.get_user(db, user_id)
    if user is None:  # account deleted since the token was issued
        raise _unauthorized("Invalid or expired token")
    # The request's session would otherwise keep this read transaction, and one pooled connection,
    # open until the response is finished: for a streamed answer, the whole generation. The user
    # is detached with its columns loaded; handlers only read ``id``, ``role`` and ``email``.
    db.expunge(user)
    db.rollback()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    """Allow administrators only; respond 403 to everyone else."""
    if user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required"
        )
    return user


AdminUser = Annotated[User, Depends(require_admin)]
