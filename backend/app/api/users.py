"""Account administration (administrators only): list, promote or demote, remove.

One sign-in page serves everyone; the account's role decides what it can do. These routes are how an
administrator changes that role, so the database is never needed for it.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Response, status

from app.deps import AdminUser, DbSession
from app.schemas import RoleUpdate, UserAdminOut
from app.services import user_service

router = APIRouter(prefix="/users", tags=["users"])


def _http_error(exc: user_service.UserAdminError) -> HTTPException:
    code = (
        status.HTTP_404_NOT_FOUND
        if isinstance(exc, user_service.UserNotFound)
        else status.HTTP_409_CONFLICT
    )
    return HTTPException(code, str(exc))


@router.get("", response_model=list[UserAdminOut], summary="All accounts")
def list_users(_: AdminUser, db: DbSession) -> list[UserAdminOut]:
    """Every account with its role and how much it owns."""
    return [
        UserAdminOut.model_validate(row, from_attributes=True)
        for row in user_service.list_users(db)
    ]


@router.patch("/{user_id}", response_model=UserAdminOut, summary="Promote or demote")
def set_role(user_id: uuid.UUID, body: RoleUpdate, _: AdminUser, db: DbSession) -> UserAdminOut:
    """Change an account's role. It applies at once, even to tokens already issued."""
    try:
        user_service.set_role(db, user_id, body.role)
    except user_service.UserAdminError as exc:
        raise _http_error(exc) from None
    row = next(r for r in user_service.list_users(db) if r.id == user_id)
    return UserAdminOut.model_validate(row, from_attributes=True)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Remove an account")
def delete_user(user_id: uuid.UUID, admin: AdminUser, db: DbSession) -> Response:
    """Remove an account and its conversations; documents it uploaded stay."""
    try:
        user_service.delete_user(db, admin.id, user_id)
    except user_service.UserAdminError as exc:
        raise _http_error(exc) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
