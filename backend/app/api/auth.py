"""Signup, login and current-user endpoints.

Handlers are synchronous on purpose: bcrypt is CPU-bound, and FastAPI runs plain
``def`` handlers in a worker thread, keeping the event loop free.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.db.models import User
from app.deps import CurrentUser, DbSession
from app.schemas import AuthResponse, LoginRequest, SignupRequest, UserOut
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


def _auth_response(user: User) -> AuthResponse:
    token, expires_in = auth_service.create_access_token(user)
    return AuthResponse(
        access_token=token, expires_in=expires_in, user=UserOut.model_validate(user)
    )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: DbSession) -> AuthResponse:
    """Create an account and sign it in. The very first account becomes `admin`."""
    try:
        user = auth_service.register_user(db, body.email, body.password)
    except auth_service.SignupDisabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Signup is disabled on this server.",
        ) from None
    except auth_service.EmailAlreadyRegistered:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists"
        ) from None
    return _auth_response(user)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: DbSession) -> AuthResponse:
    """Exchange email and password for a bearer token."""
    try:
        user = auth_service.authenticate(db, body.email, body.password)
    except auth_service.InvalidCredentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    return _auth_response(user)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    """Return the signed-in user and role."""
    return UserOut.model_validate(user)
