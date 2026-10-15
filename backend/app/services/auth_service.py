"""Authentication workflows: password hashing, access tokens, user creation.

Privacy (CLAUDE.md rule 21): never log passwords, tokens or email addresses;
log the user id only.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import bcrypt
import jwt
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import JWT_ALGORITHM, Role
from app.db.models import User

logger = logging.getLogger(__name__)

# Arbitrary constant naming the advisory lock that serializes signups.
_SIGNUP_LOCK_KEY = 0x47524147


class AuthError(Exception):
    """Base class for authentication failures."""


class EmailAlreadyRegistered(AuthError):
    """Signup with an email that already has an account."""


class SignupDisabled(AuthError):
    """Self-service signup is switched off (``SIGNUP_ENABLED=false``) and an account exists."""


class InvalidCredentials(AuthError):
    """Wrong email or password. Deliberately does not say which."""


class InvalidToken(AuthError):
    """Missing, malformed, tampered, or expired access token."""


def normalize_email(email: str) -> str:
    """Canonical form used for storage and lookup."""
    return email.strip().lower()


def hash_password(password: str) -> str:
    """Hash a password with bcrypt at the configured work factor."""
    rounds = get_settings().bcrypt_rounds
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against a bcrypt hash. Malformed input is simply a mismatch."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:  # e.g. password longer than bcrypt's 72-byte limit
        return False


@lru_cache
def _dummy_hash() -> str:
    """A throwaway hash so unknown-email logins cost the same as wrong-password ones."""
    return hash_password("not-a-real-password")


def lock_accounts(db: Session) -> None:
    """Serialize everything that creates, promotes, demotes or removes accounts, until the
    transaction ends.

    Without this, two simultaneous first signups could both see an empty ``users`` table and both
    become admin, and two administrators demoting each other at the same moment could both pass
    the "is there another admin" check and leave none (``user_service``).
    """
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SIGNUP_LOCK_KEY})


def register_user(db: Session, email: str, password: str) -> User:
    """Create an account. The first account ever registered becomes ``admin``.

    Raises ``EmailAlreadyRegistered`` if the email is taken, and ``SignupDisabled`` when signup is
    switched off and an account already exists (the first account can always be created, so a
    fresh deployment is never locked out).
    """
    email = normalize_email(email)
    password_hash = hash_password(password)  # slow: do it before holding the lock

    lock_accounts(db)
    is_first_user = db.scalar(select(func.count()).select_from(User)) == 0
    if not is_first_user and not get_settings().signup_enabled:
        db.rollback()  # checked before the email lookup: a closed door must not reveal accounts
        raise SignupDisabled
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        db.rollback()
        raise EmailAlreadyRegistered
    user = User(
        email=email,
        password_hash=password_hash,
        role=(Role.ADMIN if is_first_user else Role.USER).value,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # unique(email) is the backstop for the check above
        db.rollback()
        raise EmailAlreadyRegistered from exc
    db.refresh(user)
    logger.info("user registered", extra={"user_id": str(user.id), "role": user.role})
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    """Return the user for valid credentials, else raise ``InvalidCredentials``."""
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is not None:
        db.expunge(user)  # keeps its loaded columns for the caller
    # A bcrypt check takes about 250 ms. Give the connection back first: some forty concurrent
    # logins would otherwise hold every pooled connection and stall the whole API.
    db.rollback()
    if user is None:
        verify_password(password, _dummy_hash())  # equalize timing; result is irrelevant
        logger.info("login failed")
        raise InvalidCredentials
    if not verify_password(password, user.password_hash):
        logger.info("login failed", extra={"user_id": str(user.id)})
        raise InvalidCredentials
    return user


def create_access_token(user: User) -> tuple[str, int]:
    """Issue a short-lived JWT. Returns ``(token, expires_in_seconds)``.

    The token carries only the user id. Role and existence are re-read from the
    database on every request, so demotions and deletions take effect at once.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    lifetime = timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user.id), "iat": now, "exp": now + lifetime}
    token = jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=JWT_ALGORITHM)
    return token, int(lifetime.total_seconds())


def decode_access_token(token: str) -> uuid.UUID:
    """Validate a token and return the user id it names. Raises ``InvalidToken``."""
    try:
        payload = jwt.decode(
            token,
            get_settings().jwt_secret.get_secret_value(),
            algorithms=[JWT_ALGORITHM],  # pinned: never accept the token's own `alg`
            options={"require": ["exp", "sub"]},
        )
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise InvalidToken from exc


def get_user(db: Session, user_id: uuid.UUID) -> User | None:
    """Load a user by id."""
    return db.get(User, user_id)
