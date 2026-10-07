"""Auth service: hashing, tokens, registration and credential checks."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import JWT_ALGORITHM
from app.db.database import get_sessionmaker
from app.db.models import User
from app.services import auth_service
from app.services.auth_service import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    InvalidToken,
)

PASSWORD = "correct horse battery"


def secret() -> str:
    return get_settings().jwt_secret.get_secret_value()


def make_token(claims: dict[str, object], key: str | None = None, alg: str = JWT_ALGORITHM) -> str:
    return jwt.encode(claims, key if key is not None else secret(), algorithm=alg)


def valid_claims(**overrides: object) -> dict[str, object]:
    claims: dict[str, object] = {
        "sub": str(uuid.uuid4()),
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    claims.update(overrides)
    return claims


# --- passwords ------------------------------------------------------------------


def test_hash_is_bcrypt_and_verifies() -> None:
    hashed = auth_service.hash_password(PASSWORD)
    assert hashed.startswith("$2b$")
    assert PASSWORD not in hashed
    assert auth_service.verify_password(PASSWORD, hashed) is True
    assert auth_service.verify_password("wrong password", hashed) is False


def test_hash_is_salted() -> None:
    assert auth_service.hash_password(PASSWORD) != auth_service.hash_password(PASSWORD)


def test_hash_uses_configured_work_factor() -> None:
    cost = int(auth_service.hash_password(PASSWORD).split("$")[2])
    assert cost == get_settings().bcrypt_rounds


@pytest.mark.parametrize("stored", ["", "not-a-hash", "$2b$04$tooshort"])
def test_verify_treats_malformed_hashes_as_mismatch(stored: str) -> None:
    assert auth_service.verify_password(PASSWORD, stored) is False


def test_verify_does_not_raise_on_overlong_password() -> None:
    hashed = auth_service.hash_password(PASSWORD)
    assert auth_service.verify_password("x" * 100, hashed) is False


# --- tokens ---------------------------------------------------------------------


def test_token_round_trip_and_lifetime() -> None:
    user = User(id=uuid.uuid4(), email="t@example.com", password_hash="x", role="user")
    token, expires_in = auth_service.create_access_token(user)
    assert expires_in == get_settings().jwt_expire_minutes * 60
    assert auth_service.decode_access_token(token) == user.id

    claims = jwt.decode(token, secret(), algorithms=[JWT_ALGORITHM])
    assert set(claims) == {"sub", "iat", "exp"}, "tokens carry no role or email"
    assert claims["exp"] - claims["iat"] == expires_in


def test_expired_token_is_rejected() -> None:
    token = make_token(valid_claims(exp=datetime.now(UTC) - timedelta(seconds=5)))
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(token)


def test_token_signed_with_another_key_is_rejected() -> None:
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(make_token(valid_claims(), key="x" * 40))


def test_tampered_token_is_rejected() -> None:
    token = make_token(valid_claims())
    header, payload, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(f"{header}.{payload}.{flipped}")


def test_algorithm_none_token_is_rejected() -> None:
    """Classic JWT attack: an unsigned token claiming alg=none."""
    unsigned = jwt.encode(valid_claims(), key=None, algorithm="none")  # type: ignore[arg-type]
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(unsigned)


@pytest.mark.filterwarnings("ignore:The HMAC key is:UserWarning")  # HS384 wants a longer key
def test_token_signed_with_a_different_hmac_algorithm_is_rejected() -> None:
    """The accepted algorithm is pinned; a token's own `alg` header is not trusted."""
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(make_token(valid_claims(), alg="HS384"))


@pytest.mark.parametrize(
    "claims",
    [
        {"iat": datetime.now(UTC)},  # no exp
        {"exp": datetime.now(UTC) + timedelta(minutes=5)},  # no sub
        {"sub": "not-a-uuid", "exp": datetime.now(UTC) + timedelta(minutes=5)},
    ],
)
def test_token_missing_or_malformed_claims_is_rejected(claims: dict[str, object]) -> None:
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(make_token(claims))


@pytest.mark.parametrize("garbage", ["", "abc", "a.b.c", "Bearer xyz"])
def test_garbage_is_rejected(garbage: str) -> None:
    with pytest.raises(InvalidToken):
        auth_service.decode_access_token(garbage)


# --- registration ---------------------------------------------------------------


def test_first_user_is_admin_and_later_users_are_not(db: Session) -> None:
    first = auth_service.register_user(db, "first@example.com", PASSWORD)
    second = auth_service.register_user(db, "second@example.com", PASSWORD)
    assert (first.role, second.role) == ("admin", "user")


def test_password_is_stored_hashed(db: Session) -> None:
    user = auth_service.register_user(db, "p@example.com", PASSWORD)
    assert user.password_hash != PASSWORD
    assert auth_service.verify_password(PASSWORD, user.password_hash)


def test_email_is_normalised_and_duplicates_are_case_insensitive(db: Session) -> None:
    user = auth_service.register_user(db, "  Mixed.Case@Example.COM ", PASSWORD)
    assert user.email == "mixed.case@example.com"
    with pytest.raises(EmailAlreadyRegistered):
        auth_service.register_user(db, "MIXED.CASE@example.com", PASSWORD)
    assert db.scalar(select(func.count()).select_from(User)) == 1


def test_failed_signup_does_not_consume_the_first_admin_slot(db: Session) -> None:
    auth_service.register_user(db, "a@example.com", PASSWORD)
    with pytest.raises(EmailAlreadyRegistered):
        auth_service.register_user(db, "a@example.com", PASSWORD)
    nobody_yet_admin = auth_service.register_user(db, "b@example.com", PASSWORD)
    assert nobody_yet_admin.role == "user"


@contextmanager
def slow_session(delay: float = 0.15) -> Iterator[Session]:
    """A session whose ``add`` pauses.

    ``register_user`` decides "is this the first user?" and then calls ``add`` and
    commits, so the pause widens the gap between check and commit. Without a
    lock every concurrent signup sees an empty table inside that gap, which makes
    the race reproducible instead of a matter of luck.
    """
    with get_sessionmaker()() as session:
        real_add = session.add

        def add(instance: object, *args: Any, **kwargs: Any) -> None:
            time.sleep(delay)
            real_add(instance, *args, **kwargs)

        session.add = add  # type: ignore[method-assign]
        yield session


def test_simultaneous_first_signups_produce_exactly_one_admin(db: Session) -> None:
    """The advisory lock must serialize signups; otherwise several become admin."""
    count = 6
    start = threading.Barrier(count)

    def signup(index: int) -> str:
        with slow_session() as session:
            start.wait()
            return auth_service.register_user(session, f"user{index}@example.com", PASSWORD).role

    with ThreadPoolExecutor(max_workers=count) as pool:
        roles = list(pool.map(signup, range(count)))

    assert roles.count("admin") == 1
    assert roles.count("user") == count - 1


def test_simultaneous_duplicate_signups_create_one_account(db: Session) -> None:
    """Holds with the lock (the pre-check rejects) and without it (unique(email) backstop)."""
    count = 6
    start = threading.Barrier(count)

    def signup(_: int) -> bool:
        with slow_session() as session:
            start.wait()
            try:
                auth_service.register_user(session, "same@example.com", PASSWORD)
            except EmailAlreadyRegistered:
                return False
            return True

    with ThreadPoolExecutor(max_workers=count) as pool:
        outcomes = list(pool.map(signup, range(count)))

    assert outcomes.count(True) == 1
    assert db.scalar(select(func.count()).select_from(User)) == 1


# --- authentication ---------------------------------------------------------------


def test_authenticate_accepts_correct_credentials_in_any_email_case(db: Session) -> None:
    created = auth_service.register_user(db, "who@example.com", PASSWORD)
    assert auth_service.authenticate(db, "WHO@example.com", PASSWORD).id == created.id


def test_wrong_password_and_unknown_email_fail_identically(db: Session) -> None:
    auth_service.register_user(db, "who@example.com", PASSWORD)
    with pytest.raises(InvalidCredentials) as wrong_password:
        auth_service.authenticate(db, "who@example.com", "not the password")
    with pytest.raises(InvalidCredentials) as unknown_email:
        auth_service.authenticate(db, "nobody@example.com", PASSWORD)
    assert type(wrong_password.value) is type(unknown_email.value)
    assert str(wrong_password.value) == str(unknown_email.value)


def test_unknown_email_still_pays_for_a_bcrypt_check(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, response time would reveal which emails have accounts."""
    calls: list[str] = []
    real_verify = auth_service.verify_password

    def spy(password: str, password_hash: str) -> bool:
        calls.append(password_hash)
        return real_verify(password, password_hash)

    monkeypatch.setattr(auth_service, "verify_password", spy)
    with pytest.raises(InvalidCredentials):
        auth_service.authenticate(db, "ghost@example.com", PASSWORD)
    assert len(calls) == 1
    assert calls[0].startswith("$2b$")


def test_get_user(db: Session) -> None:
    created = auth_service.register_user(db, "who@example.com", PASSWORD)
    assert auth_service.get_user(db, created.id).email == "who@example.com"  # type: ignore[union-attr]
    assert auth_service.get_user(db, uuid.uuid4()) is None


# --- privacy ----------------------------------------------------------------------


def test_logs_never_contain_emails_passwords_or_hashes(
    db: Session, caplog: pytest.LogCaptureFixture
) -> None:
    secret_password = "swordfish-but-longer"
    with caplog.at_level(logging.DEBUG):
        user = auth_service.register_user(db, "private.person@example.com", secret_password)
        auth_service.authenticate(db, "private.person@example.com", secret_password)
        with pytest.raises(InvalidCredentials):
            auth_service.authenticate(db, "private.person@example.com", "wrong-password-1")
        with pytest.raises(InvalidCredentials):
            auth_service.authenticate(db, "stranger@example.com", "wrong-password-2")
        token, _ = auth_service.create_access_token(user)

    output = "\n".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    for forbidden in (
        "private.person",
        "stranger@",
        secret_password,
        "wrong-password",
        user.password_hash,
        token,
    ):
        assert forbidden not in output
    assert "user registered" in output  # the event itself is still logged
