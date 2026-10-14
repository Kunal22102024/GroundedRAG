"""Signup, login, /auth/me and the auth dependencies, through the real HTTP stack."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import JWT_ALGORITHM
from app.deps import AdminUser
from app.main import app

PASSWORD = "correct horse battery"


async def _admin_only(user: AdminUser) -> dict[str, str]:
    return {"email": user.email}


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    if not any(getattr(route, "path", "") == "/api/_test/admin-only" for route in app.routes):
        app.add_api_route("/api/_test/admin-only", _admin_only)
    with TestClient(app) as test_client:
        yield test_client


def signup(client: TestClient, email: str, password: str = PASSWORD) -> Any:
    return client.post("/api/auth/signup", json={"email": email, "password": password})


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def token_for(client: TestClient, email: str) -> str:
    response = signup(client, email)
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


# --- signup ---------------------------------------------------------------------


def test_signup_returns_a_working_token_and_the_user(client: TestClient) -> None:
    response = signup(client, "First@Example.com")
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == get_settings().jwt_expire_minutes * 60
    assert body["user"]["email"] == "first@example.com"
    assert uuid.UUID(body["user"]["id"])

    me = client.get("/api/auth/me", headers=bearer(body["access_token"]))
    assert me.status_code == 200
    assert me.json() == body["user"]


def test_first_account_is_admin_then_users(client: TestClient) -> None:
    assert signup(client, "boss@example.com").json()["user"]["role"] == "admin"
    assert signup(client, "worker@example.com").json()["user"]["role"] == "user"
    assert signup(client, "another@example.com").json()["user"]["role"] == "user"


def test_responses_never_expose_the_password_or_its_hash(client: TestClient) -> None:
    response = signup(client, "leak@example.com")
    assert PASSWORD not in response.text
    assert "password" not in response.text.lower()
    assert "$2b$" not in response.text


def test_duplicate_email_is_rejected_regardless_of_case(client: TestClient) -> None:
    assert signup(client, "dup@example.com").status_code == 201
    response = signup(client, "DUP@Example.com")
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.parametrize(
    "email", ["", "plainaddress", "no-at.example.com", "a@b", "a b@c.com", "@x.com"]
)
def test_malformed_email_is_rejected(client: TestClient, email: str) -> None:
    assert signup(client, email).status_code == 422


def test_internal_domains_are_accepted(client: TestClient) -> None:
    assert signup(client, "sarah@corp.local").status_code == 201


@pytest.mark.parametrize("password", ["", "short", "1234567"])
def test_short_password_is_rejected(client: TestClient, password: str) -> None:
    assert signup(client, "x@example.com", password).status_code == 422


def test_password_length_limit_is_in_bytes_because_bcrypt_is(client: TestClient) -> None:
    assert signup(client, "ok72@example.com", "a" * 72).status_code == 201
    assert signup(client, "bad73@example.com", "a" * 73).status_code == 422
    # 37 two-byte characters is 37 characters but 74 bytes.
    assert signup(client, "bad74@example.com", "é" * 37).status_code == 422
    assert signup(client, "ok36@example.com", "é" * 36).status_code == 201


def test_signup_rejects_unknown_fields_and_missing_fields(client: TestClient) -> None:
    extra = client.post(
        "/api/auth/signup", json={"email": "e@example.com", "password": PASSWORD, "role": "admin"}
    )
    assert extra.status_code == 422, "clients must not be able to pick their own role"
    assert client.post("/api/auth/signup", json={"email": "e@example.com"}).status_code == 422
    assert client.post("/api/auth/signup", json={}).status_code == 422


def test_a_rejected_signup_creates_nothing(client: TestClient, db: Session) -> None:
    signup(client, "nope@example.com", "short")
    assert db.scalar(text("SELECT count(*) FROM users")) == 0


# --- login ----------------------------------------------------------------------


def test_login_with_correct_credentials(client: TestClient) -> None:
    signup(client, "who@example.com")
    response = client.post(
        "/api/auth/login", json={"email": "WHO@example.com", "password": PASSWORD}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["email"] == "who@example.com"
    assert client.get("/api/auth/me", headers=bearer(body["access_token"])).status_code == 200


def test_wrong_password_and_unknown_email_look_the_same(client: TestClient) -> None:
    signup(client, "who@example.com")
    wrong = client.post(
        "/api/auth/login", json={"email": "who@example.com", "password": "nope-nope"}
    )
    unknown = client.post(
        "/api/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid email or password"}
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_login_accepts_passwords_that_signup_would_now_reject(client: TestClient) -> None:
    """Login does not enforce strength rules: it must never lock out an existing account."""
    response = client.post("/api/auth/login", json={"email": "x@example.com", "password": "s"})
    assert response.status_code == 401  # a normal credential failure, not a 422


def test_login_does_not_log_in_an_empty_password(client: TestClient) -> None:
    signup(client, "who@example.com")
    response = client.post("/api/auth/login", json={"email": "who@example.com", "password": ""})
    assert response.status_code == 422


# --- /auth/me and token handling ----------------------------------------------------


def test_me_requires_a_token(client: TestClient) -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    ["Bearer", "Bearer ", "Basic dXNlcjpwYXNz", "Bearer not.a.jwt", "token abc", "Bearer a.b.c"],
)
def test_me_rejects_bad_authorization_headers(client: TestClient, header: str) -> None:
    response = client.get("/api/auth/me", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_me_rejects_an_expired_token(client: TestClient) -> None:
    token = token_for(client, "late@example.com")
    user_id = jwt.decode(token, options={"verify_signature": False})["sub"]
    expired = jwt.encode(
        {"sub": user_id, "exp": datetime.now(UTC) - timedelta(minutes=1)},
        get_settings().jwt_secret.get_secret_value(),
        algorithm=JWT_ALGORITHM,
    )
    response = client.get("/api/auth/me", headers=bearer(expired))
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}


def test_token_for_a_deleted_user_stops_working_immediately(
    client: TestClient, db: Session
) -> None:
    token = token_for(client, "gone@example.com")
    assert client.get("/api/auth/me", headers=bearer(token)).status_code == 200
    db.execute(text("DELETE FROM users"))
    db.commit()
    assert client.get("/api/auth/me", headers=bearer(token)).status_code == 401


def test_invalid_token_responses_do_not_reveal_why(client: TestClient) -> None:
    token = token_for(client, "x@example.com")
    forged = jwt.encode({"sub": str(uuid.uuid4()), "exp": 9999999999}, "k" * 40, algorithm="HS256")
    for candidate in (forged, token[:-3] + "abc"):
        response = client.get("/api/auth/me", headers=bearer(candidate))
        assert response.json() == {"detail": "Invalid or expired token"}


# --- roles ------------------------------------------------------------------------


def test_require_admin_allows_admin_and_blocks_users(client: TestClient) -> None:
    admin_token = token_for(client, "boss@example.com")
    user_token = token_for(client, "worker@example.com")

    ok = client.get("/api/_test/admin-only", headers=bearer(admin_token))
    assert (ok.status_code, ok.json()) == (200, {"email": "boss@example.com"})

    forbidden = client.get("/api/_test/admin-only", headers=bearer(user_token))
    assert forbidden.status_code == 403
    assert client.get("/api/_test/admin-only").status_code == 401


def test_role_is_read_from_the_database_on_every_request(client: TestClient, db: Session) -> None:
    """Promotions and demotions apply to tokens that were issued before the change."""
    token_for(client, "boss@example.com")
    user_token = token_for(client, "worker@example.com")
    assert client.get("/api/_test/admin-only", headers=bearer(user_token)).status_code == 403

    db.execute(text("UPDATE users SET role = 'admin' WHERE email = 'worker@example.com'"))
    db.commit()
    assert client.get("/api/auth/me", headers=bearer(user_token)).json()["role"] == "admin"
    assert client.get("/api/_test/admin-only", headers=bearer(user_token)).status_code == 200

    db.execute(text("UPDATE users SET role = 'user' WHERE email = 'worker@example.com'"))
    db.commit()
    assert client.get("/api/_test/admin-only", headers=bearer(user_token)).status_code == 403


# --- documentation ------------------------------------------------------------------


def test_openapi_describes_auth_routes_and_bearer_scheme(client: TestClient) -> None:
    spec = client.get("/api/openapi.json").json()
    for path in ("/api/auth/signup", "/api/auth/login", "/api/auth/me"):
        assert path in spec["paths"]
    assert spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"


# --- SIGNUP_ENABLED --------------------------------------------------------------------------------


def test_signup_can_be_closed_once_the_first_account_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "signup_enabled", False)

    # A fresh deployment must never be locked out: the first account is always allowed.
    first = signup(client, "boss@example.com")
    assert first.status_code == 201 and first.json()["user"]["role"] == "admin"

    closed = signup(client, "newcomer@example.com")
    assert (
        closed.status_code == 403
        and closed.json()["detail"] == "Signup is disabled on this server."
    )
    # A closed door must not reveal which addresses have an account.
    assert signup(client, "boss@example.com").status_code == 403

    login = client.post("/api/auth/login", json={"email": "boss@example.com", "password": PASSWORD})
    assert login.status_code == 200  # signing in is unaffected

    monkeypatch.setattr(get_settings(), "signup_enabled", True)
    assert signup(client, "newcomer@example.com").status_code == 201


def test_a_refused_signup_creates_no_account(
    client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    signup(client, "boss@example.com")
    monkeypatch.setattr(get_settings(), "signup_enabled", False)
    signup(client, "ghost@example.com")
    assert db.execute(text("select count(*) from users")).scalar_one() == 1
