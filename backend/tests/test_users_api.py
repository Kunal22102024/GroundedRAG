"""Account administration: list, promote, demote and remove accounts (real Postgres).

What matters: only administrators can do it, a role change applies to tokens already issued, there
is always at least one administrator (also when two administrators act at the same moment), and
removing an account takes its conversations but not the documents it uploaded.
"""

# The fixtures are imported from test_chat and used by name as test arguments.
# ruff: noqa: F401, F811

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.constants import Role
from app.db.database import get_sessionmaker
from app.db.models import Conversation, Document, QueryEvent, User
from app.services import user_service
from tests.conftest import FakeChatModel
from tests.test_chat import Headers, admin, ask, client, library, user
from tests.test_documents_api import auth, one

LEAVE_Q = "How many days of paid annual leave do full-time employees receive?"


def me(client: TestClient, headers: Headers) -> dict[str, Any]:
    return client.get("/api/auth/me", headers=headers).json()  # type: ignore[no-any-return]


def users(client: TestClient, headers: Headers) -> dict[str, dict[str, Any]]:
    response = client.get("/api/users", headers=headers)
    assert response.status_code == 200, response.text
    return {u["email"]: u for u in response.json()}


def set_role(client: TestClient, headers: Headers, user_id: str, role: str):  # noqa: ANN201
    return client.patch(f"/api/users/{user_id}", headers=headers, json={"role": role})


def admin_count(db: Session) -> int:
    db.expire_all()
    return int(db.scalar(select(func.count()).select_from(User).where(User.role == "admin")) or 0)


# --- Who may do it -------------------------------------------------------------------------------


def test_only_administrators_manage_accounts(
    client: TestClient, admin: Headers, user: Headers
) -> None:
    target = me(client, user)["id"]
    for call in (
        lambda h: client.get("/api/users", headers=h),
        lambda h: set_role(client, h, target, "admin"),
        lambda h: client.delete(f"/api/users/{target}", headers=h),
    ):
        assert call(user).status_code == 403
        assert call({}).status_code == 401
    assert client.get("/api/users", headers=admin).status_code == 200


def test_the_list_shows_roles_and_what_each_account_owns(
    client: TestClient, admin: Headers, user: Headers, library: dict
) -> None:
    listed = users(client, admin)
    assert (
        listed["boss@example.com"]["role"] == "admin"
        and listed["worker@example.com"]["role"] == "user"
    )
    assert listed["boss@example.com"]["documents"] == 3, (
        "the library was uploaded by the administrator"
    )
    assert (
        listed["worker@example.com"]["documents"] == 0
        and "password_hash" not in listed["boss@example.com"]
    )


# --- Promote and demote --------------------------------------------------------------------------


def test_a_promotion_applies_to_the_token_the_person_already_has(
    client: TestClient, admin: Headers, user: Headers
) -> None:
    assert client.get("/api/metrics/summary", headers=user).status_code == 403
    response = set_role(client, admin, me(client, user)["id"], "admin")
    assert response.status_code == 200 and response.json()["role"] == "admin"
    assert client.get("/api/metrics/summary", headers=user).status_code == 200


def test_a_demotion_applies_to_the_token_the_person_already_has(
    client: TestClient, admin: Headers, user: Headers
) -> None:
    set_role(client, admin, me(client, user)["id"], "admin")
    assert client.get("/api/users", headers=user).status_code == 200
    assert set_role(client, admin, me(client, user)["id"], "user").status_code == 200
    assert client.get("/api/users", headers=user).status_code == 403


def test_the_same_role_again_changes_nothing(
    client: TestClient, admin: Headers, user: Headers
) -> None:
    response = set_role(client, admin, me(client, user)["id"], "user")
    assert response.status_code == 200 and response.json()["role"] == "user"


def test_the_last_administrator_cannot_be_demoted_but_one_of_two_can(
    client: TestClient, admin: Headers, user: Headers, db: Session
) -> None:
    boss = me(client, admin)["id"]
    refused = set_role(client, admin, boss, "user")
    assert refused.status_code == 409 and "at least one administrator" in refused.json()["detail"]
    assert admin_count(db) == 1

    set_role(client, admin, me(client, user)["id"], "admin")
    assert set_role(client, admin, boss, "user").status_code == 200, "demoting yourself is allowed"
    assert admin_count(db) == 1
    assert client.get("/api/users", headers=admin).status_code == 403, (
        "and you lose the screen at once"
    )


def test_two_administrators_cannot_demote_each_other_into_having_none(
    client: TestClient, admin: Headers, user: Headers, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accounts lock must serialize the check and the change; widen the gap between them on
    purpose, as the signup test does, so that without the lock both would pass the check."""
    a, b = me(client, admin)["id"], me(client, user)["id"]
    set_role(client, admin, b, "admin")
    real_count = user_service._admin_count

    def slow_count(session: Session) -> int:
        count = real_count(session)
        time.sleep(0.3)
        return count

    monkeypatch.setattr(user_service, "_admin_count", slow_count)
    start = threading.Barrier(2)

    def demote(user_id: str) -> str:
        import uuid

        with get_sessionmaker()() as session:
            start.wait()
            try:
                user_service.set_role(session, uuid.UUID(user_id), Role.USER)
            except user_service.LastAdmin:
                return "refused"
            return "demoted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(demote, [a, b]))
    assert outcomes == ["demoted", "refused"]
    assert admin_count(db) == 1


def test_unknown_accounts_and_bad_roles_are_rejected(client: TestClient, admin: Headers) -> None:
    ghost = "00000000-0000-4000-8000-000000000000"
    assert set_role(client, admin, ghost, "user").status_code == 404
    assert client.delete(f"/api/users/{ghost}", headers=admin).status_code == 404
    boss = me(client, admin)["id"]
    assert set_role(client, admin, boss, "superuser").status_code == 422
    assert (
        client.patch(f"/api/users/{boss}", headers=admin, json={"role": "user", "x": 1}).status_code
        == 422
    )
    assert (
        client.patch("/api/users/not-an-id", headers=admin, json={"role": "user"}).status_code
        == 422
    )


# --- Remove ---------------------------------------------------------------------------------------------


def test_an_administrator_cannot_delete_their_own_account(
    client: TestClient, admin: Headers, user: Headers
) -> None:
    set_role(client, admin, me(client, user)["id"], "admin")
    response = client.delete(f"/api/users/{me(client, admin)['id']}", headers=admin)
    assert response.status_code == 409 and "own account" in response.json()["detail"]


def test_the_only_administrator_cannot_be_deleted_by_anyone(
    client: TestClient, admin: Headers, user: Headers, db: Session
) -> None:
    """Through the API this is reachable only by a race (the caller is an administrator, so there
    are two), hence a direct call: some other account tries to remove the one administrator."""
    import uuid

    boss, worker = uuid.UUID(me(client, admin)["id"]), uuid.UUID(me(client, user)["id"])
    with pytest.raises(user_service.LastAdmin):
        user_service.delete_user(db, actor_id=worker, user_id=boss)
    assert admin_count(db) == 1


def test_deleting_an_account_removes_its_conversations_but_keeps_its_documents_and_measurements(
    client: TestClient,
    admin: Headers,
    user: Headers,
    library: dict,
    fake_llm: FakeChatModel,
    db: Session,
) -> None:
    worker = me(client, user)["id"]
    uploaded = one(client, user, "worker-notes.md", b"# Notes\n\nThe worker wrote this note.\n")
    fake_llm.reply("Twenty five days [1].")
    ask(client, user, LEAVE_Q)
    assert db.scalar(select(func.count()).select_from(Conversation)) == 1

    assert client.delete(f"/api/users/{worker}", headers=admin).status_code == 204

    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Conversation)) == 0
    kept = db.get(Document, uploaded["id"])
    assert kept is not None and kept.owner_id is None, "shared knowledge outlives its uploader"
    event = db.scalars(select(QueryEvent)).one()
    assert event.user_id is None and event.question == LEAVE_Q
    assert "worker@example.com" not in users(client, admin)
    assert client.get("/api/auth/me", headers=user).status_code == 401, "their token stops working"
