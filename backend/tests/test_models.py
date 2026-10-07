"""Schema behavior: defaults, constraints and cascades, against a real PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Conversation, Document, Feedback, Message, User


def add_user(db: Session, email: str = "a@example.com", role: str = "user") -> User:
    user = User(email=email, password_hash="x", role=role)
    db.add(user)
    db.flush()
    return user


def add_document(db: Session, owner: User | None = None, **overrides: object) -> Document:
    fields: dict[str, object] = {
        "owner_id": owner.id if owner else None,
        "name": "policy.pdf",
        "stored_path": f"{uuid.uuid4()}.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1234,
        "creation_date": date(2026, 1, 15),
    }
    fields.update(overrides)
    document = Document(**fields)
    db.add(document)
    db.flush()
    return document


def add_chunk(db: Session, document: Document, ordinal: int = 0, **overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "id": f"{document.id}_chunk_{ordinal:04d}",
        "document_id": document.id,
        "ordinal": ordinal,
        "content": "Server reboot windows are 02:00-04:00 UTC.",
        "creation_date_int": 20260115,
        "access_level": "Internal",
    }
    fields.update(overrides)
    chunk = Chunk(**fields)
    db.add(chunk)
    db.flush()
    return chunk


def add_message(db: Session, conversation: Conversation, role: str = "assistant") -> Message:
    message = Message(conversation_id=conversation.id, role=role, content="hello")
    db.add(message)
    db.flush()
    return message


def assert_rejected(db: Session, action: object) -> None:
    """The database must refuse the statement; the session is reusable afterwards."""
    with pytest.raises(IntegrityError):
        action()  # type: ignore[operator]
    db.rollback()


# --- users --------------------------------------------------------------------


def test_user_defaults(db: Session) -> None:
    user = add_user(db)
    db.refresh(user)
    assert isinstance(user.id, uuid.UUID)
    assert user.role == "user"
    assert user.created_at is not None


def test_ids_are_generated_by_the_database_too(db: Session) -> None:
    """Raw inserts (migrations, psql) get a UUID without any Python involved."""
    new_id = db.scalar(
        text(
            "INSERT INTO users (email, password_hash) VALUES ('raw@example.com', 'x') RETURNING id"
        )
    )
    assert isinstance(new_id, uuid.UUID)


def test_email_must_be_unique(db: Session) -> None:
    add_user(db, "dup@example.com")
    assert_rejected(db, lambda: add_user(db, "dup@example.com"))


def test_email_must_be_stored_lowercase(db: Session) -> None:
    assert_rejected(db, lambda: add_user(db, "Mixed@Example.com"))


def test_role_must_be_known(db: Session) -> None:
    assert_rejected(db, lambda: add_user(db, role="superuser"))


# --- documents ----------------------------------------------------------------


def test_document_defaults(db: Session) -> None:
    document = add_document(db)
    db.refresh(document)
    assert document.status == "queued"
    assert document.chunk_count == 0
    assert document.access_level == "Internal"
    assert (document.department, document.document_type) == ("", "")
    assert document.error_message is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "done"},
        {"access_level": "Secret"},
        {"access_level": "internal"},  # exact case matters: it is also the Chroma filter value
        {"size_bytes": -1},
        {"chunk_count": -1},
    ],
)
def test_document_constraints(db: Session, overrides: dict[str, object]) -> None:
    assert_rejected(db, lambda: add_document(db, **overrides))


def test_document_survives_deletion_of_its_owner(db: Session) -> None:
    owner = add_user(db)
    document = add_document(db, owner)
    db.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner.id})
    db.expire_all()
    assert db.get(Document, document.id).owner_id is None  # type: ignore[union-attr]


# --- chunks -------------------------------------------------------------------


def test_deleting_a_document_deletes_its_chunks(db: Session) -> None:
    """Rule 24: a deleted document must leave no searchable rows behind."""
    document = add_document(db)
    add_chunk(db, document, 0)
    add_chunk(db, document, 1)
    other = add_document(db)
    add_chunk(db, other, 0)

    db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document.id})
    db.expire_all()

    remaining = db.scalars(select(Chunk.document_id)).all()
    assert remaining == [other.id]


def test_orm_delete_cascades_through_the_database(db: Session) -> None:
    document = add_document(db)
    add_chunk(db, document, 0)
    db.commit()
    db.delete(document)
    db.commit()
    assert db.scalars(select(Chunk)).all() == []


def test_chunk_defaults(db: Session) -> None:
    chunk = add_chunk(db, add_document(db))
    db.refresh(chunk)
    assert (chunk.page_number, chunk.page_end) == (1, 1)
    assert chunk.section_title == ""


def test_chunk_ordinal_is_unique_per_document(db: Session) -> None:
    document = add_document(db)
    add_chunk(db, document, 0)
    assert_rejected(db, lambda: add_chunk(db, document, 0, id=f"{document.id}_other"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"page_number": 0, "page_end": 0},
        {"page_number": 5, "page_end": 4},  # range must not run backwards
        {"access_level": "Secret"},
        {"ordinal": -1},
    ],
)
def test_chunk_constraints(db: Session, overrides: dict[str, object]) -> None:
    document = add_document(db)
    assert_rejected(db, lambda: add_chunk(db, document, **overrides))


def test_chunk_page_range_may_span_pages(db: Session) -> None:
    chunk = add_chunk(db, add_document(db), page_number=3, page_end=4)
    assert (chunk.page_number, chunk.page_end) == (3, 4)


# --- conversations, messages, feedback ------------------------------------------


def test_deleting_a_conversation_deletes_messages_and_feedback(db: Session) -> None:
    user = add_user(db)
    conversation = Conversation(user_id=user.id)
    db.add(conversation)
    db.flush()
    message = add_message(db, conversation)
    db.add(Feedback(message_id=message.id, user_id=user.id, rating="up"))
    db.flush()

    db.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conversation.id})
    assert db.scalar(text("SELECT count(*) FROM messages")) == 0
    assert db.scalar(text("SELECT count(*) FROM feedback")) == 0


def test_deleting_a_user_deletes_their_conversations(db: Session) -> None:
    user = add_user(db)
    keeper = add_user(db, "keeper@example.com")
    db.add_all([Conversation(user_id=user.id), Conversation(user_id=keeper.id)])
    db.flush()
    db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
    assert db.scalars(select(Conversation.user_id)).all() == [keeper.id]


def test_conversation_defaults_and_updated_at_moves(db: Session) -> None:
    user = add_user(db)
    conversation = Conversation(user_id=user.id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    assert conversation.title == "New conversation"
    created, first_update = conversation.created_at, conversation.updated_at

    conversation.title = "Parental leave"
    db.commit()
    db.refresh(conversation)
    assert conversation.updated_at > first_update
    assert conversation.created_at == created


def test_message_role_must_be_user_or_assistant(db: Session) -> None:
    conversation = Conversation(user_id=add_user(db).id)
    db.add(conversation)
    db.flush()
    assert_rejected(db, lambda: add_message(db, conversation, role="system"))


def test_message_json_columns_round_trip(db: Session) -> None:
    conversation = Conversation(user_id=add_user(db).id)
    db.add(conversation)
    db.flush()
    citations = [{"n": 1, "document": "policy.pdf", "page": 14, "section": "Patching"}]
    timings = {"embed_ms": 41.5, "generate_ms": 910.2}
    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content="answer [1]",
            citations=citations,
            timings=timings,
            guard_score=0.62,
        )
    )
    db.commit()
    stored = db.scalars(select(Message)).one()
    assert stored.citations == citations
    assert stored.timings == timings
    assert stored.guard_score == pytest.approx(0.62)
    assert stored.flagged is False


def test_conversation_messages_come_back_in_order(db: Session) -> None:
    conversation = Conversation(user_id=add_user(db).id)
    db.add(conversation)
    db.flush()
    for text_ in ("first", "second", "third"):
        db.add(Message(conversation_id=conversation.id, role="user", content=text_))
        db.commit()  # separate transactions so created_at differs
    db.refresh(conversation)
    assert [m.content for m in conversation.messages] == ["first", "second", "third"]


def test_one_feedback_rating_per_user_per_message(db: Session) -> None:
    user = add_user(db)
    conversation = Conversation(user_id=user.id)
    db.add(conversation)
    db.flush()
    message = add_message(db, conversation)
    db.add(Feedback(message_id=message.id, user_id=user.id, rating="up"))
    db.flush()
    assert_rejected(
        db,
        lambda: (
            db.add(Feedback(message_id=message.id, user_id=user.id, rating="down")),
            db.flush(),
        ),
    )


def test_feedback_rating_must_be_up_or_down(db: Session) -> None:
    user = add_user(db)
    conversation = Conversation(user_id=user.id)
    db.add(conversation)
    db.flush()
    message = add_message(db, conversation)
    assert_rejected(
        db,
        lambda: (
            db.add(Feedback(message_id=message.id, user_id=user.id, rating="meh")),
            db.flush(),
        ),
    )
