"""SQLAlchemy models. Columns follow TDD section 4.3.

PostgreSQL is the source of truth: Chroma and the BM25 index are derived from
``chunks`` and can be rebuilt (CLAUDE.md rule 3).

Enum-like columns are ``VARCHAR`` with named ``CHECK`` constraints rather than
PostgreSQL ENUM types, so adding a value later is a plain migration. Cascades
are enforced by the database (``ON DELETE``), which is what keeps a deleted
document from ever leaving half-searchable rows behind.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.constants import (
    ACCESS_LEVELS,
    AccessLevel,
    DocumentStatus,
    EntityType,
    EventKind,
    EventOutcome,
    FeedbackRating,
    InsightKind,
    InsightStatus,
    MessageRole,
    QuestionSource,
    Role,
)
from app.db.database import Base


def _in(column: str, values: type[StrEnum] | tuple[str, ...]) -> str:
    """Build ``column IN ('a', 'b')`` from an enum class or a tuple of strings."""
    items = [member.value for member in values] if isinstance(values, type) else list(values)
    return f"{column} IN ({', '.join(repr(item) for item in items)})"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        sa.Uuid, primary_key=True, default=uuid.uuid4, server_default=sa.text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())


class User(Base):
    """An account. ``email`` is stored lower-cased (enforced by a CHECK)."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(_in("role", Role), name="role_valid"),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(sa.String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(sa.String(255))
    role: Mapped[str] = mapped_column(
        sa.String(16), default=Role.USER.value, server_default=Role.USER.value
    )
    created_at: Mapped[datetime] = _created_at()


class Document(Base):
    """An uploaded file and its ingestion state."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(_in("status", DocumentStatus), name="status_valid"),
        CheckConstraint(_in("access_level", ACCESS_LEVELS), name="access_level_valid"),
        CheckConstraint("size_bytes >= 0", name="size_non_negative"),
        CheckConstraint("chunk_count >= 0", name="chunk_count_non_negative"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Shared corporate knowledge outlives the account that uploaded it.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(512))
    # Generated file name relative to UPLOAD_DIR; never a user-supplied path (rule 23).
    stored_path: Mapped[str] = mapped_column(sa.String(1024))
    mime_type: Mapped[str] = mapped_column(sa.String(255))
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger)
    status: Mapped[str] = mapped_column(
        sa.String(16),
        default=DocumentStatus.QUEUED.value,
        server_default=DocumentStatus.QUEUED.value,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(sa.Text)
    chunk_count: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    department: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    document_type: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    # NULL until indexing resolves it (explicit upload value > filename > file properties > upload date).
    creation_date: Mapped[date | None] = mapped_column(sa.Date)
    access_level: Mapped[str] = mapped_column(
        sa.String(16),
        default=AccessLevel.INTERNAL.value,
        server_default=AccessLevel.INTERNAL.value,
    )
    created_at: Mapped[datetime] = _created_at()

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """One retrievable passage.

    Metadata is denormalized from the document so the BM25 channel can apply
    exactly the same filter as Chroma (TDD 4.2, CLAUDE.md rule 6). ``id`` is the
    chunk id used in Chroma, e.g. ``<document uuid>_chunk_0014``.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        CheckConstraint(_in("access_level", ACCESS_LEVELS), name="access_level_valid"),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
        CheckConstraint("page_end >= page_number", name="page_range_ordered"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
    )

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(sa.Integer)
    content: Mapped[str] = mapped_column(sa.Text)
    # Formats without pages (DOCX, MD, HTML, TXT) use page 1.
    page_number: Mapped[int] = mapped_column(sa.Integer, default=1, server_default="1")
    page_end: Mapped[int] = mapped_column(sa.Integer, default=1, server_default="1")
    section_title: Mapped[str] = mapped_column(sa.String(500), default="", server_default="")
    department: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    document_type: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    # YYYYMMDD: Chroma range operators only work on numbers (TDD 3.2).
    creation_date_int: Mapped[int] = mapped_column(sa.Integer)
    access_level: Mapped[str] = mapped_column(sa.String(16))

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    """A chat thread owned by one user."""

    __tablename__ = "conversations"
    __table_args__ = (sa.Index("ix_conversations_user_id_updated_at", "user_id", "updated_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(
        sa.String(200), default="New conversation", server_default="New conversation"
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    """A user question or an assistant answer, with its citations and diagnostics."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(_in("role", MessageRole), name="role_valid"),
        sa.Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(sa.String(16))
    content: Mapped[str] = mapped_column(sa.Text)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    # Points on which the cited documents disagree:
    # [{topic, claims: [{n, document_id, document_name, claim}]}], where n is a number in
    # ``citations``. NULL when the check did not run or found nothing.
    conflicts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    # Questions offered after the answer: [{type, question, n}], n a number of the retrieved
    # passage that would answer it (kept in ``suggestions`` items as ``document_id`` too).
    suggestions: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    # Best dense cosine / reranker score seen by the guard (exposed for debugging).
    guard_score: Mapped[float | None] = mapped_column(sa.Float)
    timings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Set when citation verification had to strip invalid markers or output was truncated.
    flagged: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    created_at: Mapped[datetime] = _created_at()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    feedback: Mapped[list[Feedback]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True
    )


class Feedback(Base):
    """A user's thumbs up/down on an assistant message (one rating per user per message)."""

    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_id"),
        CheckConstraint(_in("rating", FeedbackRating), name="rating_valid"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rating: Mapped[str] = mapped_column(sa.String(4))
    comment: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = _created_at()

    message: Mapped[Message] = relationship(back_populates="feedback")


class DocumentInsight(Base):
    """A saved analysis of one document: a summary of one kind, or an extraction.

    ``params_key`` tells apart analyses of the same kind: the set of fields of an extraction, or
    the earlier version a ``key_changes`` report compares against (empty otherwise). The row is
    written ``pending`` before the model is called, so a restart can fail what it interrupted.
    ``content`` holds the text or items and the cited passages, so it goes with the document
    (``ON DELETE CASCADE``) and is also removed when a document it cites is deleted.
    """

    __tablename__ = "document_insights"
    __table_args__ = (
        UniqueConstraint("document_id", "kind", "params_key"),
        CheckConstraint(_in("kind", InsightKind), name="kind_valid"),
        CheckConstraint(_in("status", InsightStatus), name="status_valid"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(sa.String(16))
    params_key: Mapped[str] = mapped_column(sa.String(64), default="", server_default="")
    status: Mapped[str] = mapped_column(
        sa.String(16), default=InsightStatus.PENDING.value, server_default="pending"
    )
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(sa.Text)
    model: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()
    )


class KgEntity(Base):
    """A person, policy, course, concept... found in one document (the knowledge graph).

    ``key`` is the normalised name; entities with the same key in different documents are one node
    when the graph of several documents is shown or walked, which is what links the documents.
    Every row cascades from its document, so deleting a document removes its part of the graph.
    """

    __tablename__ = "kg_entities"
    __table_args__ = (
        UniqueConstraint("document_id", "key"),
        CheckConstraint(_in("type", EntityType), name="type_valid"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(200))
    key: Mapped[str] = mapped_column(sa.String(200), index=True)
    type: Mapped[str] = mapped_column(sa.String(16), default="other", server_default="other")
    description: Mapped[str] = mapped_column(sa.Text, default="", server_default="")
    created_at: Mapped[datetime] = _created_at()


class KgMention(Base):
    """An entity appears in a chunk: the link from the graph back to the source text."""

    __tablename__ = "kg_mentions"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class KgRelation(Base):
    """A relationship two entities of one document are stated to have, with its evidence chunk."""

    __tablename__ = "kg_relations"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "label", "chunk_id"),
        CheckConstraint("source_id <> target_id", name="not_self"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), index=True
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(sa.String(64))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    quote: Mapped[str] = mapped_column(sa.Text, default="", server_default="")


class QueryEvent(Base):
    """One chat turn or analysis run, as measured: the source of the evaluation dashboard.

    Written for every outcome, failures included, and never read back into an answer. It holds
    the question (shown to admins on the dashboard, and the source of "often asked" suggestions)
    and ids, never document text. ``conversation_id``/``message_id`` have no foreign key: the
    measurement outlives a deleted conversation. Old rows are pruned (``METRICS_RETENTION_DAYS``).
    """

    __tablename__ = "query_events"
    __table_args__ = (
        CheckConstraint(_in("kind", EventKind), name="kind_valid"),
        CheckConstraint(_in("outcome", EventOutcome), name="outcome_valid"),
        CheckConstraint(_in("source", QuestionSource), name="source_valid"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    kind: Mapped[str] = mapped_column(sa.String(16))
    outcome: Mapped[str] = mapped_column(sa.String(16))
    source: Mapped[str] = mapped_column(
        sa.String(16), default=QuestionSource.TYPED.value, server_default="typed"
    )
    error_code: Mapped[str | None] = mapped_column(sa.String(40))
    guard_reason: Mapped[str | None] = mapped_column(sa.String(40))
    best_similarity: Mapped[float | None] = mapped_column(sa.Float)
    mean_similarity: Mapped[float | None] = mapped_column(sa.Float)
    retrieved: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    cited: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    cited_documents: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    cited_document_ids: Mapped[list[str] | None] = mapped_column(JSONB)
    citations_removed: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    uncited: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    retried: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    partial: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    rewritten: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    conflicts: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    suggestions: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    input_tokens: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    total_tokens: Mapped[int] = mapped_column(sa.Integer, default=0, server_default="0")
    model: Mapped[str] = mapped_column(sa.String(100), default="", server_default="")
    total_ms: Mapped[float | None] = mapped_column(sa.Float)
    timings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    question: Mapped[str | None] = mapped_column(sa.Text)
    document_ids: Mapped[list[str] | None] = mapped_column(JSONB)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)
    message_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)
