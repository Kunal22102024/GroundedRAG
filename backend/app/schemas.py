"""Pydantic request/response models. API contracts live here."""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.constants import (
    PASSWORD_MAX_BYTES,
    PASSWORD_MIN_CHARS,
    QUESTION_MAX_CHARS,
    AccessLevel,
    DocumentStatus,
    FeedbackRating,
    Role,
)

# Deliberately permissive: enterprise domains such as company.local must work, and
# the real proof of an address is delivery, not syntax.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def reject_nul(value: str) -> str:
    """PostgreSQL text cannot hold a NUL character; refuse it here instead of failing later with
    a 500 (for a chat question, after the model has already been paid for)."""
    if "\x00" in value:
        raise ValueError("must not contain NUL characters")
    return value


class HealthResponse(BaseModel):
    """Liveness probe result."""

    status: Literal["ok"]
    version: str


class CheckStatus(BaseModel):
    """Result of one readiness dependency check."""

    ok: bool
    # Short, non-sensitive explanation (e.g. an exception class name).
    detail: str | None = None


class ReadyResponse(BaseModel):
    """Readiness probe result: one entry per dependency."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]


# --- Authentication -----------------------------------------------------------


class _EmailPassword(BaseModel):
    """Shared email handling: trimmed, lower-cased, loosely validated."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(max_length=320)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        email = reject_nul(value).strip().lower()
        if not _EMAIL_PATTERN.match(email):
            raise ValueError("must be a valid email address")
        return email


class SignupRequest(_EmailPassword):
    """New account. The password is length-checked in bytes because bcrypt is."""

    password: str = Field(min_length=PASSWORD_MIN_CHARS)

    @field_validator("password")
    @classmethod
    def _fits_bcrypt(cls, value: str) -> str:
        if len(value.encode()) > PASSWORD_MAX_BYTES:
            raise ValueError(f"must be at most {PASSWORD_MAX_BYTES} bytes")
        return value


class LoginRequest(_EmailPassword):
    """Credentials. No strength rules here: only signup enforces them."""

    password: str = Field(min_length=1, max_length=1024)


class UserOut(BaseModel):
    """A user as shown to clients. Never includes the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    role: Role
    created_at: datetime


class UserAdminOut(UserOut):
    """An account on the administrators' list, with what it owns."""

    documents: int = 0
    conversations: int = 0


class RoleUpdate(BaseModel):
    """New role for an account."""

    model_config = ConfigDict(extra="forbid")

    role: Role


class AuthResponse(BaseModel):
    """Result of signup or login."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds
    user: UserOut


# --- Documents ----------------------------------------------------------------------


class DocumentOut(BaseModel):
    """A document and its indexing state."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    chunk_count: int
    department: str
    document_type: str
    # Null until indexing resolves it (upload value > file name > file properties > upload date).
    creation_date: date | None
    access_level: AccessLevel
    owner_id: uuid.UUID | None
    created_at: datetime
    # State of the automatic executive summary: none, pending, ready or failed.
    executive_summary: Literal["none", "pending", "ready", "failed"] = "none"


class LibrarySummaryOut(BaseModel):
    """Totals across every document the user may see, ignoring list filters."""

    total_files: int
    total_chunks: int
    by_status: dict[str, int]


class DocumentListResponse(BaseModel):
    """One page of documents."""

    items: list[DocumentOut]
    total: int
    limit: int
    offset: int
    summary: LibrarySummaryOut


class RejectedFileOut(BaseModel):
    """A file in an upload that was refused."""

    filename: str
    reason: str


class UploadResponse(BaseModel):
    """Result of an upload: accepted files are queued for indexing."""

    documents: list[DocumentOut]
    rejected: list[RejectedFileOut]


# --- Chat -----------------------------------------------------------------------------


class ChatFilters(BaseModel):
    """Optional narrowing filters (PRD 4.2). The caller's access levels are always applied on top."""

    model_config = ConfigDict(extra="forbid")

    department: str | None = Field(default=None, max_length=100)
    document_type: str | None = Field(default=None, max_length=100)
    date_from: date | None = None
    date_to: date | None = None
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=200)

    @field_validator("department", "document_type")
    @classmethod
    def _no_nul(cls, value: str | None) -> str | None:
        return None if value is None else reject_nul(value)

    @model_validator(mode="after")
    def _dates_ordered(self) -> ChatFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class ChatRequest(BaseModel):
    """One question, optionally continuing a conversation and narrowed by filters."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    conversation_id: uuid.UUID | None = None
    filters: ChatFilters | None = None
    # How the question was asked, for the dashboard: typed, or picked from a suggestion.
    source: Literal["typed", "suggestion", "followup"] = "typed"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return reject_nul(value)


# --- Conversations and feedback -------------------------------------------------------


class ConversationOut(BaseModel):
    """A conversation in the sidebar list."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    """One stored message; citations are the same objects the chat stream sent."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, object]] | None = None
    # Where the documents behind the answer disagree: [{topic, claims: [{n, document_id, claim}]}].
    conflicts: list[dict[str, object]] | None = None
    # Questions offered after the answer: [{type, question, document_id}].
    suggestions: list[dict[str, object]] | None = None
    guard_score: float | None = None
    flagged: bool = False
    # True when this answer was served from the saved-answer cache (no search, no model call).
    cached: bool = False
    created_at: datetime
    feedback: Literal["up", "down"] | None = None


class ConversationDetail(ConversationOut):
    """A conversation with all of its messages, oldest first."""

    messages: list[MessageOut]


class RenameRequest(BaseModel):
    """New conversation title."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _no_nul(cls, value: str) -> str:
        return reject_nul(value)


class FeedbackRequest(BaseModel):
    """Thumbs up/down, with an optional comment."""

    model_config = ConfigDict(extra="forbid")

    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("comment")
    @classmethod
    def _no_nul(cls, value: str | None) -> str | None:
        return None if value is None else reject_nul(value)


# --- Analysis: summaries, extraction, comparison ------------------------------------------------

SummaryKind = Literal["executive", "chapters", "faq", "glossary", "key_changes"]


class SummaryRequest(BaseModel):
    """Summarize one document. ``compare_to`` (an older version) is for ``key_changes`` only."""

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    kind: SummaryKind
    compare_to: uuid.UUID | None = None
    refresh: bool = False

    @model_validator(mode="after")
    def _compare_only_for_changes(self) -> SummaryRequest:
        if self.compare_to is not None and self.kind != "key_changes":
            raise ValueError("compare_to is only for the key_changes summary")
        if self.compare_to == self.document_id:
            raise ValueError("compare_to must be a different document")
        return self


class ExtractRequest(BaseModel):
    """Extract fields from one document: preset keys (``dates``, ``parties``...) or any name."""

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    fields: list[str] = Field(min_length=1, max_length=50)
    refresh: bool = False

    @field_validator("fields")
    @classmethod
    def _field_names(cls, value: list[str]) -> list[str]:
        for name in value:
            reject_nul(name)
            if len(name) > 200:
                raise ValueError("a field name is too long")
        return value


class CompareRequest(BaseModel):
    """One question, asked of each chosen document on its own."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    document_ids: list[uuid.UUID] = Field(min_length=2, max_length=8)

    @field_validator("question")
    @classmethod
    def _clean_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return reject_nul(value)


class QueryEventOut(BaseModel):
    """A failed, refused or flagged query on the dashboard (admins only)."""

    id: uuid.UUID
    created_at: datetime
    kind: str
    outcome: str
    source: str
    user_email: str | None = None
    question: str | None = None
    error_code: str | None = None
    guard_reason: str | None = None
    best_similarity: float | None = None
    cited: int = 0
    citations_removed: int = 0
    uncited: bool = False
    retried: bool = False
    partial: bool = False
    total_ms: float | None = None
    total_tokens: int = 0
    conversation_id: uuid.UUID | None = None
    document_ids: list[str] | None = None


class DocumentQuestionOut(BaseModel):
    """A question a chosen document answers."""

    question: str
    document_id: uuid.UUID
    document_name: str


class PopularQuestionOut(BaseModel):
    """A question several people asked and got a cited answer to."""

    question: str
    asked_by: int


class SuggestionsOut(BaseModel):
    """Questions to offer before the first question of a chat."""

    documents: list[DocumentQuestionOut]
    popular: list[PopularQuestionOut]
    # Documents whose suggested questions are being written: ask again shortly.
    pending: list[uuid.UUID]


class GraphDocumentOut(BaseModel):
    """How one document's knowledge graph is doing."""

    id: uuid.UUID
    name: str
    status: Literal["none", "pending", "ready", "failed"]
    entities: int = 0
    relations: int = 0
    # Share of the document's chunks the model read (1.0 = all of them).
    coverage: float | None = None
    error: str | None = None


class GraphNodeOut(BaseModel):
    """An entity, merged by name across the documents shown."""

    id: str
    name: str
    type: str
    description: str
    document_ids: list[str]
    mentions: int
    degree: int


class GraphEdgeOut(BaseModel):
    """Relations between two entities with one label (``count`` of them)."""

    source: str
    target: str
    label: str
    count: int


class GraphViewOut(BaseModel):
    """The knowledge graph of the chosen documents."""

    documents: list[GraphDocumentOut]
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    total_nodes: int
    truncated: bool


class EntityNeighbourOut(BaseModel):
    id: str
    name: str
    type: str


class EntityRelationOut(BaseModel):
    """A relation of an entity, in the direction it was stated, with the passage that states it."""

    label: str
    direction: Literal["out", "in"]
    quote: str
    other: EntityNeighbourOut
    evidence: dict[str, object]


class EntityDocumentOut(BaseModel):
    id: uuid.UUID
    name: str


class EntityDetailOut(BaseModel):
    """One entity: where it is mentioned and how it relates to others (passages open in the drawer)."""

    id: str
    name: str
    type: str
    description: str
    documents: list[EntityDocumentOut]
    mentions: list[dict[str, object]]
    relations: list[EntityRelationOut]


class GraphBuildOut(BaseModel):
    """A graph build was queued."""

    status: Literal["pending"]


class InsightOut(BaseModel):
    """A saved analysis of a document."""

    id: uuid.UUID
    kind: str
    params_key: str
    status: Literal["pending", "ready", "failed"]
    content: dict[str, object] | None = None
    error_message: str | None = None
    updated_at: datetime
