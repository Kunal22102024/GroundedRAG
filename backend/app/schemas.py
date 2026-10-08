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
    Role,
)

# Deliberately permissive: enterprise domains such as company.local must work, and
# the real proof of an address is delivery, not syntax.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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
        email = value.strip().lower()
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

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value
