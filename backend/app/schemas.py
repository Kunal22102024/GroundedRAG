"""Pydantic request/response models. API contracts live here."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants import PASSWORD_MAX_BYTES, PASSWORD_MIN_CHARS, Role

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
