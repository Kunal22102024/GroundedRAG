"""Shared constants and enums. Import these; never retype the strings."""

from __future__ import annotations

from enum import StrEnum

API_PREFIX = "/api"

JWT_ALGORITHM = "HS256"

# bcrypt only uses the first 72 bytes of a password and bcrypt 5 raises on more,
# so longer passwords are rejected at the API boundary instead of truncated.
PASSWORD_MIN_CHARS = 8
PASSWORD_MAX_BYTES = 72

# Returned verbatim, without calling the LLM, when the documents cannot answer
# (CLAUDE.md rules 8 and 11).
REFUSAL_MESSAGE = "Insufficient information available in the selected documents."


class Role(StrEnum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"


class AccessLevel(StrEnum):
    """Document access levels, stored as these exact strings in Postgres and Chroma."""

    PUBLIC = "Public"
    INTERNAL = "Internal"
    RESTRICTED = "Restricted"


ACCESS_LEVELS: tuple[str, ...] = tuple(level.value for level in AccessLevel)

# Role -> access levels that role may read (PRD 4.4, TDD section 8).
ROLE_ACCESS_LEVELS: dict[Role, tuple[AccessLevel, ...]] = {
    Role.ADMIN: (AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED),
    Role.USER: (AccessLevel.PUBLIC, AccessLevel.INTERNAL),
}


class DocumentStatus(StrEnum):
    """Ingestion state machine (TDD 3.4): queued -> processing -> indexed | failed."""

    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MessageRole(StrEnum):
    """Who authored a chat message."""

    USER = "user"
    ASSISTANT = "assistant"


class FeedbackRating(StrEnum):
    """Thumbs up/down on an assistant message."""

    UP = "up"
    DOWN = "down"
