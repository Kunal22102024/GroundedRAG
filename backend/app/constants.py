"""Shared constants and enums. Import these; never retype the strings."""

from __future__ import annotations

from enum import StrEnum

API_PREFIX = "/api"

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
