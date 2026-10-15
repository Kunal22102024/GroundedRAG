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

# Suggested values for the library filters. Free text is still accepted at upload
# (matched case-insensitively to these when possible); IT appears in the TDD examples.
DEPARTMENTS: tuple[str, ...] = ("HR", "Engineering", "Legal", "Operations", "IT")
DOCUMENT_TYPES: tuple[str, ...] = ("Policy", "Manual", "Technical Spec", "Contract")

# Role -> access levels that role may read (PRD 4.4, TDD section 8).
ROLE_ACCESS_LEVELS: dict[Role, tuple[AccessLevel, ...]] = {
    Role.ADMIN: (AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED),
    Role.USER: (AccessLevel.PUBLIC, AccessLevel.INTERNAL),
}


def allowed_access_levels(role: str) -> tuple[AccessLevel, ...]:
    """Access levels a role may read. Unknown roles get nothing (fail closed)."""
    try:
        return ROLE_ACCESS_LEVELS[Role(role)]
    except ValueError:
        return ()


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


class InsightKind(StrEnum):
    """What a saved analysis of a document is (``document_insights.kind``)."""

    EXECUTIVE = "executive"
    CHAPTERS = "chapters"
    FAQ = "faq"
    GLOSSARY = "glossary"
    KEY_CHANGES = "key_changes"
    EXTRACTION = "extraction"
    QUESTIONS = "questions"  # suggested questions the document answers
    GRAPH = "graph"  # build status of the document's knowledge graph (the graph is in kg_* tables)


class EntityType(StrEnum):
    """What a knowledge-graph entity is (``kg_entities.type``). Anything else is ``other``."""

    PERSON = "person"
    ORGANIZATION = "organization"
    ROLE = "role"
    CONCEPT = "concept"
    COURSE = "course"
    POLICY = "policy"
    PROCESS = "process"
    PRODUCT = "product"
    LOCATION = "location"
    EVENT = "event"
    OTHER = "other"


class InsightStatus(StrEnum):
    """Progress of a saved analysis: it is written ``pending`` before the model is called."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class EventKind(StrEnum):
    """What a ``query_events`` row measures: a chat turn or one analysis run."""

    CHAT = "chat"
    SUMMARY = "summary"
    EXTRACT = "extract"
    COMPARE = "compare"
    SUGGEST = "suggest"
    GRAPH = "graph"  # building a document's knowledge graph


class EventOutcome(StrEnum):
    """How a chat turn or analysis run ended."""

    ANSWERED = "answered"
    REFUSED = "refused"
    FAILED = "failed"


class QuestionSource(StrEnum):
    """How a chat question was asked: typed, or picked from a suggestion."""

    TYPED = "typed"
    SUGGESTION = "suggestion"
    FOLLOWUP = "followup"


class SuggestionType(StrEnum):
    """Kinds of question offered after an answer."""

    FOLLOW_UP = "follow_up"
    SUB_QUESTION = "sub_question"
    RELATED = "related"


class FeedbackRating(StrEnum):
    """Thumbs up/down on an assistant message."""

    UP = "up"
    DOWN = "down"


# --- Chat -----------------------------------------------------------------------------

# A question is embedded as one query, so it shares the embedding input ceiling below.
QUESTION_MAX_CHARS = 2000
# Follow-up rewriting sees the last few messages only; earlier turns add cost, not meaning.
REWRITE_HISTORY_MESSAGES = 6
REWRITE_HISTORY_ANSWER_CHARS = 400
REWRITE_MAX_TOKENS = 256
CONVERSATION_TITLE_MAX_CHARS = 80

# --- Uploads and ingestion -----------------------------------------------------

# Extension -> canonical media type (used when serving the original; never taken
# from the client).
ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}

# gemini-embedding-001 accepts 2,048 tokens per input and silently truncates the rest
# (CLAUDE.md rule 16). One token is at least one character in the worst case (CJK), so
# these character ceilings keep every input under the limit without a tokenizer.
CHUNK_HARD_MAX_CHARS = 1500
EMBEDDING_INPUT_MAX_CHARS = 2000

# Chunks shorter than this are merged into a neighbour when possible.
MIN_CHUNK_CHARS = 20

# A PDF with fewer than this share of text-bearing pages is treated as scanned.
SCANNED_PDF_MAX_TEXT_PAGE_RATIO = 0.2
# A page counts as text-bearing with at least this many non-space characters.
MIN_PAGE_TEXT_CHARS = 20

SCANNED_PDF_MESSAGE = (
    "This looks like a scanned PDF, OCR not supported (no extractable text found)."
)
