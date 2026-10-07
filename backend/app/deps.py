"""FastAPI dependencies. Auth is applied through these, never ad hoc."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db.database import session_scope


def get_db() -> Iterator[Session]:
    """Provide a database session for one request."""
    yield from session_scope()
