"""SQLAlchemy engine and session factory.

PostgreSQL is the source of truth (CLAUDE.md rule 3). The engine is created
lazily so importing this module never needs a live database or settings.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_CONNECT_TIMEOUT_SECONDS = 3

# Deterministic constraint names keep migrations reviewable and make later
# ALTERs possible without guessing auto-generated names.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine.

    ``hide_parameters`` keeps bound values (which can contain document text or
    user input) out of SQLAlchemy error messages and therefore out of logs
    (CLAUDE.md rule 21).
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def session_scope() -> Iterator[Session]:
    """Yield a session and always close it. Used by the ``get_db`` dependency."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def ping() -> None:
    """Run ``SELECT 1`` to prove Postgres is reachable. Raises on failure."""
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def dispose_engine() -> None:
    """Close pooled connections (called on shutdown)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
