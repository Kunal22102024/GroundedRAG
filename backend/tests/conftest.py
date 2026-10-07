"""Shared test setup.

Tests never need network access or a Gemini key: the LLM and embeddings are
mocked in the tests that use them. Settings are forced to harmless dummy values
*before* any ``app`` module is imported, so a test can never reach the real
database even when run inside the Docker stack (where real values exist).

Database tests use a throwaway database (``groundedrag_test_<random>``) created
on the stack's PostgreSQL server, migrated with the real Alembic migrations and
dropped at the end of the session. They skip when no server is reachable.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Captured BEFORE the override below: inside the container this is the real
# server. It is only ever used to CREATE/DROP throwaway databases.
_SERVER_URL = os.environ.get("DATABASE_URL")

# Force (not setdefault): inside the container the real DATABASE_URL is set.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@127.0.0.1:1/test"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret-0123456789"
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "INFO"
os.environ["BCRYPT_ROUNDS"] = "4"  # minimum cost; real hashing speed is not under test

TEST_DB_PREFIX = "groundedrag_test_"
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


class LogCollector(logging.Handler):
    """Collects log records emitted on the logger it is attached to."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def request_logs() -> Iterator[LogCollector]:
    """Capture records from the ``app.request`` access logger."""
    collector = LogCollector()
    logger = logging.getLogger("app.request")
    previous_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    yield collector
    logger.removeHandler(collector)
    logger.setLevel(previous_level)


# --- Throwaway PostgreSQL databases ---------------------------------------------


def _reset_app_caches() -> None:
    """Make the app rebuild settings, engine and sessionmaker from the environment."""
    from app.config import get_settings
    from app.db import database

    database.dispose_engine()
    get_settings.cache_clear()
    database.get_engine.cache_clear()
    database.get_sessionmaker.cache_clear()


@contextmanager
def use_database(url: str) -> Iterator[None]:
    """Point the whole app (settings, engine, Alembic) at ``url`` for the block."""
    previous = os.environ["DATABASE_URL"]
    os.environ["DATABASE_URL"] = url
    _reset_app_caches()
    try:
        yield
    finally:
        os.environ["DATABASE_URL"] = previous
        _reset_app_caches()


@contextmanager
def temporary_database() -> Iterator[str]:
    """Create an empty throwaway database on the server and yield its URL."""
    if not _SERVER_URL:
        pytest.skip("no PostgreSQL server configured (run the tests inside the compose stack)")
    admin = create_engine(_SERVER_URL, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        admin.dispose()
        pytest.skip("PostgreSQL server is not reachable")

    name = f"{TEST_DB_PREFIX}{uuid.uuid4().hex[:10]}"
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(_SERVER_URL).set(database=name).render_as_string(hide_password=False)
    try:
        yield url
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def run_migrations(target: str = "head", *, downgrade: bool = False) -> None:
    """Run Alembic against whatever database the app is currently pointed at."""
    from alembic.config import Config

    from alembic import command

    config = Config(str(ALEMBIC_INI))
    config.attributes["configure_logger"] = False  # keep the app's logging setup
    (command.downgrade if downgrade else command.upgrade)(config, target)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A migrated throwaway database that the app is pointed at for the session."""
    with temporary_database() as url, use_database(url):
        run_migrations("head")
        yield url


@pytest.fixture
def db(database_url: str) -> Iterator[Session]:
    """A session on the throwaway database, with every table emptied first."""
    from app.db.database import Base, get_engine, get_sessionmaker

    engine = get_engine()
    assert engine.url.database is not None
    assert engine.url.database.startswith(TEST_DB_PREFIX), "refusing to touch a non-test database"
    tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    with get_sessionmaker()() as session:
        yield session
