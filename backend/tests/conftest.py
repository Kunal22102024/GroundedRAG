"""Shared test setup.

Tests never need network access or a Gemini key: the LLM and embeddings are
mocked in the tests that use them. Settings are forced to harmless dummy values
*before* any ``app`` module is imported, so a test can never reach the real
database even when run inside the Docker stack (where real values exist).

Database tests use a throwaway database (``groundedrag_test_<random>``) created
on the stack's PostgreSQL server, migrated with the real Alembic migrations and
dropped at the end of the session. They skip when no server is reachable, unless
``REQUIRE_STACK=1`` is set (as in CI): then a missing server is a failure, because a run in which
every database and Chroma test skipped proves nothing.
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

# Tests assert the documented defaults (one ingestion worker, temperature 0, 800-character chunks,
# the 0.60 threshold, ...). Those must not depend on whatever the container's .env overrides, so
# every setting is dropped from the environment except the infrastructure wiring, and the few
# values tests need are forced below.
from app.config import Settings  # noqa: E402  (building a Settings is lazy; this reads no env)

_KEEP_FROM_ENV = {"DATABASE_URL", "CHROMA_HOST", "CHROMA_PORT"}
for _field in Settings.model_fields:
    if _field.upper() not in _KEEP_FROM_ENV:
        os.environ.pop(_field.upper(), None)

# Force (not setdefault): inside the container the real DATABASE_URL is set.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@127.0.0.1:1/test"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret-0123456789"
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "INFO"
os.environ["BCRYPT_ROUNDS"] = "4"  # minimum cost; real hashing speed is not under test
os.environ["EMBEDDING_MAX_RETRIES"] = "3"
os.environ["EMBEDDING_TEXTS_PER_MINUTE"] = "0"  # the limiter has its own tests
# Follow-up suggestions add a model call after every answer, which would change every chat test's
# scripted replies and call counts; tests/test_suggestions.py switches them on.
os.environ["FOLLOWUP_SUGGESTIONS"] = "false"
# The saved-answer cache would answer a repeated question without calling the scripted model, which
# changes call counts in every chat test; tests/test_semantic_cache.py switches it on.
os.environ["SEMANTIC_CACHE"] = "false"
# A collection name no real deployment uses, unique per run, so tests never touch real vectors.
os.environ["CHROMA_COLLECTION"] = f"test_{uuid.uuid4().hex[:8]}"


def stack_unavailable(reason: str) -> None:
    """Skip the test, or fail it when ``REQUIRE_STACK=1`` says the stack must be there."""
    if os.environ.get("REQUIRE_STACK") == "1":
        pytest.fail(f"{reason} (REQUIRE_STACK=1)")
    pytest.skip(reason)


TEST_DB_PREFIX = "groundedrag_test_"
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"

# The real startup function, captured before the autouse fixture below replaces it.
from app.services import index_service as _index_service  # noqa: E402

REAL_RUN_STARTUP = _index_service.run_startup


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
        stack_unavailable(
            "no PostgreSQL server configured (run the tests inside the compose stack)"
        )
    admin = create_engine(_SERVER_URL, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        admin.dispose()
        stack_unavailable("PostgreSQL server is not reachable")

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


# --- Startup, derived indexes, embeddings, uploads -----------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _no_startup() -> Iterator[None]:
    """App lifespan startup talks to Postgres and Chroma; tests call REAL_RUN_STARTUP instead.

    Session scoped on purpose: a module-scoped ``TestClient`` fixture starts the app before any
    function-scoped fixture runs, and would otherwise execute the real startup against whatever
    rows an earlier test left behind.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr(_index_service, "run_startup", lambda: None)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def _isolated_derived_state() -> Iterator[None]:
    """Every test starts with an empty BM25 index, no cached embedding or readiness state, and
    no after-indexing hooks: a test that runs the real startup registers the automatic summary,
    which would otherwise make model calls in every later test that indexes a document."""
    from app.core import answer_cache, bm25, embeddings
    from app.services import analysis_service, health_service, ingestion

    bm25.reset_index()
    answer_cache.get_cache().clear()
    embeddings.clear_caches()
    health_service.reset_embedding_cache()
    health_service.reset_index_cache()
    hooks = list(ingestion._indexed_hooks)
    ingestion._indexed_hooks.clear()
    yield
    bm25.reset_index()
    answer_cache.get_cache().clear()
    ingestion._indexed_hooks[:] = hooks
    analysis_service.shutdown()


@pytest.fixture
def vectors() -> Iterator[object]:
    """The vector store module, with a clean, verified test collection (skips without Chroma)."""
    import contextlib

    from app.core import vectorstore

    try:
        vectorstore.heartbeat()
    except vectorstore.VectorStoreError:
        stack_unavailable("ChromaDB is not reachable (run the tests inside the compose stack)")
    vectorstore.reset_collection_cache()
    with contextlib.suppress(vectorstore.VectorStoreError):
        vectorstore.drop_collection()
    vectorstore.get_collection()
    yield vectorstore
    with contextlib.suppress(vectorstore.VectorStoreError):
        vectorstore.drop_collection()


def hashing_vector(text: str, dim: int) -> list[float]:
    """A deterministic bag-of-words embedding: texts sharing words have high cosine similarity."""
    import math
    import zlib

    from app.core import bm25

    vector = [0.0] * dim
    for token in bm25.tokenize(text):
        vector[zlib.crc32(token.encode()) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        vector[0], norm = 1.0, 1.0
    return [x / norm for x in vector]


class FakeEmbeddings:
    """Stands in for the Gemini wrapper: no network, no key, unit-length vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []
        self.error: Exception | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        self.document_batches.append(list(texts))
        return [hashing_vector(t, self.dim) for t in texts]

    def embed_query(self, text: str, *, interactive: bool = False) -> list[float]:
        if self.error is not None:
            raise self.error
        self.queries.append(text)
        return hashing_vector(text, self.dim)


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddings:
    """Replace the embedding functions the services call."""
    from app.config import get_settings
    from app.core import embeddings

    fake = FakeEmbeddings(get_settings().embedding_dim)
    monkeypatch.setattr(embeddings, "embed_documents", fake.embed_documents)
    monkeypatch.setattr(embeddings, "embed_query", fake.embed_query)
    return fake


@pytest.fixture
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private upload directory for the test."""
    from app.config import get_settings

    directory = tmp_path / "uploads"
    monkeypatch.setattr(get_settings(), "upload_dir", directory)
    return directory


@pytest.fixture
def deferred_ingestion(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Record submitted documents without running them (they stay ``queued``)."""
    from app.services import ingestion

    submitted: list[uuid.UUID] = []
    monkeypatch.setattr(ingestion, "submit", submitted.append)
    return submitted


@pytest.fixture
def inline_ingestion(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Run each submitted document's ingestion immediately, in the calling thread."""
    from app.services import ingestion

    submitted: list[uuid.UUID] = []

    def run(document_id: uuid.UUID) -> None:
        submitted.append(document_id)
        ingestion.ingest_document(document_id)

    monkeypatch.setattr(ingestion, "submit", run)
    return submitted


# --- Chat model ------------------------------------------------------------------------------


class FakeChatModel:
    """Stands in for ``ChatGoogleGenerativeAI``: scripted replies, no network, counts its calls.

    Queue replies with ``reply``/``fail``/``block``. Every ``astream`` call is recorded in
    ``calls`` (the messages sent), so a test can prove the LLM was or was not called. With an
    empty script the model answers with the refusal sentence.
    """

    def __init__(self) -> None:
        self.script: list[object] = []
        self.calls: list[list[object]] = []
        self.slept: list[float] = []

    def reply(self, *pieces: str, finish: str = "STOP") -> FakeChatModel:
        self.script.append({"pieces": list(pieces), "finish": finish})
        return self

    def fail(self, exc: Exception) -> FakeChatModel:
        self.script.append(exc)
        return self

    def block(self) -> FakeChatModel:
        self.script.append({"block": True})
        return self

    @property
    def answer_calls(self) -> int:
        """Calls that were not follow-up rewrites."""
        return sum(1 for call in self.calls if "LAST USER QUESTION" not in str(call[-1].content))

    async def astream(self, messages: list[object]):  # noqa: ANN201 - async generator
        from langchain_core.messages import AIMessageChunk

        from app.constants import REFUSAL_MESSAGE

        self.calls.append(list(messages))
        step = (
            self.script.pop(0) if self.script else {"pieces": [REFUSAL_MESSAGE], "finish": "STOP"}
        )
        if isinstance(step, Exception):
            raise step
        if isinstance(step, dict) and step.get("block"):
            yield AIMessageChunk(
                content="", response_metadata={"prompt_feedback": {"block_reason": "SAFETY"}}
            )
            return
        pieces = step["pieces"]  # type: ignore[index]
        if not pieces:
            yield AIMessageChunk(content="", response_metadata={"finish_reason": step["finish"]})  # type: ignore[index]
        for index, piece in enumerate(pieces):
            last = index == len(pieces) - 1
            yield AIMessageChunk(
                content=piece,
                response_metadata={"finish_reason": step["finish"]} if last else {},  # type: ignore[index]
                usage_metadata={"input_tokens": 120, "output_tokens": 12, "total_tokens": 132}
                if last
                else None,
            )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    """Replace the chat model (answering and rewriting) and make rate-limit waits instant."""
    from app.core import llm

    fake = FakeChatModel()
    monkeypatch.setattr(llm, "get_chat_model", lambda kind="answer": fake)

    async def no_sleep(seconds: float) -> None:
        fake.slept.append(seconds)

    monkeypatch.setattr(llm, "_sleep", no_sleep)
    return fake
