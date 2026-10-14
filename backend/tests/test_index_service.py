"""Startup consistency: collection verification, interrupted jobs, BM25 reload, refusal to start."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import bm25, vectorstore
from app.db.models import Chunk, Document
from app.main import app
from app.services import index_service, ingestion
from app.services.index_service import IndexInconsistentError
from tests.conftest import REAL_RUN_STARTUP, FakeEmbeddings

TEXTS = [
    "Error ERR_404_NET means the upstream gateway could not be reached.",
    "The retry limit for API timeouts is three attempts with exponential backoff.",
]


@pytest.fixture(autouse=True)
def stack(
    db: Session, vectors: Any, fake_embeddings: FakeEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> list[uuid.UUID]:
    """Real Postgres and Chroma. Background workers are replaced by a recorder."""
    submitted: list[uuid.UUID] = []
    monkeypatch.setattr(ingestion, "submit", submitted.append)
    monkeypatch.setattr(ingestion, "start_workers", lambda: None)
    return submitted


def add_document(
    db: Session, status: str, with_vectors: bool, fake: FakeEmbeddings, texts: list[str] = TEXTS
) -> Document:
    document = Document(
        name=f"{status}.md", stored_path=f"{uuid.uuid4().hex}.md", mime_type="text/markdown", size_bytes=10,
        status=status, chunk_count=len(texts) if status == "indexed" else 0, department="IT",
        document_type="Policy", creation_date=date(2026, 1, 1), access_level="Internal",
    )  # fmt: skip
    db.add(document)
    db.flush()
    ids = [f"{document.id}_chunk_{n:04d}" for n in range(len(texts))]
    for n, (chunk_id, text) in enumerate(zip(ids, texts, strict=True)):
        db.add(
            Chunk(
                id=chunk_id,
                document_id=document.id,
                ordinal=n,
                content=text,
                department="IT",
                document_type="Policy",
                creation_date_int=20260101,
                access_level="Internal",
            )
        )
    db.commit()
    if with_vectors:
        vectorstore.upsert(
            ids, [fake.embed_query(t) for t in texts], texts,
            [vectorstore.chunk_metadata(document_id=str(document.id), document_name="x", department="IT", document_type="Policy", creation_date=date(2026, 1, 1), creation_date_int=20260101, access_level="Internal", page_number=1, page_end=1, section_title="") for _ in ids],
        )  # fmt: skip
    return document


# --- Healthy starts ----------------------------------------------------------------------------------


def test_an_empty_system_starts_cleanly() -> None:
    report = REAL_RUN_STARTUP()
    assert (report.indexed_chunks, report.vectors, report.requeued) == (0, 0, 0)
    assert report.collection == get_settings().collection_name
    assert len(bm25.get_index()) == 0


def test_startup_rebuilds_the_bm25_index_from_postgres(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    document = add_document(db, "indexed", True, fake_embeddings)
    assert len(bm25.get_index()) == 0, "BM25 lives in memory and starts empty"
    report = REAL_RUN_STARTUP()
    assert (report.indexed_chunks, report.vectors) == (2, 2)
    assert len(bm25.get_index()) == 2
    hit = bm25.get_index().search("ERR_404_NET", 3)[0]
    assert hit.chunk_id == f"{document.id}_chunk_0000"


def test_only_fully_indexed_documents_are_loaded_into_bm25(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    add_document(db, "indexed", True, fake_embeddings)
    add_document(db, "failed", False, fake_embeddings, ["Quokka text that must stay unsearchable."])
    REAL_RUN_STARTUP()
    assert bm25.get_index().search("quokka", 3) == []
    assert len(bm25.get_index()) == 2


# --- Interrupted work -----------------------------------------------------------------------------------------


def test_interrupted_documents_are_failed_and_cleaned_up_and_queued_ones_are_resubmitted(
    db: Session, fake_embeddings: FakeEmbeddings, stack: list[uuid.UUID]
) -> None:
    stuck = add_document(db, "processing", True, fake_embeddings)
    waiting = add_document(db, "queued", False, fake_embeddings, [])
    REAL_RUN_STARTUP()

    db.expire_all()
    stuck = db.get(Document, stuck.id)  # type: ignore[assignment]
    assert (
        stuck.status == "failed"
        and stuck.error_message == ingestion.INTERRUPTED_MESSAGE
        and stuck.chunk_count == 0
    )
    assert (
        db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == stuck.id)) == 0
    )
    assert vectorstore.ids_for_document(str(stuck.id)) == [], (
        "partial vectors must not stay searchable"
    )
    assert stack == [waiting.id]
    assert db.get(Document, waiting.id).status == "queued"  # type: ignore[union-attr]


# --- Refusing to start ------------------------------------------------------------------------------------------


def test_startup_refuses_when_postgres_has_chunks_but_chroma_is_empty(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    """What a changed EMBEDDING_DIM or a lost Chroma volume looks like: silent empty search otherwise."""
    add_document(db, "indexed", False, fake_embeddings)
    with pytest.raises(IndexInconsistentError, match="reembed"):
        REAL_RUN_STARTUP()


def test_chunks_without_a_vector_are_logged_as_an_error_but_do_not_stop_startup(
    db: Session, fake_embeddings: FakeEmbeddings, caplog: pytest.LogCaptureFixture
) -> None:
    add_document(db, "indexed", True, fake_embeddings)
    add_document(db, "indexed", False, fake_embeddings, ["A third chunk without a vector yet."])
    with caplog.at_level(logging.WARNING):
        report = REAL_RUN_STARTUP()
    assert (report.indexed_chunks, report.vectors) == (3, 2)
    (record,) = [r for r in caplog.records if "no vector" in r.getMessage()]
    assert record.levelno == logging.ERROR and "reembed" in record.getMessage()
    assert record.missing == 1  # type: ignore[attr-defined]


def test_vectors_without_a_chunk_are_only_a_warning(
    vectors: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.test_index_health import fill

    fill(vectors, 3)  # leftovers: nothing in Postgres points at them
    with caplog.at_level(logging.WARNING):
        report = REAL_RUN_STARTUP()
    assert (report.indexed_chunks, report.vectors) == (0, 3)
    (record,) = [r for r in caplog.records if "more vectors" in r.getMessage()]
    assert record.levelno == logging.WARNING


def test_startup_refuses_a_collection_built_for_another_embedding_setup(vectors: Any) -> None:
    vectors.drop_collection()
    vectors.get_client().create_collection(
        name=get_settings().collection_name,
        metadata={"embedding_model": "models/other", "embedding_dim": 768},
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    with pytest.raises(vectorstore.VectorStoreMismatchError):
        REAL_RUN_STARTUP()


def test_the_application_itself_refuses_to_start_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise IndexInconsistentError(
            "PostgreSQL holds 5 indexed chunks but the collection is empty"
        )

    monkeypatch.setattr(index_service, "run_startup", fail)
    with pytest.raises(IndexInconsistentError), TestClient(app):
        pass


def test_the_application_runs_startup_once_and_stops_workers_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(index_service, "run_startup", lambda: events.append("startup"))
    monkeypatch.setattr(ingestion, "shutdown_workers", lambda: events.append("shutdown"))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert events == ["startup"]
    assert events == ["startup", "shutdown"]


# --- Index health at startup -----------------------------------------------------------------------------------


def test_startup_repairs_vectors_that_search_cannot_reach(
    db: Session,
    fake_embeddings: FakeEmbeddings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    add_document(db, "indexed", True, fake_embeddings)
    repaired: list[list[str]] = []
    monkeypatch.setattr(vectorstore, "find_unreachable", lambda *a, **k: ["x1", "x2"])
    monkeypatch.setattr(
        vectorstore, "repair_unreachable", lambda ids: repaired.append(list(ids)) or len(ids)
    )
    with caplog.at_level(logging.WARNING):
        report = REAL_RUN_STARTUP()
    assert report.repaired == 2 and repaired == [["x1", "x2"]]
    assert any("unreachable by search" in r.getMessage() for r in caplog.records)


def test_a_healthy_startup_repairs_nothing(db: Session, fake_embeddings: FakeEmbeddings) -> None:
    add_document(db, "indexed", True, fake_embeddings)
    assert REAL_RUN_STARTUP().repaired == 0


def test_startup_skips_verification_for_very_large_collections(
    db: Session, fake_embeddings: FakeEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_document(db, "indexed", True, fake_embeddings)
    monkeypatch.setattr(index_service, "MAX_STARTUP_VERIFY_VECTORS", 1)

    def must_not_run(*args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("verification should have been skipped")

    monkeypatch.setattr(vectorstore, "find_unreachable", must_not_run)
    assert REAL_RUN_STARTUP().vectors == 2
