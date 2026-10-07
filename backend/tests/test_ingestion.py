"""Ingestion end to end: real PostgreSQL and ChromaDB, fake embedder (no network, no key)."""

from __future__ import annotations

import io
import logging
import re
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import CHUNK_HARD_MAX_CHARS
from app.core import bm25, embeddings, parsers, vectorstore
from app.core.metadata import ExplicitMetadata
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document, User
from app.services import document_service, ingestion
from app.services.document_service import IncomingFile
from tests.conftest import FakeEmbeddings
from tests.pdf_factory import make_pdf
from tests.test_parsers import body

MD = (
    "# Retry Policy\n\nThe retry limit for API timeouts is three attempts with exponential backoff.\n\n"
    "## Errors\n\nError ERR_404_NET means the upstream gateway could not be reached.\n"
)


@pytest.fixture(autouse=True)
def stack(db: Session, upload_dir: Path, vectors: Any, fake_embeddings: FakeEmbeddings) -> None:
    """Real Postgres + Chroma, a private upload dir and a fake embedder for every test here."""


def make_user(db: Session, email: str = "u@example.com", role: str = "user") -> User:
    user = User(email=email, password_hash="x", role=role)
    db.add(user)
    db.commit()
    return user


def upload(
    db: Session, user: User, filename: str, content: bytes, explicit: ExplicitMetadata | None = None
) -> Document:
    result = document_service.create_documents(
        db,
        user,
        [IncomingFile(filename, None, io.BytesIO(content), len(content))],
        explicit or ExplicitMetadata(),
        submit=lambda _id: None,
    )
    assert result.accepted, result.rejected
    return result.accepted[0]


def fresh(db: Session, document: Document) -> Document:
    db.expire_all()
    return db.get(Document, document.id)  # type: ignore[return-value]


def chunk_rows(db: Session, document: Document) -> list[Chunk]:
    db.expire_all()
    return list(
        db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal))
    )


def assert_nothing_left_behind(db: Session, document: Document) -> None:
    assert chunk_rows(db, document) == []
    assert vectorstore.ids_for_document(str(document.id)) == []
    assert not [c for c in bm25.get_index().chunk_ids() if c.startswith(str(document.id))]


# --- The happy path ------------------------------------------------------------------------------------


def test_a_markdown_file_is_indexed_in_every_store(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    doc = upload(db, make_user(db), "IT/API_Policy_2026-01-15.md", MD.encode())
    assert doc.status == "queued" and doc.creation_date is None and doc.chunk_count == 0

    ingestion.ingest_document(doc.id)

    doc = fresh(db, doc)
    assert (doc.status, doc.chunk_count, doc.error_message) == ("indexed", 2, None)
    rows = chunk_rows(db, doc)
    ids = [f"{doc.id}_chunk_0000", f"{doc.id}_chunk_0001"]
    assert [r.id for r in rows] == ids
    assert [r.section_title for r in rows] == ["Retry Policy", "Errors"]
    assert set(vectorstore.ids_for_document(str(doc.id))) == set(ids) and vectorstore.count() == 2

    # metadata resolved from the folder, file name and date, and written back
    assert (doc.department, doc.document_type, doc.creation_date) == (
        "IT",
        "Policy",
        date(2026, 1, 15),
    )
    assert {(r.department, r.creation_date_int, r.access_level) for r in rows} == {
        ("IT", 20260115, "Internal")
    }

    stored = vectorstore.get_collection().get(ids=ids, include=["metadatas", "documents"])
    for metadata in stored["metadatas"]:
        assert metadata["department"] == "IT" and metadata["creation_date_int"] == 20260115
        assert (
            metadata["document_id"] == str(doc.id)
            and metadata["document_name"] == "API_Policy_2026-01-15.md"
        )
        assert all(isinstance(v, (str, int)) for v in metadata.values())
    assert set(stored["documents"]) == {r.content for r in rows}

    # both retrieval channels find the right chunk
    assert bm25.get_index().search("ERR_404_NET", 3)[0].chunk_id == ids[1]
    dense = vectorstore.query(fake_embeddings.embed_query("retry limit API timeouts attempts"), 1)[
        0
    ]
    assert dense.chunk_id == ids[0] and dense.similarity > 0.3


def test_explicit_upload_metadata_wins_over_everything(db: Session) -> None:
    admin = make_user(db, "boss@example.com", "admin")
    explicit = ExplicitMetadata("Legal", "Contract", date(2020, 5, 5), "Restricted")
    doc = upload(db, admin, "HR_Leave_Policy_2026.md", MD.encode(), explicit)
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert (doc.department, doc.document_type, doc.creation_date, doc.access_level) == (
        "Legal",
        "Contract",
        date(2020, 5, 5),
        "Restricted",
    )
    stored = vectorstore.get_collection().get(
        where={"document_id": str(doc.id)}, include=["metadatas"]
    )
    assert {m["access_level"] for m in stored["metadatas"]} == {"Restricted"}
    assert {c.access_level for c in chunk_rows(db, doc)} == {"Restricted"}


def test_pdf_chunks_report_the_physical_pages_their_text_is_on(db: Session) -> None:
    pages = [[("h", f"Section {n}"), ("p", f"Marker{n}x {body(n, 60)}")] for n in range(1, 7)]
    doc = upload(db, make_user(db), "manual.pdf", make_pdf(pages))
    ingestion.ingest_document(doc.id)

    rows = chunk_rows(db, fresh(db, doc))
    assert rows
    seen_pages = set()
    for row in rows:
        markers = [int(m) for m in re.findall(r"Marker(\d+)x", row.content)]
        assert markers, "every chunk here starts with a page marker"
        assert row.page_number <= min(markers) and row.page_end >= max(markers)
        assert row.page_number == min(markers) and row.page_end == max(markers)
        seen_pages.add(row.page_number)
    assert seen_pages == {1, 2, 3, 4, 5, 6}
    stored = vectorstore.get_collection().get(
        where={"document_id": str(doc.id)}, include=["metadatas"]
    )
    assert {m["page_number"] for m in stored["metadatas"]} == seen_pages


def test_every_text_sent_to_the_embedder_is_below_the_truncation_ceiling(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    giant = ("x" * 5000 + "\n\n" + "Long sentence without a break. " * 400).encode()
    doc = upload(db, make_user(db), "giant.txt", giant)
    ingestion.ingest_document(doc.id)
    sent = [t for batch in fake_embeddings.document_batches for t in batch]
    assert sent and max(len(t) for t in sent) <= CHUNK_HARD_MAX_CHARS
    assert fresh(db, doc).status == "indexed"


# --- Ordering and guarantees -------------------------------------------------------------------------------


def test_chunks_reach_postgres_before_anything_reaches_chroma(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int] = {}
    real_upsert = vectorstore.upsert

    def spy(**kwargs: Any) -> None:
        with (
            get_sessionmaker()() as other
        ):  # a separate connection: only committed rows are visible
            seen["postgres_rows"] = other.scalar(select(func.count()).select_from(Chunk)) or 0
        seen["vectors"] = vectorstore.count()
        real_upsert(**kwargs)

    monkeypatch.setattr(vectorstore, "upsert", spy)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    assert seen == {"postgres_rows": 2, "vectors": 0}


def test_the_job_moves_through_processing(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    states: list[str] = []

    def spy_parse(path: Path, extension: str) -> Any:
        with get_sessionmaker()() as other:
            states.append(other.scalar(select(Document.status)) or "")
        return real_parse(path, extension)

    real_parse = parsers.parse_document
    monkeypatch.setattr(parsers, "parse_document", spy_parse)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    assert states == ["processing"] and fresh(db, doc).status == "indexed"


@pytest.mark.parametrize("extension", [".md", ".txt"])
def test_documents_that_are_not_queued_are_left_alone(
    db: Session, fake_embeddings: FakeEmbeddings, extension: str
) -> None:
    doc = upload(db, make_user(db), f"a{extension}", MD.encode())
    ingestion.ingest_document(doc.id)
    calls, stored = len(fake_embeddings.document_batches), vectorstore.count()
    assert stored >= 1
    ingestion.ingest_document(doc.id)  # already indexed: must not run again
    assert len(fake_embeddings.document_batches) == calls and vectorstore.count() == stored


def test_a_document_deleted_before_it_runs_is_a_quiet_no_op(db: Session) -> None:
    doc = upload(db, make_user(db), "a.md", MD.encode())
    db.execute(delete(Document).where(Document.id == doc.id))
    db.commit()
    ingestion.ingest_document(doc.id)
    assert vectorstore.count() == 0 and len(bm25.get_index()) == 0


def test_a_busy_document_is_skipped(db: Session) -> None:
    doc = upload(db, make_user(db), "a.md", MD.encode())
    lock = ingestion.document_lock(doc.id)
    assert lock.acquire(blocking=False)
    try:
        ingestion.ingest_document(doc.id)
    finally:
        lock.release()
    assert fresh(db, doc).status == "queued"


def test_a_document_deleted_while_it_is_being_indexed_leaves_nothing_behind(
    db: Session, monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    doc = upload(db, make_user(db), "a.md", MD.encode())
    real = fake_embeddings.embed_documents

    def delete_then_embed(texts: list[str]) -> list[list[float]]:
        with get_sessionmaker()() as other:  # the document vanishes mid-job
            other.execute(delete(Document).where(Document.id == doc.id))
            other.commit()
        return real(texts)

    monkeypatch.setattr(embeddings, "embed_documents", delete_then_embed)
    ingestion.ingest_document(doc.id)
    assert vectorstore.count() == 0, "vectors written after the delete must be cleaned up"
    assert len(bm25.get_index()) == 0
    assert db.scalar(select(func.count()).select_from(Chunk)) == 0


# --- Failures: clear message, nothing half-indexed ------------------------------------------------------


def test_a_scanned_pdf_fails_with_the_ocr_message_and_leaves_no_index_entries(
    db: Session, upload_dir: Path
) -> None:
    doc = upload(db, make_user(db), "scan.pdf", make_pdf(["image", "image", "image"]))
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and doc.chunk_count == 0
    assert "scanned PDF, OCR not supported" in (doc.error_message or "")
    assert_nothing_left_behind(db, doc)
    assert (upload_dir / doc.stored_path).exists(), (
        "the file stays so the uploader can still see and delete it"
    )


def test_a_blank_pdf_fails_with_the_same_message(db: Session) -> None:
    doc = upload(db, make_user(db), "blank.pdf", make_pdf([None, None]))
    ingestion.ingest_document(doc.id)
    assert "OCR not supported" in (fresh(db, doc).error_message or "")


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("empty.txt", b"   \n\n  ", "no extractable text"),
        ("heading_only.md", b"# Just a heading\n", "no extractable text"),
        ("noise.txt", b"...\n---\n!!!", "no extractable text"),
    ],
)
def test_documents_without_text_fail_clearly(
    db: Session, filename: str, content: bytes, expected: str
) -> None:
    doc = upload(db, make_user(db), filename, content)
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and expected in (doc.error_message or "")
    assert_nothing_left_behind(db, doc)


def test_an_embedding_failure_removes_the_chunks_already_in_postgres(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    fake_embeddings.error = embeddings.EmbeddingRateLimitError(
        "The embedding service is rate limited (free-tier keys allow about 100 requests per minute)."
    )
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and "rate limited" in (doc.error_message or "")
    assert_nothing_left_behind(db, doc)


def test_a_rejected_api_key_is_reported_without_leaking_anything(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    fake_embeddings.error = embeddings.EmbeddingAuthError(
        "The embedding service rejected the API key. Check GOOGLE_API_KEY."
    )
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    assert "Check GOOGLE_API_KEY" in (fresh(db, doc).error_message or "")


def test_a_vector_store_failure_after_postgres_was_written_is_cleaned_up(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_: Any) -> None:
        raise vectorstore.VectorStoreError("ChromaDB upsert failed: ConnectError")

    monkeypatch.setattr(vectorstore, "upsert", boom)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and "search index is unavailable" in (doc.error_message or "")
    assert "ConnectError" not in (doc.error_message or "")
    assert_nothing_left_behind(db, doc)


def test_an_unexpected_error_gets_a_generic_message_and_is_logged_with_a_traceback(
    db: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(path: Path, extension: str) -> Any:
        raise ZeroDivisionError("TOPSECRET document content")

    monkeypatch.setattr(parsers, "parse_document", explode)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    with caplog.at_level(logging.WARNING):
        ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and "unexpectedly" in (doc.error_message or "")
    assert "TOPSECRET" not in (doc.error_message or "")
    record = next(r for r in caplog.records if r.getMessage() == "ingestion failed")
    assert (
        record.levelno == logging.ERROR
        and record.exc_info is not None
        and record.error == "ZeroDivisionError"
    )  # type: ignore[attr-defined]


def test_stale_vectors_of_a_failed_document_are_removed_too(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    doc = upload(db, make_user(db), "a.md", MD.encode())
    vectorstore.upsert(
        [f"{doc.id}_chunk_0099"],
        [fake_embeddings.embed_query("stale")],
        ["stale text"],
        [
            vectorstore.chunk_metadata(
                document_id=str(doc.id),
                document_name="a.md",
                department="",
                document_type="",
                creation_date=date(2026, 1, 1),
                creation_date_int=20260101,
                access_level="Internal",
                page_number=1,
                page_end=1,
                section_title="",
            )
        ],
    )
    fake_embeddings.error = embeddings.EmbeddingError("down")
    ingestion.ingest_document(doc.id)
    assert_nothing_left_behind(db, fresh(db, doc))


# --- Privacy ------------------------------------------------------------------------------------------------


def test_logs_contain_ids_counts_and_timings_but_never_names_or_text(
    db: Session, caplog: pytest.LogCaptureFixture
) -> None:
    doc = upload(db, make_user(db), "Salary_Review_Confidential.md", MD.encode())
    with caplog.at_level(logging.DEBUG):
        ingestion.ingest_document(doc.id)
    done = next(r for r in caplog.records if r.getMessage() == "document indexed")
    assert done.document_id == str(doc.id) and done.chunks == 2 and done.total_ms > 0  # type: ignore[attr-defined]
    assert {"parse_ms", "chunk_ms", "db_ms", "embed_ms", "index_ms"} <= set(done.__dict__)
    rendered = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    for forbidden in ("Salary_Review", "exponential backoff", "ERR_404_NET", "upstream gateway"):
        assert forbidden not in rendered


# --- The worker pool -----------------------------------------------------------------------------------------


def wait_until(predicate: Any, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_the_background_worker_processes_documents_one_at_a_time(
    db: Session, monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    active = {"now": 0, "max": 0}
    guard = threading.Lock()
    real = fake_embeddings.embed_documents

    def tracked(texts: list[str]) -> list[list[float]]:
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.15)
        try:
            return real(texts)
        finally:
            with guard:
                active["now"] -= 1

    monkeypatch.setattr(embeddings, "embed_documents", tracked)
    ingestion.shutdown_workers()
    user = make_user(db)
    docs = [
        upload(db, user, f"doc{n}.md", MD.replace("Retry", f"Retry{n}").encode()) for n in range(3)
    ]
    try:
        for doc in docs:
            ingestion.submit(doc.id)
        assert wait_until(lambda: all(fresh(db, d).status == "indexed" for d in docs)), (
            "documents were not indexed in time"
        )
    finally:
        ingestion.shutdown_workers()
    assert active["max"] == 1, "INGESTION_WORKERS=1 means one document at a time"
    assert vectorstore.count() == 6


def test_chunk_ids_are_stable_and_unique() -> None:
    document_id = uuid.uuid4()
    assert ingestion.chunk_id_for(document_id, 14) == f"{document_id}_chunk_0014"
    assert len({ingestion.chunk_id_for(document_id, n) for n in range(2000)}) == 2000
    assert get_settings().chunk_target_chars == 800


# --- Index health: new vectors must be provably searchable ---------------------------------------------------


def test_new_vectors_are_verified_searchable_before_the_document_is_indexed(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = vectorstore.ensure_reachable

    def spy(ids: list[str]) -> int:
        with get_sessionmaker()() as other:
            seen["status"] = other.scalar(select(Document.status))
        seen["ids"] = list(ids)
        return real(ids)

    monkeypatch.setattr(vectorstore, "ensure_reachable", spy)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    assert seen["status"] == "processing", (
        "verification happens before the document is marked indexed"
    )
    assert len(seen["ids"]) == 2 and all(i.startswith(str(doc.id)) for i in seen["ids"])
    assert fresh(db, doc).status == "indexed"


def test_an_unreachable_vector_is_repaired_during_ingestion(
    db: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real_find = vectorstore.find_unreachable
    calls = {"n": 0}

    def flaky(ids: Any = None, **kwargs: Any) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 1 and ids:
            return [ids[0]]  # the first check claims one new vector is not searchable
        return real_find(ids, **kwargs)

    monkeypatch.setattr(vectorstore, "find_unreachable", flaky)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    with caplog.at_level(logging.INFO):
        ingestion.ingest_document(doc.id)
    assert fresh(db, doc).status == "indexed"
    done = next(r for r in caplog.records if r.getMessage() == "document indexed")
    assert done.repaired_vectors == 1 and "verify_ms" in done.__dict__  # type: ignore[attr-defined]


def test_a_vector_that_cannot_be_made_searchable_fails_the_document_and_cleans_up(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def impossible(ids: list[str]) -> int:
        raise vectorstore.VectorStoreError("3 vectors could not be made searchable")

    monkeypatch.setattr(vectorstore, "ensure_reachable", impossible)
    doc = upload(db, make_user(db), "a.md", MD.encode())
    ingestion.ingest_document(doc.id)
    doc = fresh(db, doc)
    assert doc.status == "failed" and "search index is unavailable" in (doc.error_message or "")
    assert_nothing_left_behind(db, doc)
