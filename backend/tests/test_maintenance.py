"""`python -m app.maintenance reembed`: rebuild vectors from PostgreSQL, resumably."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app import maintenance
from app.config import get_settings
from app.core import embeddings, vectorstore
from app.db.models import Chunk, Document
from app.services import index_service
from tests.conftest import REAL_RUN_STARTUP, FakeEmbeddings


@pytest.fixture(autouse=True)
def stack(db: Session, vectors: Any, fake_embeddings: FakeEmbeddings) -> None:
    """Real Postgres and Chroma with a fake embedder."""


def seed(db: Session, status: str = "indexed", chunks: int = 3, name: str = "doc.pdf") -> Document:
    """Chunks that exist in PostgreSQL only (as after a lost Chroma volume)."""
    document = Document(
        name=name, stored_path=f"{uuid.uuid4().hex}.pdf", mime_type="application/pdf", size_bytes=1,
        status=status, chunk_count=chunks, department="IT", document_type="Policy",
        creation_date=date(2026, 1, 15), access_level="Restricted",
    )  # fmt: skip
    db.add(document)
    db.flush()
    for n in range(chunks):
        db.add(
            Chunk(
                id=f"{document.id}_chunk_{n:04d}", document_id=document.id, ordinal=n,
                content=f"Chunk {n} about retention periods and archive rules.", page_number=n + 1,
                page_end=n + 1, section_title=f"Section {n}", department="IT", document_type="Policy",
                creation_date_int=20260115, access_level="Restricted",
            )
        )  # fmt: skip
    db.commit()
    return document


def test_reembed_restores_vectors_with_full_metadata(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    document = seed(db)
    assert vectorstore.count() == 0
    embedded, skipped = maintenance.reembed()
    assert (embedded, skipped) == (3, 0) and vectorstore.count() == 3

    stored = vectorstore.get_collection().get(
        ids=[f"{document.id}_chunk_0001"], include=["metadatas", "documents"]
    )
    metadata = stored["metadatas"][0]
    assert metadata == {
        "document_id": str(document.id), "document_name": "doc.pdf", "department": "IT",
        "document_type": "Policy", "creation_date": "2026-01-15", "creation_date_int": 20260115,
        "access_level": "Restricted", "page_number": 2, "page_end": 2, "section_title": "Section 1",
    }  # fmt: skip
    assert stored["documents"][0] == "Chunk 1 about retention periods and archive rules."


def test_after_reembed_the_backend_starts_again(db: Session) -> None:
    seed(db)
    with pytest.raises(index_service.IndexInconsistentError):
        REAL_RUN_STARTUP()
    maintenance.reembed()
    assert REAL_RUN_STARTUP().vectors == 3


def test_reembed_is_resumable_and_only_embeds_what_is_missing(
    db: Session, fake_embeddings: FakeEmbeddings
) -> None:
    seed(db, chunks=4)
    maintenance.reembed()
    calls = len(fake_embeddings.document_batches)
    assert maintenance.reembed() == (0, 4), "everything is already present: nothing to do"
    assert len(fake_embeddings.document_batches) == calls

    first = vectorstore.all_ids()
    vectorstore.delete_document(next(iter({i.split("_chunk_")[0] for i in first})))
    seed(db, chunks=2, name="second.pdf")
    embedded, skipped = maintenance.reembed()
    assert embedded == 6 and skipped == 0


def test_only_indexed_documents_are_embedded(db: Session) -> None:
    seed(db, status="failed", chunks=2)
    seed(db, status="processing", chunks=2)
    seed(db, status="indexed", chunks=1)
    assert maintenance.reembed() == (1, 0)


def test_an_interrupted_run_continues_where_it_stopped(
    db: Session, monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    monkeypatch.setattr(maintenance, "_BATCH", 2)
    seed(db, chunks=5)
    real = fake_embeddings.embed_documents
    calls = {"n": 0}

    def flaky(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise embeddings.EmbeddingQuotaError("quota used up")
        return real(texts)

    monkeypatch.setattr(embeddings, "embed_documents", flaky)
    with pytest.raises(embeddings.EmbeddingQuotaError):
        maintenance.reembed()
    assert vectorstore.count() == 2, "the first batch was written before the failure"

    monkeypatch.setattr(embeddings, "embed_documents", real)
    assert maintenance.reembed() == (3, 2) and vectorstore.count() == 5


def test_the_command_line_reports_a_stop_clearly(
    db: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seed(db)

    def stop() -> tuple[int, int]:
        raise embeddings.EmbeddingQuotaError("The embedding quota is exhausted.")

    monkeypatch.setattr(maintenance, "reembed", stop)
    assert maintenance.main(["reembed"]) == 1
    assert "Run the command again to resume" in capsys.readouterr().err


def test_the_command_line_prints_a_summary(db: Session, capsys: pytest.CaptureFixture[str]) -> None:
    seed(db)
    assert maintenance.main(["reembed"]) == 0
    assert "3 chunks embedded, 0 already present" in capsys.readouterr().out
    assert get_settings().collection_name in vectorstore.get_collection().name


# --- verify-index ------------------------------------------------------------------------------------------------


def test_verify_index_checks_every_vector_and_repairs_the_unreachable_ones(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_index_health import fill, hide

    ids = fill(vectors, 30)
    hide(monkeypatch, vectors, {ids[4], ids[20]}, heals_on_upsert=True)
    assert maintenance.verify_index() == (30, 2)
    assert maintenance.verify_index() == (30, 0)


def test_verify_index_command_line(vectors: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from tests.test_index_health import fill

    fill(vectors, 12)
    assert maintenance.main(["verify-index"]) == 0
    assert "12 vectors checked, 0 repaired" in capsys.readouterr().out


# --- Rule 14 and deletions during a run ------------------------------------------------------------------------


def test_reembed_proves_every_written_vector_is_searchable(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(db, chunks=4)
    checked: list[str] = []
    real = vectorstore.ensure_reachable

    def spy(ids: Any) -> int:
        checked.extend(ids)
        return real(ids)

    monkeypatch.setattr(vectorstore, "ensure_reachable", spy)
    maintenance.reembed()
    assert sorted(checked) == sorted(vectorstore.all_ids()) and len(checked) == 4


def test_a_document_deleted_while_its_batch_is_embedded_gets_no_vectors_back(
    db: Session, monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    from sqlalchemy import delete

    from app.db.database import get_sessionmaker

    keep = seed(db, chunks=2, name="keep.pdf")
    gone = seed(db, chunks=2, name="gone.pdf")
    real = fake_embeddings.embed_documents

    def delete_meanwhile(texts: list[str]) -> list[list[float]]:
        out = real(texts)
        with get_sessionmaker()() as other:  # a user deletes the document while it is embedded
            other.execute(delete(Document).where(Document.id == gone.id))
            other.commit()
        return out

    monkeypatch.setattr(embeddings, "embed_documents", delete_meanwhile)
    embedded, _ = maintenance.reembed()
    stored = vectorstore.all_ids()
    assert embedded == 2 and len(stored) == 2
    assert all(chunk_id.startswith(str(keep.id)) for chunk_id in stored)
