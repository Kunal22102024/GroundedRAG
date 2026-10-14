"""Maintenance commands.

    python -m app.maintenance reembed
    python -m app.maintenance verify-index

``reembed`` rebuilds the Chroma vectors from the chunks stored in PostgreSQL (the source
of truth, TDD 3.4). Use it after changing ``EMBEDDING_MODEL`` or ``EMBEDDING_DIM`` (the
collection name changes with them, so the new collection starts empty), or after losing
the Chroma volume. The backend refuses to start in those situations, so run it with the
entrypoint overridden:

    docker compose run --rm --no-deps --entrypoint python backend -m app.maintenance reembed

It is resumable: chunks already present in the collection are skipped, so an interrupted
run (for example by an API quota) continues where it stopped. It does not need the
backend to be running and prints counts only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from sqlalchemy import select

from app.config import get_settings
from app.constants import DocumentStatus
from app.core import embeddings, vectorstore
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document
from app.middleware import configure_logging

logger = logging.getLogger("app.maintenance")

_BATCH = 128  # chunks embedded and written per round (four embedding API batches)


def reembed() -> tuple[int, int]:
    """Embed every indexed chunk that is missing from the collection.

    Returns ``(embedded, skipped)``.
    """
    vectorstore.ensure_collection()
    existing = vectorstore.all_ids()
    logger.info("reembed starting", extra={"already_present": len(existing)})

    statement = (
        select(Chunk, Document.name, Document.creation_date)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == DocumentStatus.INDEXED.value)
        .order_by(Document.id, Chunk.ordinal)
        .execution_options(yield_per=500)
    )
    embedded = skipped = 0
    batch: list[tuple[Chunk, str, object]] = []

    def flush() -> None:
        nonlocal embedded
        if not batch:
            return
        vectors = embeddings.embed_documents([chunk.content for chunk, _, _ in batch])
        # Embedding takes a while, and the chunk rows above come from a cursor opened earlier: a
        # document deleted in the meantime must not get its vectors written back.
        with get_sessionmaker()() as check:
            alive = set(
                check.scalars(
                    select(Chunk.id)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(
                        Chunk.id.in_([chunk.id for chunk, _, _ in batch]),
                        Document.status == DocumentStatus.INDEXED.value,
                    )
                )
            )
        kept = [
            (row, vector) for row, vector in zip(batch, vectors, strict=True) if row[0].id in alive
        ]
        if kept:
            vectorstore.upsert(
                ids=[chunk.id for (chunk, _, _), _ in kept],
                embeddings=[vector for _, vector in kept],
                documents=[chunk.content for (chunk, _, _), _ in kept],
                metadatas=[
                    vectorstore.chunk_metadata(
                        document_id=str(chunk.document_id),
                        document_name=name,
                        department=chunk.department,
                        document_type=chunk.document_type,
                        creation_date=created,  # type: ignore[arg-type]
                        creation_date_int=chunk.creation_date_int,
                        access_level=chunk.access_level,
                        page_number=chunk.page_number,
                        page_end=chunk.page_end,
                        section_title=chunk.section_title,
                    )
                    for (chunk, name, created), _ in kept
                ],
            )
            # Chroma can store a vector its index never returns; prove these are searchable (rule 14).
            vectorstore.ensure_reachable([chunk.id for (chunk, _, _), _ in kept])
        embedded += len(kept)
        logger.info(
            "reembed progress",
            extra={"embedded": embedded, "skipped": skipped, "dropped": len(batch) - len(kept)},
        )
        batch.clear()

    with get_sessionmaker()() as db:
        for chunk, name, created in db.execute(statement):
            if chunk.id in existing:
                skipped += 1
                continue
            batch.append((chunk, name, created))
            if len(batch) >= _BATCH:
                flush()
        flush()
    return embedded, skipped


def verify_index() -> tuple[int, int]:
    """Check that every stored vector is reachable by search, and repair any that are not.

    Chroma can hold a vector its index never returns; searches then silently miss it (see
    ``core/vectorstore.py``). No embedding calls are needed: vectors are re-inserted from what
    Chroma already stores. Returns ``(vectors checked, vectors repaired)``.
    """
    vectorstore.get_collection()
    total = vectorstore.count()
    broken = vectorstore.find_unreachable()
    logger.info("verify-index checked", extra={"vectors": total, "unreachable": len(broken)})
    return total, vectorstore.repair_unreachable(broken)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m app.maintenance", description=__doc__.split("\n")[0]
    )
    parser.add_argument("command", choices=["reembed", "verify-index"])
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_level)
    started = time.perf_counter()
    try:
        if args.command == "reembed":
            embedded, skipped = reembed()
            print(
                f"reembed finished: {embedded} chunks embedded, {skipped} already present, "
                f"{time.perf_counter() - started:.1f}s"
            )
        elif args.command == "verify-index":
            checked, repaired = verify_index()
            print(
                f"verify-index finished: {checked} vectors checked, {repaired} repaired, "
                f"{time.perf_counter() - started:.1f}s"
            )
    except (embeddings.EmbeddingError, vectorstore.VectorStoreError) as exc:
        print(f"reembed stopped: {exc}\nRun the command again to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
