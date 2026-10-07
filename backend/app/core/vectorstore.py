"""ChromaDB access.

The only module in the backend that imports ``chromadb`` (CLAUDE.md rule 2).

* The collection uses **cosine** distance (Chroma defaults to L2) and is always given
  **explicit embeddings**: the client's ``get_or_create_collection`` would otherwise
  attach Chroma's built-in embedding function, which must never be used (rule 14).
* The collection name encodes the embedding model and dimension, and its metadata
  records them too. ``ensure_collection`` verifies all of it, so vectors from two
  different embedding configurations can never be mixed (rule 13).
* Chroma metadata must be flat: str, int, float or bool, never ``None`` or nested
  (rule 15). ``upsert`` enforces this.

Server/client version pinning (rule 28) cannot be checked at runtime: the server's
``/api/v2/version`` endpoint returns a fixed API version ("1.0.0"), not the build
version. It is enforced statically by ``tests/test_pinned_versions.py`` instead.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, TypeVar

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

T = TypeVar("T")


class VectorStoreError(Exception):
    """Raised when ChromaDB cannot be reached or misbehaves."""


class VectorStoreMismatchError(VectorStoreError):
    """The collection exists but was built with a different embedding configuration."""


@dataclass(frozen=True)
class DenseHit:
    """One dense search result. ``similarity`` is ``1 - cosine distance`` (0..1 for unit vectors)."""

    chunk_id: str
    similarity: float
    text: str
    metadata: dict[str, Any]


@lru_cache
def get_client() -> ClientAPI:
    """Return the process-wide Chroma HTTP client (server mode, internal network)."""
    settings = get_settings()
    try:
        return chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except Exception as exc:  # the client raises a variety of transport errors
        raise VectorStoreError(f"cannot connect to ChromaDB: {type(exc).__name__}") from exc


def heartbeat() -> int:
    """Return the server heartbeat (nanoseconds). Raises ``VectorStoreError`` on failure."""
    try:
        return int(get_client().heartbeat())
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"ChromaDB heartbeat failed: {type(exc).__name__}") from exc


# --- Collection --------------------------------------------------------------------------

_collection_lock = threading.Lock()
_collection: Any | None = None


def _expected_metadata() -> dict[str, str | int]:
    settings = get_settings()
    return {"embedding_model": settings.embedding_model, "embedding_dim": settings.embedding_dim}


def ensure_collection() -> Any:
    """Create the collection if needed and verify it matches the configured embedding setup.

    Raises ``VectorStoreMismatchError`` if an existing collection with this name was
    created with a different model, dimension or distance function.
    """
    name = get_settings().collection_name
    expected = _expected_metadata()
    try:
        collection = get_client().get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=expected,
            embedding_function=None,  # never Chroma's default embedder (rule 14)
        )
        metadata = dict(collection.metadata or {})
        space = ((collection.configuration or {}).get("hnsw") or {}).get("space")
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"cannot open collection: {type(exc).__name__}") from exc

    problems = [
        f"{key} is {metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if space != "cosine":
        problems.append(f"distance is {space!r}, expected 'cosine'")
    if problems:
        raise VectorStoreMismatchError(
            f"Chroma collection '{name}' does not match the configured embedding setup: "
            + "; ".join(problems)
        )
    return collection


def get_collection() -> Any:
    """The verified collection, opened once per process."""
    global _collection
    with _collection_lock:
        if _collection is None:
            _collection = ensure_collection()
        return _collection


def reset_collection_cache() -> None:
    """Forget the cached collection handle (after drop, or in tests)."""
    global _collection
    with _collection_lock:
        _collection = None


def drop_collection(name: str | None = None) -> None:
    """Delete a collection (the configured one by default). Used by tests and maintenance."""
    target = name or get_settings().collection_name
    try:
        get_client().delete_collection(target)
    except Exception as exc:
        raise VectorStoreError(f"cannot delete collection: {type(exc).__name__}") from exc
    reset_collection_cache()


def _call(operation: str, func: Callable[[], T]) -> T:
    """Run a Chroma call, turning any client error into ``VectorStoreError`` (class name only)."""
    try:
        return func()
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"ChromaDB {operation} failed: {type(exc).__name__}") from exc


# --- Writes ---------------------------------------------------------------------------------


def chunk_metadata(
    *,
    document_id: str,
    document_name: str,
    department: str,
    document_type: str,
    creation_date: date,
    creation_date_int: int,
    access_level: str,
    page_number: int,
    page_end: int,
    section_title: str,
) -> dict[str, str | int]:
    """The flat metadata stored with every chunk (TDD 3.2): primitives only, never ``None``.

    These key names are the contract ``core/filters.py`` builds Chroma ``where`` clauses
    against, so they are defined here and nowhere else.
    """
    return {
        "document_id": document_id,
        "document_name": document_name,
        "department": department,
        "document_type": document_type,
        "creation_date": creation_date.isoformat(),
        "creation_date_int": creation_date_int,
        "access_level": access_level,
        "page_number": page_number,
        "page_end": page_end,
        "section_title": section_title,
    }


def validate_flat_metadata(metadata: dict[str, Any]) -> None:
    """Chroma metadata must be flat str/int/float/bool values (rule 15)."""
    for key, value in metadata.items():
        # bool is a subclass of int, so it passes; None, lists and dicts do not.
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise ValueError(f"Chroma metadata must be flat primitives; bad entry {key!r}")


def upsert(
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
) -> None:
    """Insert or replace chunks, in batches below the server's maximum batch size."""
    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError("ids, embeddings, documents and metadatas must have the same length")
    dimension = get_settings().embedding_dim
    for vector in embeddings:
        if len(vector) != dimension:
            raise ValueError(f"embedding has {len(vector)} dimensions, expected {dimension}")
    for metadata in metadatas:
        validate_flat_metadata(metadata)
    if not ids:
        return

    collection = get_collection()
    batch = max(1, int(_call("batch size lookup", lambda: get_client().get_max_batch_size())))
    for start in range(0, len(ids), batch):
        end = start + batch
        _call(
            "upsert",
            lambda s=start, e=end: collection.upsert(
                ids=list(ids[s:e]),
                embeddings=[list(v) for v in embeddings[s:e]],
                documents=list(documents[s:e]),
                metadatas=list(metadatas[s:e]),
            ),
        )


def delete_document(document_id: str) -> None:
    """Remove every vector of one document. Safe to call when it has none."""
    collection = get_collection()
    _call("delete", lambda: collection.delete(where={"document_id": document_id}))


# --- Index health ---------------------------------------------------------------------------------
#
# Chroma can hold a vector (``get`` returns it, with its embedding) that its HNSW index no longer
# reaches, so no search ever returns it. This was observed against ChromaDB 1.5.9: after a batch
# of deletions and re-uploads, 15 of 155 vectors (all of the smallest documents) were stored but
# unreachable: recall@3 against exact search fell to 8/18 and even a query filtered by
# ``document_id`` could not find them, while a fresh copy of the same vectors scored 18/18. Such a
# hole is silent, so the index is verified after every write and at startup, and repaired without
# re-embedding anything (the vectors are still stored).

_VERIFY_BATCH = 100
# Cosine distance of a vector to itself, or to an identical twin, is zero up to float32 noise.
_SELF_DISTANCE = 1e-4


def _stored_id_batches(
    collection: Any, ids: Sequence[str] | None, size: int
) -> Iterator[list[str]]:
    if ids is not None:
        for start in range(0, len(ids), size):
            yield list(ids[start : start + size])
        return
    offset = 0
    while True:
        page = _call("get", lambda o=offset: collection.get(limit=size, offset=o, include=[]))
        if not page["ids"]:
            return
        yield list(page["ids"])
        offset += size


def find_unreachable(
    ids: Sequence[str] | None = None, *, collection: Any | None = None
) -> list[str]:
    """Stored vectors that their own embedding does not retrieve.

    Every stored vector is used as a query (batched): it must come back as the nearest hit, or an
    identical twin at distance ~0 must. ``ids=None`` checks the whole collection.
    """
    target = collection if collection is not None else get_collection()
    broken: list[str] = []
    for batch in _stored_id_batches(target, ids, _VERIFY_BATCH):
        got = _call("get", lambda b=batch: target.get(ids=b, include=["embeddings"]))
        if not len(got["ids"]):
            continue
        found = _call(
            "query",
            lambda g=got: target.query(
                query_embeddings=[list(e) for e in g["embeddings"]],
                n_results=1,
                include=["distances"],
            ),
        )
        for stored_id, hit_ids, distances in zip(
            got["ids"], found["ids"], found["distances"], strict=True
        ):
            if not hit_ids or (hit_ids[0] != stored_id and distances[0] > _SELF_DISTANCE):
                broken.append(stored_id)
    return broken


def rebuild_collection(page_size: int = 500) -> int:
    """Copy every stored vector into a fresh collection and swap it in. Returns the vector count.

    Needs no embedding calls: the vectors, texts and metadata are read back from Chroma. There is a
    brief moment between dropping the old collection and renaming the new one when searches fail.
    """
    name = get_settings().collection_name
    temporary = f"{name}__rebuild"
    client = get_client()
    old = get_collection()
    with contextlib.suppress(Exception):
        client.delete_collection(temporary)
    fresh = _call(
        "create",
        lambda: client.create_collection(
            name=temporary,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=_expected_metadata(),
            embedding_function=None,
        ),
    )
    try:
        copied = offset = 0
        while True:
            page = _call(
                "get",
                lambda o=offset: old.get(
                    limit=page_size, offset=o, include=["embeddings", "documents", "metadatas"]
                ),
            )
            if not len(page["ids"]):
                break
            _call(
                "add",
                lambda p=page: fresh.add(
                    ids=list(p["ids"]),
                    embeddings=[list(e) for e in p["embeddings"]],
                    documents=list(p["documents"]),
                    metadatas=list(p["metadatas"]),
                ),
            )
            copied += len(page["ids"])
            offset += page_size
        if int(fresh.count()) != int(old.count()):
            raise VectorStoreError(f"rebuild copied {fresh.count()} of {old.count()} vectors")
        if find_unreachable(collection=fresh):
            raise VectorStoreError("rebuilt collection is still not fully reachable")
    except BaseException:
        with contextlib.suppress(Exception):
            client.delete_collection(temporary)
        raise
    _call("swap", lambda: client.delete_collection(name))
    _call("swap", lambda: fresh.modify(name=name))
    reset_collection_cache()
    return copied


def repair_unreachable(ids: Sequence[str]) -> int:
    """Make the given stored-but-unreachable vectors searchable again. Returns how many.

    First delete and re-insert just those vectors (cheap). If any are still unreachable, rebuild
    the whole collection. Raises ``VectorStoreError`` if even that does not help.
    """
    if not ids:
        return 0
    collection = get_collection()
    for start in range(0, len(ids), _VERIFY_BATCH):
        batch = list(ids[start : start + _VERIFY_BATCH])
        rows = _call(
            "get",
            lambda b=batch: collection.get(ids=b, include=["embeddings", "documents", "metadatas"]),
        )
        if not len(rows["ids"]):
            continue
        _call("delete", lambda b=list(rows["ids"]): collection.delete(ids=b))
        _call(
            "upsert",
            lambda r=rows: collection.upsert(
                ids=list(r["ids"]),
                embeddings=[list(e) for e in r["embeddings"]],
                documents=list(r["documents"]),
                metadatas=list(r["metadatas"]),
            ),
        )
    if find_unreachable(ids):
        rebuild_collection()
        if find_unreachable(ids):
            raise VectorStoreError(f"{len(ids)} vectors could not be made searchable")
    return len(ids)


def ensure_reachable(ids: Sequence[str]) -> int:
    """Verify freshly written vectors can be found, repairing any that cannot. Returns repairs made."""
    return repair_unreachable(find_unreachable(ids))


# --- Reads ----------------------------------------------------------------------------------


def count() -> int:
    """Number of vectors in the collection."""
    collection = get_collection()
    return int(_call("count", lambda: collection.count()))


def all_ids(page_size: int = 5000) -> set[str]:
    """Every chunk id in the collection (used to resume a re-embedding run)."""
    collection = get_collection()
    ids: set[str] = set()
    offset = 0
    while True:
        page = _call("get", lambda o=offset: collection.get(limit=page_size, offset=o, include=[]))
        ids.update(page["ids"])
        if len(page["ids"]) < page_size:
            return ids
        offset += page_size


def ids_for_document(document_id: str) -> list[str]:
    """Chunk ids stored for one document."""
    collection = get_collection()
    result = _call("get", lambda: collection.get(where={"document_id": document_id}, include=[]))
    return list(result["ids"])


def query(
    embedding: Sequence[float], n_results: int, where: dict[str, Any] | None = None
) -> list[DenseHit]:
    """Nearest chunks to ``embedding`` by cosine similarity, best first.

    ``where`` is a Chroma filter built by ``core/filters.py``; it is applied before the
    nearest-neighbour search. The query vector is passed explicitly.
    """
    collection = get_collection()
    total = int(_call("count", lambda: collection.count()))
    if total == 0:
        return []
    result = _call(
        "query",
        lambda: collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(n_results, total),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        ),
    )
    ids = result["ids"][0]
    return [
        DenseHit(chunk_id, 1.0 - float(distance), text or "", dict(metadata or {}))
        for chunk_id, text, metadata, distance in zip(
            ids, result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]
