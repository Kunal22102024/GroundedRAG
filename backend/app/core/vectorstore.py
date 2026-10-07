"""ChromaDB access.

The only module in the backend that imports ``chromadb`` (CLAUDE.md rule 2).
Collection management, upsert, dense query and delete-by-document are added
alongside the ingestion and retrieval work; this module currently provides the
client and the connectivity probe used by the readiness check.

Server/client version pinning (rule 28) cannot be checked at runtime: the
server's ``/api/v2/version`` endpoint returns a fixed API version ("1.0.0"),
not the build version. It is enforced statically instead by
``tests/test_pinned_versions.py``.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


class VectorStoreError(Exception):
    """Raised when ChromaDB cannot be reached or misbehaves."""


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
