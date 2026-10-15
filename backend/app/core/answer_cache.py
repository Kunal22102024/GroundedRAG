"""Semantic answer cache: a finished, verified answer is reused for a near-identical question.

An entry is the whole outcome of one clean chat turn (answer, citations, conflicts, follow-ups),
keyed by the question's vector and a *scope*: the access levels, the filters and the document
selection it was asked under, plus the model, embedding and prompt versions. Two readers with
different access levels, or the same reader with different filters, never share an entry
(CLAUDE.md rule 6). A hit needs a cosine similarity of at least ``SEMANTIC_CACHE_THRESHOLD``
between the standalone question vectors (they are L2-normalised, so it is a dot product).

The cache is derived data and lives in this process only: the backend runs one worker (rule 4)
and a restart starts empty, so a changed model or setting cannot outlive its deployment.
Entries leave it by

* ``invalidate_document``: a document in scope was indexed, or deleted (its text is quoted in the
  entry's citations, rule 24), or its graph was rebuilt,
* ``evict_message``: someone rated the answer down,
* age (``SEMANTIC_CACHE_TTL_SECONDS``) and size (least recently used first).

``generation`` closes a race: a turn that started before an invalidation must not store its
answer after it, because it may have been written from the documents as they were.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import get_settings
from app.core import prompts
from app.core.bm25 import ChunkMeta
from app.core.filters import SearchFilter


@dataclass
class CacheEntry:
    """One cached turn. The fields a reader is shown are copies of what the turn produced."""

    scope: str
    vector: np.ndarray
    search_filter: SearchFilter
    answer: str
    citations: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    guard: dict[str, Any]
    cited_chunk_ids: tuple[str, ...]
    cited_document_ids: frozenset[str]
    saved_tokens: int  # what answering it cost the first time, for the dashboard
    created_at: float = field(default_factory=time.monotonic)
    message_ids: set[str] = field(default_factory=set)
    hits: int = 0


@dataclass(frozen=True)
class CacheHit:
    """A lookup result: the entry, its id (to evict it) and how close the question was."""

    entry_id: int
    entry: CacheEntry
    similarity: float


def scope_key(search_filter: SearchFilter) -> str:
    """A stable hash of everything that makes two questions comparable.

    Order never matters (a document selection in another order is the same selection), and a
    change to the model, the embedding collection, the relevance threshold, the number of
    passages or the answer prompt gives every entry a new scope.
    """
    settings = get_settings()
    description = {
        "levels": sorted(search_filter.access_levels),
        "department": search_filter.department,
        "document_type": search_filter.document_type,
        "date_from": search_filter.date_from,
        "date_to": search_filter.date_to,
        "documents": sorted(search_filter.document_ids)
        if search_filter.document_ids is not None
        else None,
        "llm": settings.llm_model,
        "collection": settings.collection_name,
        "threshold": settings.relevance_threshold,
        "top_k": settings.retrieval_top_k,
        # Answers written with and without the graph's extra passages are different answers.
        "graph_hops": settings.graph_hops if settings.graph_retrieval_active else 0,
        "prompt": prompts.PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


class AnswerCache:
    """A bounded, thread-safe store of entries with per-scope nearest-vector lookup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self._next_id = 0
        self._generation = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def generation(self) -> int:
        """Changes whenever entries were invalidated; see the module docstring."""
        with self._lock:
            return self._generation

    def lookup(self, scope: str, vector: list[float]) -> CacheHit | None:
        """The closest live entry of this scope at or above the threshold, or ``None``."""
        settings = get_settings()
        query = np.asarray(vector, dtype=np.float32)
        oldest = time.monotonic() - settings.semantic_cache_ttl_seconds
        best: tuple[int, float] | None = None
        with self._lock:
            for entry_id in [i for i, e in self._entries.items() if e.created_at < oldest]:
                del self._entries[entry_id]
            for entry_id, entry in self._entries.items():
                if entry.scope != scope or entry.vector.shape != query.shape:
                    continue
                similarity = float(np.dot(entry.vector, query))
                if similarity >= settings.semantic_cache_threshold and (
                    best is None or similarity > best[1]
                ):
                    best = (entry_id, similarity)
            if best is None:
                return None
            entry = self._entries[best[0]]
            entry.hits += 1
            self._entries.move_to_end(best[0])
            return CacheHit(best[0], entry, best[1])

    def store(self, entry: CacheEntry, generation: int) -> int | None:
        """Add an entry unless something was invalidated since ``generation`` was read.

        Returns the new entry's id, or ``None`` when it was not stored.
        """
        with self._lock:
            if generation != self._generation:
                return None
            entry_id = self._next_id
            self._next_id += 1
            self._entries[entry_id] = entry
            while len(self._entries) > get_settings().semantic_cache_max_entries:
                self._entries.popitem(last=False)
            return entry_id

    def register_message(self, entry_id: int, message_id: str) -> None:
        """Remember that a stored message carries this entry's answer (for ``evict_message``)."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                entry.message_ids.add(message_id)

    def evict(self, entry_id: int) -> None:
        """Drop one entry (a hit that failed re-validation)."""
        with self._lock:
            self._entries.pop(entry_id, None)
            self._generation += 1

    def evict_message(self, message_id: str) -> int:
        """Drop the entry a message was produced from or served from. Returns how many."""
        with self._lock:
            gone = [i for i, e in self._entries.items() if message_id in e.message_ids]
            for entry_id in gone:
                del self._entries[entry_id]
            self._generation += 1
            return len(gone)

    def invalidate_document(self, document_id: str, meta: ChunkMeta | None) -> int:
        """Drop every entry a change to this document can make wrong. Returns how many.

        An entry is affected when it cites the document (its text is quoted in it), or when the
        document falls inside the entry's filter, so it would now take part in the search.
        Without ``meta`` (the document's metadata could not be read) everything is dropped.
        """
        with self._lock:
            if meta is None:
                gone = list(self._entries)
            else:
                gone = [
                    i
                    for i, e in self._entries.items()
                    if document_id in e.cited_document_ids or e.search_filter.allows(meta)
                ]
            for entry_id in gone:
                del self._entries[entry_id]
            self._generation += 1
            return len(gone)

    def clear(self) -> None:
        """Drop everything (tests, and a changed configuration)."""
        with self._lock:
            self._entries.clear()
            self._generation += 1


_cache = AnswerCache()


def get_cache() -> AnswerCache:
    """The process-wide cache."""
    return _cache
