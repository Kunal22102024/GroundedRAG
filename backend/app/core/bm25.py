"""In-memory BM25 keyword index (``bm25s``).

Why it exists: dense search can miss exact tokens such as error codes and IDs, so a
keyword channel runs alongside it and the two are fused by RRF (TDD 4.2, 5.3).

* **Identifier-aware tokenizer.** ``ERR_404_NET`` is indexed as the whole token plus
  ``err``, ``404`` and ``net``; queries are tokenized identically.
* **Filter parity.** Each indexed chunk carries the same metadata Chroma filters on
  (document, department, type, date, access level). ``search`` takes a predicate over
  that metadata, so both channels search an identical subset (CLAUDE.md rule 6).
* **Derived data.** PostgreSQL is the source of truth; this index is rebuilt from the
  ``chunks`` table at startup and updated after each ingest and delete.
* **Concurrency.** The index state is immutable and replaced atomically. Searches read
  the current state without locking, so a long ingestion never blocks a question, and
  writers (``add_document``, ``remove_document``, ``rebuild``) are serialized.

``bm25s`` cannot add documents incrementally, so each update rebuilds the index from
the stored token lists (cheap next to embedding, and the tokens are interned so the
memory cost is a pointer per token). This single-process design is why the backend
runs one worker (rule 4).
"""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Letters/digits (any script), optionally joined by identifier separators.
_IDENTIFIER = re.compile(r"[^\W_]+(?:[_\-./:#@+][^\W_]+)*")
_SEPARATORS = re.compile(r"[_\-./:#@+]")
# bm25s cannot index a document with no tokens; this placeholder never matches a query.
_EMPTY_DOCUMENT = ("<empty>",)


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens: identifiers whole, plus their parts; single letters are dropped."""
    tokens: list[str] = []
    for match in _IDENTIFIER.finditer(text):
        whole = match.group(0).casefold()
        parts = [p for p in _SEPARATORS.split(whole) if p]
        if len(parts) > 1:
            tokens.append(whole)
            tokens.extend(parts)
        else:
            tokens.append(whole)
    return [sys.intern(t) for t in tokens if len(t) > 1 or t.isdigit()]


@dataclass(frozen=True)
class ChunkMeta:
    """The per-chunk fields the keyword channel needs to filter on."""

    chunk_id: str
    document_id: str
    department: str
    document_type: str
    creation_date_int: int
    access_level: str


@dataclass(frozen=True)
class KeywordHit:
    """One keyword search result."""

    chunk_id: str
    score: float


class _State:
    """An immutable snapshot of the index."""

    __slots__ = ("metas", "retriever", "tokens")

    def __init__(self, metas: tuple[ChunkMeta, ...], tokens: tuple[tuple[str, ...], ...]) -> None:
        self.metas = metas
        self.tokens = tokens
        self.retriever: Any | None = None
        if metas:
            import bm25s

            retriever = bm25s.BM25()
            retriever.index([list(t) or list(_EMPTY_DOCUMENT) for t in tokens], show_progress=False)
            self.retriever = retriever


class Bm25Index:
    """Thread-safe BM25 index over chunk text."""

    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._state = _State((), ())

    def __len__(self) -> int:
        return len(self._state.metas)

    def chunk_ids(self) -> set[str]:
        """Ids of every indexed chunk."""
        return {m.chunk_id for m in self._state.metas}

    def rebuild(self, chunks: Iterable[tuple[ChunkMeta, str]]) -> None:
        """Replace the whole index (startup, or recovery)."""
        metas: list[ChunkMeta] = []
        tokens: list[tuple[str, ...]] = []
        for meta, text in chunks:
            metas.append(meta)
            tokens.append(tuple(tokenize(text)))
        state = _State(tuple(metas), tuple(tokens))
        with self._write_lock:
            self._state = state

    def add_document(self, chunks: Iterable[tuple[ChunkMeta, str]]) -> None:
        """Add (or replace) one document's chunks."""
        new = [(meta, tuple(tokenize(text))) for meta, text in chunks]
        if not new:
            return
        replaced = {meta.document_id for meta, _ in new}
        with self._write_lock:
            old = self._state
            keep = [i for i, m in enumerate(old.metas) if m.document_id not in replaced]
            metas = tuple(old.metas[i] for i in keep) + tuple(m for m, _ in new)
            tokens = tuple(old.tokens[i] for i in keep) + tuple(t for _, t in new)
            self._state = _State(metas, tokens)

    def remove_document(self, document_id: str) -> int:
        """Drop a document's chunks. Returns how many were removed."""
        with self._write_lock:
            old = self._state
            keep = [i for i, m in enumerate(old.metas) if m.document_id != document_id]
            removed = len(old.metas) - len(keep)
            if removed:
                self._state = _State(
                    tuple(old.metas[i] for i in keep), tuple(old.tokens[i] for i in keep)
                )
            return removed

    def search(
        self,
        query: str,
        n: int,
        allow: Callable[[ChunkMeta], bool] | None = None,
    ) -> list[KeywordHit]:
        """Top ``n`` chunks by BM25 score (best first), among those ``allow`` accepts.

        Chunks that share no token with the query are never returned.
        """
        state = self._state  # one atomic read: the snapshot cannot change under us
        tokens = list(dict.fromkeys(tokenize(query)))
        if state.retriever is None or not tokens or n <= 0:
            return []
        scores = np.asarray(state.retriever.get_scores(tokens), dtype=np.float64)
        if allow is not None:
            mask = np.fromiter((allow(m) for m in state.metas), dtype=bool, count=len(state.metas))
            scores = np.where(mask, scores, 0.0)
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:n]
        return [KeywordHit(state.metas[i].chunk_id, float(scores[i])) for i in order]


_index = Bm25Index()


def get_index() -> Bm25Index:
    """The process-wide keyword index."""
    return _index


def reset_index() -> None:
    """Empty the process-wide index (tests)."""
    _index.rebuild([])
