"""Search filters: the one place a retrieval filter is built (CLAUDE.md rule 2).

``build_filter`` turns the caller's role and the optional request filters into one neutral
``SearchFilter``. That single object is translated three ways, so every channel searches an
identical subset (rule 6, TDD 5.2):

* ``to_chroma()``      the ``where`` clause for the dense channel,
* ``allows()``         the predicate for the BM25 channel,
* ``sql_conditions()`` conditions on the ``chunks`` table, applied again when candidates are
                       loaded from PostgreSQL.

The access-level constraint comes from the role and is part of every translation. There is no way
to build a filter without it, so no code path can skip it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from app.constants import allowed_access_levels
from app.core import metadata
from app.core.bm25 import ChunkMeta


class HasRole(Protocol):
    """Anything with a role (the ``User`` model)."""

    role: str


class FilterRequest(Protocol):
    """The optional narrowing filters a chat request may carry (``schemas.ChatFilters``)."""

    department: str | None
    document_type: str | None
    date_from: date | None
    date_to: date | None
    document_ids: Sequence[Any] | None


@dataclass(frozen=True)
class SearchFilter:
    """One logical filter: access levels (always) plus optional metadata constraints."""

    access_levels: tuple[str, ...]
    department: str | None = None
    document_type: str | None = None
    date_from: int | None = None  # YYYYMMDD, inclusive
    date_to: int | None = None  # YYYYMMDD, inclusive
    document_ids: tuple[str, ...] | None = None

    @property
    def matches_nothing(self) -> bool:
        """True when no chunk can satisfy the filter (no readable level, or an inverted date range)."""
        if not self.access_levels:
            return True
        return (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        )

    def to_chroma(self) -> dict[str, Any]:
        """The Chroma ``where`` clause. Never call it for a filter that matches nothing."""
        if self.matches_nothing:
            raise ValueError("this filter matches nothing; there is nothing to query")
        clauses: list[dict[str, Any]] = [{"access_level": {"$in": list(self.access_levels)}}]
        if self.department is not None:
            clauses.append({"department": {"$eq": self.department}})
        if self.document_type is not None:
            clauses.append({"document_type": {"$eq": self.document_type}})
        if self.date_from is not None:
            clauses.append({"creation_date_int": {"$gte": self.date_from}})
        if self.date_to is not None:
            clauses.append({"creation_date_int": {"$lte": self.date_to}})
        if self.document_ids is not None:
            clauses.append({"document_id": {"$in": list(self.document_ids)}})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def allows(self, meta: ChunkMeta) -> bool:
        """The same filter as a predicate over the metadata the BM25 index keeps per chunk."""
        if meta.access_level not in self.access_levels:
            return False
        if self.department is not None and meta.department != self.department:
            return False
        if self.document_type is not None and meta.document_type != self.document_type:
            return False
        if self.date_from is not None and meta.creation_date_int < self.date_from:
            return False
        if self.date_to is not None and meta.creation_date_int > self.date_to:
            return False
        return self.document_ids is None or meta.document_id in self.document_ids

    def sql_conditions(self, chunk: Any) -> list[Any]:
        """The same filter as SQLAlchemy conditions on the ``Chunk`` columns (pass the model class)."""
        conditions = [chunk.access_level.in_(self.access_levels)]
        if self.department is not None:
            conditions.append(chunk.department == self.department)
        if self.document_type is not None:
            conditions.append(chunk.document_type == self.document_type)
        if self.date_from is not None:
            conditions.append(chunk.creation_date_int >= self.date_from)
        if self.date_to is not None:
            conditions.append(chunk.creation_date_int <= self.date_to)
        if self.document_ids is not None:
            conditions.append(chunk.document_id.in_([uuid.UUID(d) for d in self.document_ids]))
        return conditions


def build_filter(user: HasRole, request: FilterRequest | None = None) -> SearchFilter:
    """Combine the user's access levels with the request's optional filters.

    Unknown roles get no access levels (fail closed). An empty document list means "no
    restriction", which is what a UI with nothing selected sends.
    """
    levels = tuple(level.value for level in allowed_access_levels(user.role))
    if request is None:
        return SearchFilter(access_levels=levels)
    department = (request.department or "").strip()
    document_type = (request.document_type or "").strip()
    return SearchFilter(
        access_levels=levels,
        department=metadata.canonical_department(department) if department else None,
        document_type=metadata.canonical_document_type(document_type) if document_type else None,
        date_from=metadata.date_to_int(request.date_from) if request.date_from else None,
        date_to=metadata.date_to_int(request.date_to) if request.date_to else None,
        document_ids=tuple(str(d) for d in request.document_ids or ()) or None,
    )
