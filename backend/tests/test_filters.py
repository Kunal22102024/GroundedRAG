"""The one filter builder: the access constraint is in every translation (CLAUDE.md rule 6)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

import pytest

from app.core.bm25 import ChunkMeta
from app.core.filters import SearchFilter, build_filter


@dataclass
class FakeUser:
    role: str


@dataclass
class Request:
    department: str | None = None
    document_type: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    document_ids: list[uuid.UUID] | None = None


def meta(**overrides: object) -> ChunkMeta:
    fields: dict[str, object] = {
        "chunk_id": "c1",
        "document_id": "d1",
        "department": "IT",
        "document_type": "Policy",
        "creation_date_int": 20250601,
        "access_level": "Internal",
    }
    fields.update(overrides)
    return ChunkMeta(**fields)  # type: ignore[arg-type]


def test_admin_and_user_levels() -> None:
    assert build_filter(FakeUser("admin")).access_levels == ("Public", "Internal", "Restricted")
    assert build_filter(FakeUser("user")).access_levels == ("Public", "Internal")


def test_unknown_role_fails_closed() -> None:
    nobody = build_filter(FakeUser("intruder"))
    assert nobody.access_levels == () and nobody.matches_nothing
    with pytest.raises(ValueError):
        nobody.to_chroma()
    assert not nobody.allows(meta(access_level="Public"))


@pytest.mark.parametrize(
    "request_filters",
    [
        None,
        Request(),
        Request(department="IT"),
        Request(document_type="Policy", date_from=date(2025, 1, 1), date_to=date(2025, 12, 31)),
        Request(document_ids=[uuid.uuid4()]),
    ],
)
def test_every_chroma_filter_carries_the_access_constraint(request_filters: Request | None) -> None:
    where = build_filter(FakeUser("user"), request_filters).to_chroma()
    clauses = where["$and"] if "$and" in where else [where]
    assert {"access_level": {"$in": ["Public", "Internal"]}} in clauses


def test_chroma_clauses_for_each_filter() -> None:
    doc = uuid.uuid4()
    where = build_filter(
        FakeUser("admin"),
        Request(
            department="it",
            document_type="policy",
            date_from=date(2025, 1, 1),
            date_to=date(2025, 6, 30),
            document_ids=[doc],
        ),
    ).to_chroma()
    assert where["$and"] == [
        {"access_level": {"$in": ["Public", "Internal", "Restricted"]}},
        {"department": {"$eq": "IT"}},  # matched case-insensitively to the canonical spelling
        {"document_type": {"$eq": "Policy"}},
        {"creation_date_int": {"$gte": 20250101}},
        {"creation_date_int": {"$lte": 20250630}},
        {"document_id": {"$in": [str(doc)]}},
    ]


def test_keyword_predicate_matches_the_same_subset() -> None:
    f = build_filter(FakeUser("user"), Request(department="IT", date_from=date(2025, 1, 1)))
    assert f.allows(meta())
    assert not f.allows(meta(access_level="Restricted"))  # the access constraint
    assert not f.allows(meta(department="HR"))
    assert not f.allows(meta(creation_date_int=20241231))


def test_empty_document_list_means_no_restriction() -> None:
    assert build_filter(FakeUser("user"), Request(document_ids=[])).document_ids is None


def test_inverted_date_range_matches_nothing() -> None:
    f = SearchFilter(access_levels=("Public",), date_from=20250201, date_to=20250101)
    assert f.matches_nothing
