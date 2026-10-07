"""Vector store against a real ChromaDB server: cosine, explicit embeddings, flat metadata, filters."""

from __future__ import annotations

import math
from datetime import date
from types import ModuleType
from typing import Any

import pytest

from app.config import get_settings
from app.core.vectorstore import VectorStoreError, VectorStoreMismatchError

DIM = 768


def unit(*values: float) -> list[float]:
    vector = list(values) + [0.0] * (DIM - len(values))
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def meta(
    vs: ModuleType,
    doc: str = "d1",
    dept: str = "IT",
    level: str = "Internal",
    day: int = 20260101,
    page: int = 1,
) -> dict[str, Any]:
    return vs.chunk_metadata(  # type: ignore[attr-defined, no-any-return]
        document_id=doc,
        document_name=f"{doc}.pdf",
        department=dept,
        document_type="Policy",
        creation_date=date(day // 10000, day // 100 % 100, day % 100),
        creation_date_int=day,
        access_level=level,
        page_number=page,
        page_end=page,
        section_title="Intro",
    )


def put(
    vs: ModuleType,
    ids: list[str],
    vectors: list[list[float]],
    metas: list[dict[str, Any]] | None = None,
) -> None:
    vs.upsert(ids, vectors, [f"text of {i}" for i in ids], metas or [meta(vs) for _ in ids])  # type: ignore[attr-defined]


# --- The collection --------------------------------------------------------------------------------


def test_collection_is_cosine_without_a_default_embedder_and_records_the_model(
    vectors: Any,
) -> None:
    settings = get_settings()
    collection = vectors.get_collection()
    assert (
        collection.name
        == settings.collection_name
        == f"{settings.chroma_collection}__gemini-embedding-001__768"
    )
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.configuration["embedding_function"] is None, (
        "Chroma's built-in embedder must not be attached"
    )
    assert collection.metadata["embedding_model"] == settings.embedding_model
    assert collection.metadata["embedding_dim"] == settings.embedding_dim


def test_ensure_collection_is_idempotent(vectors: Any) -> None:
    put(vectors, ["a"], [unit(1)])
    assert vectors.ensure_collection().count() == 1
    assert vectors.ensure_collection().count() == 1


def test_a_collection_built_with_another_model_is_refused(vectors: Any) -> None:
    vectors.drop_collection()
    vectors.get_client().create_collection(
        name=get_settings().collection_name,
        metadata={"embedding_model": "models/some-other-model", "embedding_dim": 768},
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    with pytest.raises(VectorStoreMismatchError, match="embedding_model"):
        vectors.ensure_collection()


def test_a_collection_built_with_another_dimension_is_refused(vectors: Any) -> None:
    vectors.drop_collection()
    vectors.get_client().create_collection(
        name=get_settings().collection_name,
        metadata={"embedding_model": get_settings().embedding_model, "embedding_dim": 3072},
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    with pytest.raises(VectorStoreMismatchError, match="embedding_dim"):
        vectors.ensure_collection()


def test_a_collection_using_l2_distance_is_refused(vectors: Any) -> None:
    vectors.drop_collection()
    vectors.get_client().create_collection(
        name=get_settings().collection_name,
        metadata={"embedding_model": get_settings().embedding_model, "embedding_dim": 768},
        embedding_function=None,  # Chroma's own default distance is L2
    )
    with pytest.raises(VectorStoreMismatchError, match="cosine"):
        vectors.ensure_collection()


# --- Search ------------------------------------------------------------------------------------------


def test_similarity_is_one_minus_cosine_distance(vectors: Any) -> None:
    put(vectors, ["same", "diag", "ortho"], [unit(1, 0), unit(1, 1), unit(0, 0, 1)])
    hits = {h.chunk_id: h.similarity for h in vectors.query(unit(1, 0), 3)}
    assert hits["same"] == pytest.approx(1.0, abs=1e-5)
    assert hits["diag"] == pytest.approx(
        math.sqrt(0.5), abs=1e-4
    )  # 45 degrees -> 0.7071 (L2 would not give this)
    assert hits["ortho"] == pytest.approx(0.0, abs=1e-5)


def test_results_come_best_first_with_text_and_metadata(vectors: Any) -> None:
    put(
        vectors,
        ["far", "near"],
        [unit(0, 1), unit(1, 0.1)],
        [meta(vectors, page=9), meta(vectors, page=3)],
    )
    hits = vectors.query(unit(1, 0), 5)
    assert [h.chunk_id for h in hits] == ["near", "far"]
    assert hits[0].text == "text of near" and hits[0].metadata["page_number"] == 3
    assert hits[0].similarity > hits[1].similarity


def test_n_results_is_capped_by_the_collection_size_and_empty_collections_return_nothing(
    vectors: Any,
) -> None:
    assert vectors.query(unit(1), 5) == []
    put(vectors, ["only"], [unit(1)])
    assert len(vectors.query(unit(1), 30)) == 1


def test_where_filters_are_applied_before_ranking(vectors: Any) -> None:
    put(
        vectors,
        ["it-int", "it-res", "hr-int", "hr-old"],
        [unit(1, 0.0), unit(1, 0.01), unit(1, 0.02), unit(1, 0.03)],
        [
            meta(vectors, "a", "IT", "Internal", 20260101),
            meta(vectors, "b", "IT", "Restricted", 20260101),
            meta(vectors, "c", "HR", "Internal", 20260101),
            meta(vectors, "d", "HR", "Internal", 20200101),
        ],
    )
    query = unit(1, 0)
    ids = lambda where: {h.chunk_id for h in vectors.query(query, 10, where)}  # noqa: E731
    assert ids({"department": {"$eq": "HR"}}) == {"hr-int", "hr-old"}
    assert ids({"access_level": {"$in": ["Public", "Internal"]}}) == {"it-int", "hr-int", "hr-old"}
    assert ids({"creation_date_int": {"$gte": 20250101}}) == {"it-int", "it-res", "hr-int"}
    assert ids(
        {
            "$and": [
                {"department": {"$eq": "HR"}},
                {"creation_date_int": {"$gte": 20250101}},
                {"access_level": {"$in": ["Internal"]}},
            ]
        }
    ) == {"hr-int"}
    assert ids({"document_id": {"$eq": "nope"}}) == set()


# --- Writes ------------------------------------------------------------------------------------------


def test_upsert_replaces_an_existing_chunk(vectors: Any) -> None:
    put(vectors, ["x"], [unit(1)])
    vectors.upsert(["x"], [unit(0, 1)], ["new text"], [meta(vectors)])
    assert vectors.count() == 1
    assert vectors.query(unit(0, 1), 1)[0].text == "new text"


def test_delete_document_removes_only_that_documents_vectors(vectors: Any) -> None:
    put(
        vectors,
        ["a1", "a2", "b1"],
        [unit(1), unit(0, 1), unit(0, 0, 1)],
        [meta(vectors, "A"), meta(vectors, "A"), meta(vectors, "B")],
    )
    vectors.delete_document("A")
    assert (
        vectors.count() == 1
        and vectors.ids_for_document("B") == ["b1"]
        and vectors.ids_for_document("A") == []
    )
    vectors.delete_document("never-existed")  # deleting nothing is fine
    assert vectors.count() == 1


def test_large_writes_are_split_below_the_servers_batch_limit(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vectors.get_client(), "get_max_batch_size", lambda: 2)
    collection = vectors.get_collection()
    real_upsert, sizes = collection.upsert, []

    def spy(**kwargs: Any) -> None:
        sizes.append(len(kwargs["ids"]))
        real_upsert(**kwargs)

    monkeypatch.setattr(collection, "upsert", spy)
    put(vectors, [f"c{i}" for i in range(5)], [unit(1, i) for i in range(5)])
    assert sizes == [2, 2, 1] and vectors.count() == 5


def test_all_ids_pages_through_the_collection(vectors: Any) -> None:
    put(vectors, [f"c{i}" for i in range(5)], [unit(1, i) for i in range(5)])
    assert vectors.all_ids(page_size=2) == {f"c{i}" for i in range(5)}


# --- Validation ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [{"a": None}, {"a": {"nested": 1}}, {"a": [1, 2]}, {"a": (1,)}, {1: "int key"}],
)
def test_metadata_must_be_flat_primitives(vectors: Any, bad: dict[Any, Any]) -> None:
    with pytest.raises(ValueError, match="flat"):
        vectors.upsert(["x"], [unit(1)], ["t"], [bad])
    assert vectors.count() == 0


def test_flat_primitives_including_bool_are_accepted(vectors: Any) -> None:
    vectors.upsert(
        ["x"], [unit(1)], ["t"], [{"s": "text", "i": 3, "f": 1.5, "b": True, "empty": ""}]
    )
    assert vectors.count() == 1


def test_wrong_dimension_and_mismatched_lengths_are_rejected(vectors: Any) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        vectors.upsert(["x"], [[1.0, 0.0]], ["t"], [meta(vectors)])
    with pytest.raises(ValueError, match="same length"):
        vectors.upsert(["x", "y"], [unit(1)], ["t"], [meta(vectors)])
    assert vectors.count() == 0


def test_chunk_metadata_has_exactly_the_documented_flat_keys(vectors: Any) -> None:
    m = meta(vectors)
    assert set(m) == {
        "document_id", "document_name", "department", "document_type", "creation_date",
        "creation_date_int", "access_level", "page_number", "page_end", "section_title",
    }  # fmt: skip
    assert m["creation_date"] == "2026-01-01" and m["creation_date_int"] == 20260101
    assert all(isinstance(v, (str, int)) for v in m.values())


# --- Errors ---------------------------------------------------------------------------------------------


def test_chroma_failures_are_reported_by_class_name_only(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(vectors, ["x"], [unit(1)])
    collection = vectors.get_collection()

    def boom(**_: Any) -> None:
        raise RuntimeError("leaked confidential question text")

    monkeypatch.setattr(collection, "query", boom)
    with pytest.raises(VectorStoreError) as caught:
        vectors.query(unit(1), 3)
    assert "confidential" not in str(caught.value) and "RuntimeError" in str(caught.value)
