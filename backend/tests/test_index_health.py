"""Chroma can store a vector its index never returns. These tests prove it is detected and repaired.

Real Chroma cannot be made to lose vectors on demand, so the symptom is simulated by wrapping the
collection's ``query`` so chosen vectors are never returned (a query with such a vector's own
embedding gets a different, farther hit). Detection, re-insertion and rebuilding all run for real.
"""

from __future__ import annotations

import math
import random
from datetime import date
from typing import Any

import pytest

from app.config import get_settings
from app.core import vectorstore
from app.core.vectorstore import VectorStoreError

DIM = 768


def unit(rng: random.Random) -> list[float]:
    vector = [rng.gauss(0, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def fill(vs: Any, count: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    ids = [f"doc{n // 5}_chunk_{n % 5:04d}" for n in range(count)]
    metas = [
        vs.chunk_metadata(
            document_id=i.split("_chunk_")[0], document_name="x.pdf", department="IT",
            document_type="Policy", creation_date=date(2026, 1, 1), creation_date_int=20260101,
            access_level="Internal", page_number=1, page_end=1, section_title=f"S{n}",
        )
        for n, i in enumerate(ids)
    ]  # fmt: skip
    vs.upsert(ids, [unit(rng) for _ in ids], [f"text of {i}" for i in ids], metas)
    return ids


def hide(monkeypatch: pytest.MonkeyPatch, vs: Any, hidden: set[str], heals_on_upsert: bool) -> None:
    """Make ``hidden`` vectors invisible to search, like Chroma's unreachable HNSW nodes."""
    collection = vs.get_collection()
    stored = collection.get(include=["embeddings"])
    by_id = {i: list(e) for i, e in zip(stored["ids"], stored["embeddings"], strict=True)}
    real_query, real_upsert = collection.query, collection.upsert

    def query(**kwargs: Any) -> Any:
        result = real_query(**kwargs)
        for k, embedding in enumerate(kwargs["query_embeddings"]):
            if any(
                max(abs(a - b) for a, b in zip(embedding, by_id[h], strict=True)) < 1e-9
                for h in hidden
            ):
                result["ids"][k], result["distances"][k] = ["decoy"], [0.5]
        return result

    def upsert(**kwargs: Any) -> Any:
        out = real_upsert(**kwargs)
        if heals_on_upsert:
            hidden.difference_update(kwargs["ids"])
        return out

    monkeypatch.setattr(collection, "query", query)
    monkeypatch.setattr(collection, "upsert", upsert)


# Embeddings come back as float32, so a round trip can differ by ~1e-8: compare with a tolerance (`same`).
def snapshot(vs: Any) -> dict[str, tuple[Any, ...]]:
    got = vs.get_collection().get(include=["embeddings", "documents", "metadatas"])
    return {
        i: ([float(x) for x in e], d, tuple(sorted(m.items())))
        for i, e, d, m in zip(
            got["ids"], got["embeddings"], got["documents"], got["metadatas"], strict=True
        )
    }


def same(a: dict[str, tuple[Any, ...]], b: dict[str, tuple[Any, ...]]) -> bool:
    """Same ids, texts and metadata, and embeddings equal to float32 precision."""
    return a.keys() == b.keys() and all(
        a[i][1:] == b[i][1:] and a[i][0] == pytest.approx(b[i][0], abs=1e-6) for i in a
    )


# --- Detection -------------------------------------------------------------------------------------


def test_a_healthy_index_has_no_unreachable_vectors(vectors: Any) -> None:
    ids = fill(vectors, 50)
    assert vectors.find_unreachable() == []
    assert vectors.find_unreachable(ids[:7]) == []
    assert vectors.ensure_reachable(ids) == 0


def test_an_empty_collection_is_fine(vectors: Any) -> None:
    assert vectors.find_unreachable() == [] and vectors.repair_unreachable([]) == 0


def test_identical_twins_are_not_mistaken_for_missing_vectors(vectors: Any) -> None:
    """Re-uploading the same file stores identical vectors under new ids; both must count as reachable."""
    rng = random.Random(3)
    same = unit(rng)
    metas = [vectors.chunk_metadata(document_id=d, document_name="x", department="", document_type="", creation_date=date(2026, 1, 1), creation_date_int=20260101, access_level="Internal", page_number=1, page_end=1, section_title="") for d in ("a", "b")]  # fmt: skip
    vectors.upsert(["a_chunk_0000", "b_chunk_0000"], [same, list(same)], ["t", "t"], metas)
    assert vectors.find_unreachable() == []


def test_hidden_vectors_are_found_and_nothing_else(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 40)
    hidden = {ids[3], ids[17], ids[18]}
    hide(monkeypatch, vectors, set(hidden), heals_on_upsert=True)
    assert set(vectors.find_unreachable()) == hidden
    assert vectors.find_unreachable(ids[:10]) == [ids[3]], "only the requested ids are checked"


def test_a_query_that_returns_nothing_counts_as_unreachable(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 10)
    collection = vectors.get_collection()
    real = collection.query
    monkeypatch.setattr(
        collection,
        "query",
        lambda **kw: {
            **real(**kw),
            "ids": [[] for _ in kw["query_embeddings"]],
            "distances": [[] for _ in kw["query_embeddings"]],
        },
    )
    assert set(vectors.find_unreachable()) == set(ids)


# --- Repair ----------------------------------------------------------------------------------------------


def test_reinserting_unreachable_vectors_repairs_them_without_changing_any_data(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 40)
    before = snapshot(vectors)
    hide(monkeypatch, vectors, {ids[3], ids[17]}, heals_on_upsert=True)

    assert vectors.repair_unreachable(vectors.find_unreachable()) == 2

    assert vectors.find_unreachable() == []
    assert same(snapshot(vectors), before), "vectors, texts and metadata are exactly as before"
    assert vectors.count() == 40


def test_ensure_reachable_reports_how_many_it_repaired(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 20)
    hide(monkeypatch, vectors, {ids[5]}, heals_on_upsert=True)
    assert vectors.ensure_reachable(ids) == 1
    assert vectors.ensure_reachable(ids) == 0


def test_when_reinserting_is_not_enough_the_whole_collection_is_rebuilt(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 40)
    before = snapshot(vectors)
    old_collection = vectors.get_collection()
    hide(monkeypatch, vectors, {ids[3], ids[17]}, heals_on_upsert=False)

    assert vectors.repair_unreachable(vectors.find_unreachable()) == 2

    rebuilt = vectors.get_collection()
    assert rebuilt is not old_collection, "a new collection object was swapped in"
    assert rebuilt.name == get_settings().collection_name
    assert (
        rebuilt.configuration["hnsw"]["space"] == "cosine"
        and rebuilt.metadata["embedding_dim"] == DIM
    )
    assert vectors.find_unreachable() == [] and same(snapshot(vectors), before)
    assert [
        c.name for c in vectors.get_client().list_collections() if c.name.endswith("__rebuild")
    ] == []


def test_a_vector_that_cannot_be_made_searchable_is_reported(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill(vectors, 20)
    hide(monkeypatch, vectors, {ids[2]}, heals_on_upsert=False)
    monkeypatch.setattr(vectorstore, "rebuild_collection", lambda *a, **k: 0)
    with pytest.raises(VectorStoreError, match="could not be made searchable"):
        vectors.repair_unreachable(vectors.find_unreachable())


# --- Rebuilding ------------------------------------------------------------------------------------------------


def test_rebuilding_a_healthy_collection_changes_nothing(vectors: Any) -> None:
    fill(vectors, 33)
    before = snapshot(vectors)
    assert vectors.rebuild_collection(page_size=10) == 33
    assert same(snapshot(vectors), before)
    assert vectors.get_collection().name == get_settings().collection_name


def test_a_rebuild_left_over_from_a_crash_does_not_get_in_the_way(vectors: Any) -> None:
    fill(vectors, 12)
    name = f"{get_settings().collection_name}__rebuild"
    vectors.get_client().create_collection(name=name, embedding_function=None)
    assert vectors.rebuild_collection() == 12
    assert [c.name for c in vectors.get_client().list_collections() if c.name == name] == []


def test_a_failed_rebuild_leaves_the_original_collection_untouched(
    vectors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill(vectors, 12)
    before = snapshot(vectors)
    monkeypatch.setattr(vectorstore, "find_unreachable", lambda *a, **k: ["still-broken"])
    with pytest.raises(VectorStoreError, match="still not fully reachable"):
        vectors.rebuild_collection()
    assert (
        same(snapshot(vectors), before)
        and vectors.get_collection().name == get_settings().collection_name
    )
    assert [
        c.name for c in vectors.get_client().list_collections() if c.name.endswith("__rebuild")
    ] == []


# --- Ids that were never stored, timeouts, and a rebuild a crash cut short ---------------------------------------


def test_ids_that_were_never_stored_count_as_unreachable_and_fail_loudly(vectors: Any) -> None:
    """A write that stored nothing is just as unsearchable as one the index cannot reach."""
    ids = fill(vectors, 10)
    asked = [*ids[:3], "ghost_chunk_0000"]
    assert vectors.find_unreachable(asked) == ["ghost_chunk_0000"]
    with pytest.raises(VectorStoreError, match="missing from the index"):
        vectors.ensure_reachable(asked)
    assert vectors.ensure_reachable(ids[:3]) == 0  # the real ones are fine


def test_the_client_has_request_timeouts_instead_of_waiting_forever(vectors: Any) -> None:
    timeout = vectors.get_client()._server._session.timeout
    assert timeout.read == 60.0 and timeout.connect == 5.0


def make_copy(vs: Any, count: int) -> Any:
    """A ``<name>__rebuild`` collection holding ``count`` vectors, as a rebuild leaves it."""
    name = f"{get_settings().collection_name}__rebuild"
    copy = vs.get_client().create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=vs._expected_metadata(),
        embedding_function=None,
    )
    rng = random.Random(9)
    ids = [f"copy_chunk_{n:04d}" for n in range(count)]
    copy.add(ids=ids, embeddings=[unit(rng) for _ in ids], documents=ids)
    return copy


def names(vs: Any) -> set[str]:
    """This test run's collections only: the Chroma server also holds the real ones."""
    prefix = get_settings().collection_name
    return {
        n for n in (getattr(c, "name", c) for c in vs.get_client().list_collections())
        if n.startswith(prefix)
    }  # fmt: skip


def test_nothing_to_recover_when_no_rebuild_was_interrupted(vectors: Any) -> None:
    fill(vectors, 5)
    assert vectors.recover_interrupted_rebuild() is None
    assert vectors.count() == 5


def test_an_incomplete_copy_beside_a_normal_collection_is_discarded(vectors: Any) -> None:
    fill(vectors, 10)
    make_copy(vectors, 4)  # the process died while copying
    assert vectors.recover_interrupted_rebuild() == "discarded"
    assert vectors.count() == 10
    assert names(vectors) == {get_settings().collection_name}


def test_a_verified_copy_replaces_a_collection_that_is_missing(vectors: Any) -> None:
    """Died between deleting the old collection and renaming the copy: every vector was stranded."""
    fill(vectors, 10)
    make_copy(vectors, 10)
    vectors.drop_collection()  # the crash happened right after this
    assert vectors.recover_interrupted_rebuild() == "completed"
    assert vectors.count() == 10
    assert names(vectors) == {get_settings().collection_name}
    assert vectors.find_unreachable() == []


def test_a_fuller_copy_replaces_an_empty_collection_created_by_a_restart(vectors: Any) -> None:
    """A restart before this fix opened (and so created) an empty collection beside the copy."""
    vectors.get_collection()  # exists, empty
    make_copy(vectors, 7)
    assert vectors.recover_interrupted_rebuild() == "completed"
    assert vectors.count() == 7


def test_startup_recovers_the_interrupted_rebuild_before_it_opens_the_collection(
    vectors: Any, db: Any
) -> None:
    from tests.conftest import REAL_RUN_STARTUP

    fill(vectors, 6)
    make_copy(vectors, 6)
    vectors.drop_collection()
    report = REAL_RUN_STARTUP()  # no indexed chunks in Postgres, so only the vectors matter
    assert report.vectors == 6


def test_writers_wait_for_a_rebuild_instead_of_being_lost_in_it(vectors: Any) -> None:
    import threading
    import time as _time

    fill(vectors, 20)
    started, finished = threading.Event(), threading.Event()
    real_add = None

    collection = vectors.get_collection()
    real_get = collection.get

    def slow_get(*args: Any, **kwargs: Any) -> Any:
        started.set()
        _time.sleep(0.3)  # keep the rebuild busy long enough for a writer to arrive
        return real_get(*args, **kwargs)

    collection.get = slow_get  # type: ignore[method-assign]
    order: list[str] = []

    def rebuild() -> None:
        vectors.rebuild_collection()
        order.append("rebuild done")
        finished.set()

    thread = threading.Thread(target=rebuild)
    thread.start()
    assert started.wait(10)
    rng = random.Random(5)
    meta = vectors.chunk_metadata(
        document_id="late", document_name="x", department="", document_type="",
        creation_date=date(2026, 1, 1), creation_date_int=20260101, access_level="Internal",
        page_number=1, page_end=1, section_title="",
    )  # fmt: skip
    vectors.upsert(["late_chunk_0000"], [unit(rng)], ["late"], [meta])  # blocks until the swap
    order.append("upsert done")
    thread.join(20)
    assert order == ["rebuild done", "upsert done"]
    assert real_add is None and "late_chunk_0000" in vectors.all_ids()  # not lost in the copy
    assert vectors.count() == 21
