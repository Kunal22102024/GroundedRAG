"""BM25 keyword index: identifier-aware tokenizer, ranking, filtering and updates."""

from __future__ import annotations

import threading

import pytest

from app.core.bm25 import Bm25Index, ChunkMeta, tokenize


def meta(
    chunk_id: str,
    document_id: str = "d1",
    department: str = "IT",
    level: str = "Internal",
    date: int = 20260101,
) -> ChunkMeta:
    return ChunkMeta(chunk_id, document_id, department, "Policy", date, level)


CORPUS = [
    (meta("c1"), "Error ERR_404_NET means the upstream gateway could not be reached."),
    (meta("c2"), "The retry limit for API timeouts is three attempts with exponential backoff."),
    (meta("c3"), "Server reboot windows are 02:00 to 04:00 UTC on Sundays for security patches."),
    (
        meta("c4", "d2", "HR", "Restricted", 20240101),
        "Parental leave policy: 26 weeks of paid leave for all employees.",
    ),
]


@pytest.fixture
def index() -> Bm25Index:
    idx = Bm25Index()
    idx.rebuild(CORPUS)
    return idx


# --- Tokenizer ----------------------------------------------------------------------------------


def test_identifiers_are_kept_whole_and_split_into_parts() -> None:
    assert tokenize("ERR_404_NET") == ["err_404_net", "err", "404", "net"]
    assert tokenize("v2.3.1") == ["v2.3.1", "v2", "3", "1"]
    assert tokenize("192.168.0.1") == ["192.168.0.1", "192", "168", "0", "1"]
    assert tokenize("RFC-9110") == ["rfc-9110", "rfc", "9110"]


def test_plain_words_are_lowercased_and_single_letters_dropped() -> None:
    assert tokenize("The Quick brown FOX, a b c 7") == ["the", "quick", "brown", "fox", "7"]


def test_sentence_punctuation_does_not_glue_words_together() -> None:
    assert tokenize("end. Next, word;") == ["end", "next", "word"]


def test_non_ascii_letters_are_words() -> None:
    assert tokenize("Café résumé Größe") == ["café", "résumé", "grösse"]  # casefold: ß -> ss


def test_empty_and_symbol_only_text() -> None:
    assert tokenize("") == [] and tokenize("!!! ... ---") == []


# --- Searching -----------------------------------------------------------------------------------


def test_exact_identifier_finds_its_chunk_first(index: Bm25Index) -> None:
    hits = index.search("ERR_404_NET", 5)
    assert hits[0].chunk_id == "c1"


def test_identifier_parts_match_a_loosely_phrased_query(index: Bm25Index) -> None:
    assert index.search("what does 404 net error mean", 3)[0].chunk_id == "c1"


def test_matching_is_case_insensitive(index: Bm25Index) -> None:
    assert index.search("err_404_NET", 1)[0].chunk_id == "c1"


def test_ranking_prefers_the_chunk_with_more_query_terms(index: Bm25Index) -> None:
    hits = index.search("retry limit API timeouts", 4)
    assert hits[0].chunk_id == "c2"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_query_with_no_overlap_returns_nothing(index: Bm25Index) -> None:
    assert index.search("quantum chromodynamics", 5) == []
    assert index.search("", 5) == [] and index.search("a b", 5) == []


def test_n_limits_results(index: Bm25Index) -> None:
    assert len(index.search("the", 2)) == 2
    assert index.search("the", 0) == []


def test_empty_index_returns_nothing() -> None:
    assert Bm25Index().search("anything", 5) == []


# --- Filter parity -----------------------------------------------------------------------------------


def test_allow_predicate_restricts_results_before_ranking(index: Bm25Index) -> None:
    hits = index.search("leave policy employees", 5)
    assert hits[0].chunk_id == "c4"
    only_internal = index.search(
        "leave policy employees", 5, allow=lambda m: m.access_level != "Restricted"
    )
    assert "c4" not in {h.chunk_id for h in only_internal}


def test_filter_on_department_date_and_document(index: Bm25Index) -> None:
    everything = "leave retry gateway reboot"  # words from all four chunks
    assert {h.chunk_id for h in index.search(everything, 10)} == {"c1", "c2", "c3", "c4"}
    assert {
        h.chunk_id for h in index.search(everything, 10, allow=lambda m: m.department == "HR")
    } == {"c4"}
    recent = index.search(everything, 10, allow=lambda m: m.creation_date_int >= 20250101)
    assert {h.chunk_id for h in recent} == {"c1", "c2", "c3"}
    assert {
        h.chunk_id for h in index.search(everything, 10, allow=lambda m: m.document_id == "d2")
    } == {"c4"}


def test_a_filtered_out_chunk_never_leaks_even_with_an_exact_match(index: Bm25Index) -> None:
    assert index.search("ERR_404_NET", 5, allow=lambda m: False) == []


# --- Updating ----------------------------------------------------------------------------------------------


def test_add_document_makes_chunks_searchable(index: Bm25Index) -> None:
    index.add_document([(meta("n1", "d9"), "Quokka migration patterns across the continent.")])
    assert index.search("quokka", 3)[0].chunk_id == "n1"
    assert len(index) == 5


def test_add_document_replaces_the_same_documents_old_chunks(index: Bm25Index) -> None:
    index.add_document([(meta("c1x", "d1"), "Completely different text about zebras.")])
    assert index.search("ERR_404_NET", 3) == []
    assert index.search("zebras", 3)[0].chunk_id == "c1x"


def test_remove_document(index: Bm25Index) -> None:
    assert index.remove_document("d2") == 1
    assert index.search("parental leave", 5) == []
    assert index.remove_document("d2") == 0
    assert len(index) == 3


def test_rebuild_replaces_everything(index: Bm25Index) -> None:
    index.rebuild([(meta("z1"), "Only this remains.")])
    assert index.chunk_ids() == {"z1"} and index.search("retry", 3) == []


def test_chunks_without_any_tokens_do_not_break_the_index() -> None:
    idx = Bm25Index()
    idx.rebuild([(meta("e1"), "!!! ..."), (meta("e2"), "real words here")])
    assert [h.chunk_id for h in idx.search("words", 3)] == ["e2"]


def test_searches_during_updates_always_see_a_consistent_snapshot() -> None:
    idx = Bm25Index()
    idx.rebuild(CORPUS)
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        n = 0
        while not stop.is_set():
            n += 1
            idx.add_document([(meta(f"w{n}", "dw"), f"writer chunk {n} about gateway retries")])
            idx.remove_document("dw")

    def reader() -> None:
        try:
            for _ in range(200):
                for hit in idx.search("gateway", 5):
                    assert hit.score > 0
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    thread.join()
    assert errors == []
