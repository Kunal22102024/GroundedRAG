"""Chunking: structure, sentence integrity, overlap, page ranges and size ceilings."""

from __future__ import annotations

import re
import time

import pytest

from app.constants import CHUNK_HARD_MAX_CHARS
from app.core.chunking import StructureAwareChunker, split_sentences
from app.core.parsers import Segment

TARGET, OVERLAP = 800, 100
CAP = int(OVERLAP * 1.5)


def chunker() -> StructureAwareChunker:
    return StructureAwareChunker(TARGET, OVERLAP)


def numbered_sentences(count: int, start: int = 1) -> str:
    return " ".join(
        f"Sentence number {n} describes the retention rule in plain words."
        for n in range(start, start + count)
    )


# --- Packing and size ---------------------------------------------------------------------


def test_short_segments_are_packed_together_in_order() -> None:
    segments = [Segment(1, "", f"Paragraph {n} is short.") for n in range(5)]
    chunks = chunker().chunk(segments)
    assert len(chunks) == 1
    assert chunks[0].content == "\n".join(s.text for s in segments)
    assert (chunks[0].page_number, chunks[0].page_end) == (1, 1)


def test_chunks_respect_target_and_never_exceed_the_hard_ceiling() -> None:
    segments = [Segment(1 + n // 4, "S", numbered_sentences(30, n * 30)) for n in range(12)]
    for chunk in chunker().chunk(segments):
        assert len(chunk.content) <= TARGET + CAP
        assert len(chunk.content) <= CHUNK_HARD_MAX_CHARS


def test_ordinals_are_sequential_from_zero() -> None:
    chunks = chunker().chunk([Segment(1, "", numbered_sentences(60))])
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len(chunks) > 3


def test_empty_input_gives_no_chunks() -> None:
    assert chunker().chunk([]) == []


def test_chunking_is_deterministic() -> None:
    segments = [
        Segment(1, "A", numbered_sentences(40)),
        Segment(2, "B", numbered_sentences(40, 100)),
    ]
    assert chunker().chunk(segments) == chunker().chunk(segments)


# --- Sentence integrity ---------------------------------------------------------------------


def test_chunks_start_and_end_on_sentence_boundaries() -> None:
    chunks = chunker().chunk([Segment(1, "", numbered_sentences(80))])
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.content.startswith("Sentence number "), chunk.content[:40]
        assert chunk.content.endswith("plain words."), chunk.content[-40:]


def test_no_sentence_is_lost() -> None:
    text = numbered_sentences(90)
    chunks = chunker().chunk([Segment(1, "", text)])
    covered = " ".join(c.content for c in chunks)
    for n in range(1, 91):
        assert f"Sentence number {n} " in covered


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three? Four.", ["One.", "Two!", "Three?", "Four."]),
        ("Use e.g. this one. Then stop.", ["Use e.g. this one.", "Then stop."]),
        ("Ask Dr. Smith today. He agrees.", ["Ask Dr. Smith today.", "He agrees."]),
        ("1. Open the door. 2. Close it.", ["1. Open the door.", "2. Close it."]),
        ("Written by J. Smith in 2024. Done.", ["Written by J. Smith in 2024.", "Done."]),
        ("It costs 3.50 euros. Pay now.", ["It costs 3.50 euros.", "Pay now."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        (
            "no capital after this. lower case continues",
            ["no capital after this. lower case continues"],
        ),
    ],
)
def test_sentence_splitting(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


# --- Overlap ---------------------------------------------------------------------------------


def test_overlap_repeats_whole_trailing_sentences_within_the_cap() -> None:
    chunks = chunker().chunk([Segment(1, "", numbered_sentences(60))])
    overlaps = 0
    for previous, current in zip(chunks, chunks[1:], strict=False):
        sentences = split_sentences(current.content)
        shared = [s for s in sentences if s in split_sentences(previous.content)]
        if shared:
            overlaps += 1
            assert sentences[: len(shared)] == shared, "overlap must be at the start of the chunk"
            assert len(" ".join(shared)) <= CAP
            assert shared == split_sentences(previous.content)[-len(shared) :], (
                "and be the previous chunk's tail"
            )
    assert overlaps >= len(chunks) - 2, "nearly every boundary should overlap"


def test_zero_overlap_means_chunks_do_not_repeat_text() -> None:
    chunks = StructureAwareChunker(TARGET, 0).chunk([Segment(1, "", numbered_sentences(60))])
    assert not set(split_sentences(chunks[0].content)) & set(split_sentences(chunks[1].content))


def test_no_overlap_across_a_section_boundary() -> None:
    chunks = chunker().chunk(
        [Segment(1, "One", numbered_sentences(30)), Segment(1, "Two", numbered_sentences(30, 500))]
    )
    first_of_two = next(c for c in chunks if c.section_title == "Two")
    assert first_of_two.content.startswith("Sentence number 500 ")
    assert {c.section_title for c in chunks} == {"One", "Two"}


def test_no_overlap_across_a_page_boundary_so_page_ranges_stay_exact() -> None:
    page1 = Segment(1, "S", numbered_sentences(14))
    page2 = Segment(2, "S", numbered_sentences(14, 100))
    chunks = chunker().chunk([page1, page2])
    for chunk in chunks:
        numbers = [int(m) for m in re.findall(r"Sentence number (\d+) ", chunk.content)]
        on_page1 = any(n < 100 for n in numbers)
        on_page2 = any(n >= 100 for n in numbers)
        assert chunk.page_number == (1 if on_page1 else 2)
        assert chunk.page_end == (2 if on_page2 else 1)


# --- Pages and sections ----------------------------------------------------------------------


def test_a_chunk_spanning_pages_records_the_range() -> None:
    chunks = chunker().chunk(
        [Segment(1, "S", "End of page one."), Segment(2, "S", "Start of page two.")]
    )
    assert len(chunks) == 1
    assert (chunks[0].page_number, chunks[0].page_end) == (1, 2)


def test_page_numbers_are_preserved_for_long_documents() -> None:
    segments = [
        Segment(page, f"Section {page}", f"Marker{page} " + numbered_sentences(3))
        for page in range(1, 40)
    ]
    for chunk in chunker().chunk(segments):
        marker = int(re.search(r"Marker(\d+)", chunk.content).group(1))  # type: ignore[union-attr]
        assert chunk.page_number == marker and chunk.page_end == marker
        assert chunk.section_title == f"Section {marker}"


def test_different_sections_never_share_a_chunk() -> None:
    segments = [
        Segment(1, name, f"{name} body text that is quite short.")
        for name in ("Alpha", "Beta", "Gamma")
    ]
    chunks = chunker().chunk(segments)
    assert [c.section_title for c in chunks] == ["Alpha", "Beta", "Gamma"]


# --- Oversized input ---------------------------------------------------------------------------


def test_a_sentence_longer_than_the_target_is_split_at_word_boundaries() -> None:
    sentence = "word " * 400  # 2,000 characters, no sentence end
    chunks = chunker().chunk([Segment(1, "", sentence.strip())])
    assert len(chunks) >= 3
    assert all(len(c.content) <= TARGET for c in chunks)
    assert all(not c.content.startswith("ord") and not c.content.endswith("wor") for c in chunks)
    assert sum(c.content.count("word") for c in chunks) == 400


def test_a_single_token_longer_than_the_target_is_hard_cut() -> None:
    chunks = chunker().chunk([Segment(1, "", "A" * 3000)])
    assert all(len(c.content) <= TARGET for c in chunks)
    assert sum(len(c.content) for c in chunks) == 3000


def test_cjk_text_without_spaces_stays_under_the_ceiling() -> None:
    chunks = chunker().chunk([Segment(1, "", "数据保留政策适用于所有员工。" * 400)])
    assert chunks and all(len(c.content) <= CHUNK_HARD_MAX_CHARS for c in chunks)


def test_structured_text_keeps_its_line_breaks_when_split() -> None:
    rows = "\n".join(f"Row {n} | value {n} | note about row {n}" for n in range(60))
    chunks = chunker().chunk([Segment(1, "T", rows)])
    assert len(chunks) > 1
    for chunk in chunks:
        assert all(line.startswith("Row ") for line in chunk.content.split("\n"))


# --- Tiny and junk chunks ------------------------------------------------------------------------


def test_tiny_trailing_text_is_merged_into_the_previous_chunk_of_the_same_section() -> None:
    body = numbered_sentences(10)
    chunks = chunker().chunk([Segment(1, "S", body), Segment(1, "S", "End.")])
    assert "End." in chunks[-1].content
    assert all(len(c.content) >= 20 for c in chunks)


def test_punctuation_only_chunks_are_dropped() -> None:
    chunks = chunker().chunk(
        [Segment(1, "A", "Real content that is long enough."), Segment(1, "B", "...")]
    )
    assert [c.section_title for c in chunks] == ["A"]


# --- Configuration ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "overlap"),
    # 1200 + 1.5 x 200 + 1 separator is one character over the ceiling; so is 1350 + 150 + 1.
    [(50, 10), (800, 800), (800, -1), (1400, 200), (1200, 200), (1350, 100)],
)
def test_invalid_configuration_is_rejected(target: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        StructureAwareChunker(target, overlap)


def test_the_largest_allowed_configuration_still_respects_the_ceiling() -> None:
    biggest = StructureAwareChunker(1199, 200)
    for chunk in biggest.chunk([Segment(1, "", numbered_sentences(200))]):
        assert len(chunk.content) <= CHUNK_HARD_MAX_CHARS


def test_the_longest_possible_chunk_is_exactly_at_the_bound_not_over_it() -> None:
    """Overlap (up to the cap) + one separator + a full-size piece: the worst case of the packer."""
    target, overlap = 1349, 100
    cap = int(overlap * 1.5)
    chunker_ = StructureAwareChunker(target, overlap)
    short = "x" * (cap - 5) + "."  # becomes the overlap of the next chunk
    full = ("y" * (target - 1)) + "."
    chunks = chunker_.chunk(
        [
            Segment(1, "S", short),
            Segment(1, "S", full),
            Segment(1, "S", short),
            Segment(1, "S", full),
        ]
    )
    assert max(len(c.content) for c in chunks) <= target + cap + 1 <= CHUNK_HARD_MAX_CHARS


# --- Hostile text: the time taken grows with the length, not with its square ------------------------------


def seconds(fn: object) -> float:
    start = time.perf_counter()
    fn()  # type: ignore[operator]
    return time.perf_counter() - start


def test_a_run_of_abbreviation_like_boundaries_is_split_in_linear_time() -> None:
    """'A. A. A. ...' skips every boundary; re-splitting the growing candidate took 50 s per 60 KB."""
    assert seconds(lambda: split_sentences("A. " * 60_000)) < 2.0
    assert seconds(lambda: split_sentences("see e.g. Foo " * 40_000)) < 2.0


def test_abbreviations_and_list_numbers_still_do_not_end_a_sentence() -> None:
    assert split_sentences("See Fig. 3 for details. Next one.") == [
        "See Fig. 3 for details.",
        "Next one.",
    ]
    assert split_sentences("1. First item. 2. Second item.") == [
        "1. First item.",
        "2. Second item.",
    ]
    assert split_sentences("Ask J. Smith about it. Done.") == ["Ask J. Smith about it.", "Done."]


def test_a_huge_text_without_whitespace_is_cut_in_linear_time() -> None:
    chunks_: list[object] = []
    elapsed = seconds(lambda: chunks_.extend(chunker().chunk([Segment(1, "", "x" * 3_000_000)])))
    assert elapsed < 5.0 and len(chunks_) > 1000


def test_a_large_realistic_document_is_chunked_quickly() -> None:
    segments = [Segment(1, f"S{n}", numbered_sentences(40, n * 40)) for n in range(400)]
    assert seconds(lambda: chunker().chunk(segments)) < 5.0
