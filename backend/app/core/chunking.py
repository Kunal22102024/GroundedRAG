"""Chunking: pack parsed segments into ~800-character chunks that respect structure.

Rules (TDD 3.1, CLAUDE.md rule 16):

* A chunk never crosses a section (heading) boundary.
* Chunks are packed by sentence, so text is only cut mid-sentence when a single
  sentence is longer than the target.
* A chunk may span pages; it then records ``page_number`` (first) and ``page_end``
  (last), so a citation always points at pages the text really occupies.
* Overlap is made of whole trailing sentences (never a mid-sentence fragment), only
  when they sit on the same page and section as what follows, and is capped at 1.5x
  the configured overlap. A chunk is therefore never longer than target + that cap,
  which is far below the 2,048-token embedding limit (``CHUNK_HARD_MAX_CHARS``).

The chunker sits behind the ``Chunker`` protocol so a semantic strategy can replace
it later without touching ingestion.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.constants import CHUNK_HARD_MAX_CHARS, MIN_CHUNK_CHARS
from app.core.parsers import Segment


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it has an id or metadata."""

    ordinal: int
    content: str
    page_number: int
    page_end: int
    section_title: str


class Chunker(Protocol):
    """Anything that can turn parsed segments into chunks."""

    def chunk(self, segments: Sequence[Segment]) -> list[ChunkDraft]:
        """Return chunks in reading order, numbered from 0."""
        ...


# --- Sentence splitting -----------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"""([.!?]["')\]]*)(\s+)(?=["(\[]?[A-Z0-9])""")
_LIST_MARKER_ONLY = re.compile(r"^\s*\(?\d{1,3}[.)]\s*$")
_ABBREVIATIONS = frozenset(
    [
        "e.g",
        "i.e",
        "vs",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "fig",
        "figs",
        "inc",
        "ltd",
        "cf",
        "al",
        "approx",
        "sec",
        "sect",
        "eq",
        "ref",
        "vol",
        "pp",
        "p",
    ]
)


def _ends_with_abbreviation(candidate: str) -> bool:
    if _LIST_MARKER_ONLY.match(candidate):
        return True
    last = re.split(r"\s+", candidate.rstrip())[-1].rstrip(".!?\"')]").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())  # "J. Smith"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation followed by a capital or digit."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        candidate = text[start : match.end(1)]
        if _ends_with_abbreviation(candidate):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _split_long(text: str, limit: int) -> list[str]:
    """Cut text that is longer than ``limit`` at word boundaries (hard cut as a last resort)."""
    parts: list[str] = []
    while len(text) > limit:
        cut = max(text.rfind(" ", 0, limit + 1), text.rfind("\n", 0, limit + 1))
        if cut < limit // 2:  # no usable whitespace: a single enormous token
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return [p for p in parts if p]


# --- Packing ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    """A sentence-sized unit with the separator that joins it to the previous piece."""

    text: str
    page: int
    sep: str  # "\n" when it starts a new paragraph/line/row, " " when it continues one


def _pieces(segment: Segment, target: int) -> list[_Piece]:
    text = segment.text
    if len(text) <= target:
        return [_Piece(text, segment.page, "\n")]
    pieces: list[_Piece] = []
    # Structured text (lists, tables, code) keeps its line breaks.
    for line in (part for part in text.split("\n") if part.strip()):
        sentences = split_sentences(line) if len(line) > target else [line.strip()]
        for sentence_index, sentence in enumerate(sentences):
            for part_index, part in enumerate(_split_long(sentence, target)):
                # Each source line starts on a new line; sentences and cut parts continue it.
                sep = "\n" if sentence_index == 0 and part_index == 0 else " "
                pieces.append(_Piece(part, segment.page, sep))
    return pieces


class StructureAwareChunker:
    """The baseline chunking strategy (TDD 3.1, phase 1)."""

    def __init__(self, target_chars: int = 800, overlap_chars: int = 100) -> None:
        if target_chars < 100:
            raise ValueError("target_chars must be at least 100")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be between 0 and target_chars")
        self.target = target_chars
        self.overlap = overlap_chars
        self.overlap_cap = int(overlap_chars * 1.5)
        if self.target + self.overlap_cap > CHUNK_HARD_MAX_CHARS:
            raise ValueError("chunk size would exceed CHUNK_HARD_MAX_CHARS")

    def chunk(self, segments: Sequence[Segment]) -> list[ChunkDraft]:
        """Pack ``segments`` into chunks, in order, numbered from 0."""
        drafts: list[tuple[str, int, int, str]] = []  # content, first page, last page, section
        current: list[_Piece] = []
        seed_len = (
            0  # length of the overlap prefix inside ``current`` (not counted against the target)
        )
        length = 0
        section: str | None = None

        def render(pieces: list[_Piece]) -> str:
            return "".join(p.text if i == 0 else p.sep + p.text for i, p in enumerate(pieces))

        def emit() -> None:
            if current and section is not None:
                content = render(current).strip()
                if content:
                    drafts.append(
                        (
                            content,
                            min(p.page for p in current),
                            max(p.page for p in current),
                            section,
                        )
                    )

        def overlap_for(next_piece: _Piece) -> list[_Piece]:
            """Whole trailing sentences of ``current`` that may repeat at the start of the next chunk."""
            if self.overlap == 0:
                return []
            seed: list[_Piece] = []
            total = 0
            for piece in reversed(current):
                cost = len(piece.text) + (len(piece.sep) if seed else 0)
                if piece.page != next_piece.page or total + cost > self.overlap_cap:
                    break
                seed.insert(0, piece)
                total += cost
                if total >= self.overlap:
                    break
            return seed

        for segment in segments:
            if segment.section != section:
                emit()
                current, seed_len, length = [], 0, 0
                section = segment.section
            for piece in _pieces(segment, self.target):
                joined = len(piece.text) + (len(piece.sep) if current else 0)
                if current and length + joined > self.target + seed_len:
                    emit()
                    seed = overlap_for(piece)
                    current = list(seed)
                    seed_len = len(render(seed)) if seed else 0
                    length = seed_len
                    joined = len(piece.text) + (len(piece.sep) if current else 0)
                current.append(piece)
                length += joined
        emit()

        return self._finish(drafts)

    def _finish(self, drafts: list[tuple[str, int, int, str]]) -> list[ChunkDraft]:
        """Merge tiny chunks into a same-section neighbour, drop pure noise, number the rest."""
        merged: list[tuple[str, int, int, str]] = []
        for content, first, last, section in drafts:
            if (
                len(content) < MIN_CHUNK_CHARS
                and merged
                and merged[-1][3] == section
                and len(merged[-1][0]) + 1 + len(content) <= self.target + self.overlap_cap
            ):
                prev_content, prev_first, prev_last, _ = merged[-1]
                merged[-1] = (
                    f"{prev_content}\n{content}",
                    prev_first,
                    max(prev_last, last),
                    section,
                )
            else:
                merged.append((content, first, last, section))

        result: list[ChunkDraft] = []
        for content, first, last, section in merged:
            if sum(ch.isalnum() for ch in content) < 3:
                continue  # stray punctuation or a lone page marker
            assert len(content) <= CHUNK_HARD_MAX_CHARS, "chunk exceeds the embedding-safe ceiling"
            result.append(ChunkDraft(len(result), content, first, last, section))
        return result
