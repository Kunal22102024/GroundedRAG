"""Hallucination guard (TDD 6.3): the relevance gate and citation verification helpers.

* **Relevance gate.** ``evaluate`` compares the best *dense cosine similarity* (never the RRF score,
  CLAUDE.md rule 7) with ``RELEVANCE_THRESHOLD``. Below it, the LLM is not called.
* **Lexical override.** An exact token such as ``ERR_404_NET`` can have low dense similarity even
  when the right chunk exists. If the question contains an identifier-like token and the top
  keyword hit contains that exact token (case-insensitive), the gate passes below the threshold,
  but only down to ``threshold - margin``: a question nowhere near the documents is still refused.
* **Citation filter.** ``CitationFilter`` checks ``[n]`` markers while the answer streams: a marker
  that does not map to a retrieved source is removed before the user ever sees it (rule 9).
  Brackets that are not citations (``[0-9]``, ``[2023-2025]``, anything inside code) are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.constants import REFUSAL_MESSAGE

# --- Relevance gate --------------------------------------------------------------------------

# Same shape as the keyword tokenizer: alphanumeric runs joined by identifier separators.
_WORD = re.compile(r"[^\W_]+(?:[_\-./:#@+][^\W_]+)*")
_JOINERS = "_-./:#@+"


@dataclass(frozen=True)
class GuardDecision:
    """Whether generation may proceed, and why."""

    passed: bool
    reason: Literal["relevant", "lexical_override", "below_threshold", "no_candidates"]
    score: float | None  # best dense cosine similarity among the candidates
    threshold: float


def _is_identifier(token: str) -> bool:
    """Error codes, IDs, times and versions: tokens a dense embedding cannot be trusted with."""
    if token.isdigit():
        # A bare number counts from three digits, but a year such as 2025 is just a date word.
        return len(token) >= 3 and not (len(token) == 4 and 1900 <= int(token) <= 2099)
    if "_" in token:
        return True
    has_digit = any(ch.isdigit() for ch in token)
    has_letter = any(ch.isalpha() for ch in token)
    has_joiner = any(ch in _JOINERS for ch in token)
    return has_digit and (has_letter or has_joiner)


def identifier_tokens(question: str) -> list[str]:
    """Identifier-like tokens in a question, in order, without duplicates."""
    found: dict[str, str] = {}
    for match in _WORD.finditer(question):
        token = match.group(0)
        if _is_identifier(token):
            found.setdefault(token.casefold(), token)
    return list(found.values())


def contains_token(text: str, token: str) -> bool:
    """True if ``token`` occurs in ``text`` as a whole token (case-insensitive)."""
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.IGNORECASE) is not None


def evaluate(
    question: str,
    best_similarity: float | None,
    top_keyword_text: str | None,
    threshold: float,
    margin: float,
) -> GuardDecision:
    """Decide whether the retrieved context is relevant enough to answer from.

    ``top_keyword_text`` is the text of the best keyword hit among the passages that would be
    sent to the model; ``margin`` is how far below ``threshold`` an exact identifier match may
    still pass.
    """
    if best_similarity is not None and best_similarity >= threshold:
        return GuardDecision(True, "relevant", best_similarity, threshold)
    if (
        best_similarity is not None
        and best_similarity >= threshold - margin
        and top_keyword_text
        and any(contains_token(top_keyword_text, token) for token in identifier_tokens(question))
    ):
        return GuardDecision(True, "lexical_override", best_similarity, threshold)
    reason = "below_threshold" if best_similarity is not None else "no_candidates"
    return GuardDecision(False, reason, best_similarity, threshold)


# --- Citations -------------------------------------------------------------------------------------

_GROUP = re.compile(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
_GROUP_PREFIX = re.compile(r"\[[\d\s,;–-]*")
_MAX_MARKER_CHARS = 32


def _numbers(group: str, source_count: int) -> list[int] | None:
    """The source numbers in ``[1, 3]`` or ``[1-3]``, in order (ranges are expanded).

    Returns None when the brackets are not a citation at all. A dash range is one only if it runs
    upward between two real sources, so ``[0-9]`` and ``[2023-2025]`` stay ordinary text.
    """
    numbers: list[int] = []
    for part in re.split(r"[,;]", group.strip("[]").strip()):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if len(bounds) == 1:
            numbers.extend(bounds)
        elif len(bounds) == 2 and 1 <= bounds[0] <= bounds[1] <= source_count:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            return None
    return numbers


class CitationFilter:
    """Streaming filter that keeps only ``[n]`` markers that map to a retrieved source.

    Feed it text as it arrives; it returns the text that is safe to show. A marker that may still
    be incomplete (``[1`` at the end of a piece) is held back until it resolves. ``[1, 2]`` is
    normalized to ``[1][2]``; numbers outside ``1..source_count`` are dropped and counted.

    Two details keep the answer text intact. A removed marker takes the one space in front of it
    along, so ``"25 days [9]."`` becomes ``"25 days."`` without touching any other spacing; that
    is why a single space is held back until the next character shows what follows it. And
    brackets inside inline code or a fenced block (``args[0]``) are code, not citations.
    """

    def __init__(self, source_count: int) -> None:
        self.source_count = source_count
        self.cited: list[int] = []
        self.invalid = 0
        self._pending = ""  # a possible marker that is still being read
        self._held = ""  # a space not shown yet, in case a removed marker follows it
        self._ticks = 0  # length of the current run of backticks
        self._inline_code = False
        self._code_fence = False

    def _resolve(self, group: str) -> str:
        if not _GROUP.fullmatch(group):
            return group  # brackets that are not a citation, e.g. "[]"
        numbers = _numbers(group, self.source_count)
        if numbers is None:
            return group  # e.g. "[0-9]" or "[2023-2025]"
        kept: list[int] = []
        for number in numbers:
            if 1 <= number <= self.source_count:
                if number not in kept:
                    kept.append(number)
                if number not in self.cited:
                    self.cited.append(number)
            else:
                self.invalid += 1
        return "".join(f"[{n}]" for n in kept)

    def _put(self, text: str, out: list[str]) -> None:
        """Show ``text``, preceded by the space that was held back."""
        if self._held:
            out.append(self._held)
            self._held = ""
        out.append(text)

    def _give_up_marker(self, out: list[str]) -> None:
        """What was read after a ``[`` turned out not to be a citation: show it as it is."""
        if self._pending:
            self._put(self._pending, out)
            self._pending = ""

    def _end_backtick_run(self) -> None:
        ticks, self._ticks = self._ticks, 0
        if ticks >= 3:
            self._code_fence = not self._code_fence
            self._inline_code = False
        elif not self._code_fence:
            self._inline_code = not self._inline_code

    def _step(self, char: str, out: list[str]) -> None:
        if char == "`":
            self._give_up_marker(out)
            self._ticks += 1
            self._put(char, out)
            return
        if self._ticks:
            self._end_backtick_run()
        if char == "\n":
            self._inline_code = False  # inline code never spans a line break
        if self._inline_code or self._code_fence:
            self._put(char, out)
            return
        if not self._pending:
            if char == "[":
                self._pending = "["
            elif char in " \t":
                if self._held:
                    out.append(self._held)
                self._held = char
            else:
                self._put(char, out)
            return
        if char == "[":  # the previous bracket never closed, so it was not a citation
            self._give_up_marker(out)
            self._pending = "["
            return
        self._pending += char
        if char == "]":
            resolved = self._resolve(self._pending)
            self._pending = ""
            if resolved:
                self._put(resolved, out)
            else:
                self._held = ""  # every number was invalid: the marker leaves with its space
        elif not _GROUP_PREFIX.fullmatch(self._pending) or len(self._pending) > _MAX_MARKER_CHARS:
            self._give_up_marker(out)

    def feed(self, text: str) -> str:
        """Take the next piece of the answer; return what may be shown now."""
        out: list[str] = []
        for char in text:
            self._step(char, out)
        return "".join(out)

    def finish(self) -> str:
        """End of the answer: a half-written marker (cut off by the token limit) is dropped."""
        pending, self._pending = self._pending, ""
        held, self._held = self._held, ""
        return held + ("" if _GROUP_PREFIX.fullmatch(pending) else pending)


def filter_citations(text: str, source_count: int) -> tuple[str, list[int], int]:
    """Run a whole answer through ``CitationFilter``: (clean text, cited numbers, invalid count)."""
    citation_filter = CitationFilter(source_count)
    cleaned = citation_filter.feed(text) + citation_filter.finish()
    return clean_text(cleaned), citation_filter.cited, citation_filter.invalid


def clean_text(text: str) -> str:
    """Trim the answer. The gaps left by removed markers are closed by ``CitationFilter``; the
    text is otherwise left exactly as the model wrote it (``.env``, ``:8080`` and code samples
    must not be rewritten)."""
    return text.strip()


_EMPHASIS = re.compile(r"[*_`\"'“”]")


def is_refusal(text: str) -> bool:
    """True if the model answered with the refusal sentence (optionally quoted, cited or bold)."""
    plain = _EMPHASIS.sub("", re.sub(r"\[[\d\s,;–-]*\]", "", text)).strip()
    return plain.casefold().rstrip(".") == REFUSAL_MESSAGE.casefold().rstrip(".")
