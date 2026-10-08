"""Hallucination guard (TDD 6.3): the relevance gate and citation verification helpers.

* **Relevance gate.** ``evaluate`` compares the best *dense cosine similarity* (never the RRF score,
  CLAUDE.md rule 7) with ``RELEVANCE_THRESHOLD``. Below it, the LLM is not called.
* **Lexical override.** An exact token such as ``ERR_404_NET`` can have low dense similarity even
  when the right chunk exists. If the question contains an identifier-like token and the top
  keyword hit contains that exact token (case-insensitive), the gate passes regardless of the score.
* **Citation filter.** ``CitationFilter`` checks ``[n]`` markers while the answer streams: a marker
  that does not map to a retrieved source is removed before the user ever sees it (rule 9).
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
) -> GuardDecision:
    """Decide whether the retrieved context is relevant enough to answer from."""
    if best_similarity is not None and best_similarity >= threshold:
        return GuardDecision(True, "relevant", best_similarity, threshold)
    if top_keyword_text and any(
        contains_token(top_keyword_text, token) for token in identifier_tokens(question)
    ):
        return GuardDecision(True, "lexical_override", best_similarity, threshold)
    reason = "below_threshold" if best_similarity is not None else "no_candidates"
    return GuardDecision(False, reason, best_similarity, threshold)


# --- Citations -------------------------------------------------------------------------------------

_GROUP = re.compile(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
_GROUP_PREFIX = re.compile(r"\[[\d\s,;–-]*")
_MAX_MARKER_CHARS = 32
_MAX_RANGE = 20


def _numbers(group: str) -> list[int]:
    """The source numbers in ``[1, 3]`` or ``[1-3]``, in order (ranges are expanded)."""
    numbers: list[int] = []
    for part in re.split(r"[,;]", group.strip("[] \t")):
        bounds = [int(x) for x in re.findall(r"\d+", part)]
        if len(bounds) == 2 and 0 <= bounds[1] - bounds[0] < _MAX_RANGE:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            numbers.extend(bounds)
    return numbers


class CitationFilter:
    """Streaming filter that keeps only ``[n]`` markers that map to a retrieved source.

    Feed it text as it arrives; it returns the text that is safe to show. A marker that may still
    be incomplete (``[1`` at the end of a piece) is held back until it resolves. ``[1, 2]`` is
    normalized to ``[1][2]``; numbers outside ``1..source_count`` are dropped and counted.
    """

    def __init__(self, source_count: int) -> None:
        self.source_count = source_count
        self.cited: list[int] = []
        self.invalid = 0
        self._pending = ""

    def _resolve(self, group: str) -> str:
        if not _GROUP.fullmatch(group):
            return group  # brackets that are not a citation, e.g. "[]" or a code sample
        kept: list[int] = []
        for number in _numbers(group):
            if 1 <= number <= self.source_count:
                if number not in kept:
                    kept.append(number)
                if number not in self.cited:
                    self.cited.append(number)
            else:
                self.invalid += 1
        return "".join(f"[{n}]" for n in kept)

    def feed(self, text: str) -> str:
        """Take the next piece of the answer; return what may be shown now."""
        out: list[str] = []
        for char in text:
            if not self._pending:
                if char == "[":
                    self._pending = "["
                else:
                    out.append(char)
                continue
            if char == "[":  # the previous bracket never closed, so it was not a citation
                out.append(self._pending)
                self._pending = "["
                continue
            self._pending += char
            if char == "]":
                out.append(self._resolve(self._pending))
                self._pending = ""
            elif (
                not _GROUP_PREFIX.fullmatch(self._pending) or len(self._pending) > _MAX_MARKER_CHARS
            ):
                out.append(self._pending)
                self._pending = ""
        return "".join(out)

    def finish(self) -> str:
        """End of the answer: a half-written marker (cut off by the token limit) is dropped."""
        pending, self._pending = self._pending, ""
        return "" if _GROUP_PREFIX.fullmatch(pending) else pending


def filter_citations(text: str, source_count: int) -> tuple[str, list[int], int]:
    """Run a whole answer through ``CitationFilter``: (clean text, cited numbers, invalid count)."""
    citation_filter = CitationFilter(source_count)
    cleaned = citation_filter.feed(text) + citation_filter.finish()
    return clean_text(cleaned), citation_filter.cited, citation_filter.invalid


def clean_text(text: str) -> str:
    """Tidy the gaps left where invalid markers were removed."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return text.strip()


def is_refusal(text: str) -> bool:
    """True if the model answered with the refusal sentence (optionally quoted or cited)."""
    plain = re.sub(r"\[[\d\s,;–-]*\]", "", text).strip().strip("\"'`“”").strip()
    return plain.casefold().rstrip(".") == REFUSAL_MESSAGE.casefold().rstrip(".")
