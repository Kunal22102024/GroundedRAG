"""Answer generation: stream the model's answer and verify its citations as it arrives.

* Tokens are streamed as they come (CLAUDE.md rule 20). A ``CitationFilter`` removes any ``[n]``
  that does not map to a retrieved source before it is shown (rule 9).
* If the answer cites nothing real (an invalid marker, or no citation at all), it is regenerated
  **once** with a stricter reminder; the caller is told to discard the partial text (``Restart``).
  If the second attempt is still wrong the invalid markers stay removed and the result is flagged.
* Finish reasons are handled explicitly (rule 19): ``MAX_TOKENS`` keeps the partial answer and
  flags it; a safety block or an empty response raises ``LlmError``, never a blank answer.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from app.constants import REFUSAL_MESSAGE
from app.core import guard, llm, prompts
from app.core.retrieval import Candidate

BLOCKED_MESSAGE = "The AI service declined to answer this request. Try rephrasing your question."
EMPTY_MESSAGE = "The AI service returned an empty answer. Please try again."


@dataclass(frozen=True)
class Restart:
    """The answer so far is discarded and a second attempt starts."""

    reason: str


@dataclass(frozen=True)
class Token:
    """Text safe to show to the user, in order."""

    text: str


@dataclass(frozen=True)
class AnswerResult:
    """The final, verified answer. Always the last event."""

    text: str
    cited: list[int]  # source numbers actually cited, in order of first use
    invalid_removed: int
    uncited: bool
    truncated: bool
    refused: bool  # the model itself answered with the refusal sentence
    retried: bool
    usage: dict[str, int]
    ttft_ms: float | None
    generate_ms: float


async def generate_answer(
    question: str, candidates: Sequence[Candidate]
) -> AsyncIterator[llm.Wait | Restart | Token | AnswerResult]:
    """Stream a cited answer to ``question`` from ``candidates``; raises ``llm.LlmError``."""
    started = time.perf_counter()
    usage: Counter[str] = Counter()
    ttft_ms: float | None = None
    attempt = 0
    while True:
        attempt += 1
        citation_filter = guard.CitationFilter(len(candidates))
        shown: list[str] = []
        finish: str | None = None
        received = False
        messages = prompts.build_answer_messages(question, candidates, strict=attempt == 2)
        async for item in llm.stream(messages, "answer"):
            if isinstance(item, llm.Wait):
                yield item
                continue
            usage.update(item.usage)
            finish = item.finish_reason or finish
            if item.blocked:
                raise llm.LlmError("blocked", BLOCKED_MESSAGE)
            if item.text:
                received = True
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started) * 1000, 1)
                visible = citation_filter.feed(item.text)
                if visible:
                    shown.append(visible)
                    yield Token(visible)
        tail = citation_filter.finish()
        if tail:
            shown.append(tail)
            yield Token(tail)

        if finish not in (None, "STOP", "MAX_TOKENS"):  # SAFETY, RECITATION, BLOCKLIST, ...
            raise llm.LlmError("blocked", BLOCKED_MESSAGE)
        if not received:
            raise llm.LlmError("empty", EMPTY_MESSAGE)

        text = guard.clean_text("".join(shown))
        refused = guard.is_refusal(text)
        verified = refused or (citation_filter.invalid == 0 and bool(citation_filter.cited))
        if verified or attempt == 2:
            break
        yield Restart("citations")

    if refused:
        text = REFUSAL_MESSAGE
    yield AnswerResult(
        text=text,
        cited=[] if refused else citation_filter.cited,
        invalid_removed=citation_filter.invalid,
        uncited=not refused and not citation_filter.cited,
        truncated=finish == "MAX_TOKENS",
        refused=refused,
        retried=attempt == 2,
        usage=dict(usage),
        ttft_ms=ttft_ms,
        generate_ms=round((time.perf_counter() - started) * 1000, 1),
    )
