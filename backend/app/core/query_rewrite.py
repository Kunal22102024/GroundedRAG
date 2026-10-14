"""Follow-up rewriting (TDD 5.1): turn "what about contractors?" into a standalone question.

Skipped on the first turn (no history, no LLM call). Uses a small, thinking-disabled call. If the
rewrite fails for any reason the original question is used: retrieval can still work without it,
and the answer call reports a real outage itself.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.constants import QUESTION_MAX_CHARS
from app.core import llm, prompts

logger = logging.getLogger(__name__)

_LABEL = re.compile(r"^\s*(?:standalone\s+question|rewritten\s+question|question)\s*:\s*", re.I)
# Resolving "what about contractors?" adds a few words. A rewrite many times longer than the
# follow-up is not a rewrite: the model answered, or followed text it found in an earlier answer.
_GROWTH_LIMIT = 3
_MIN_LENGTH_LIMIT = 300


def _clean(raw: str, original: str) -> str:
    """The rewritten question: first non-empty line, without labels or surrounding quotes.

    Returns an empty string (so the caller falls back to ``original``) when the line is far longer
    than a rewrite of ``original`` can reasonably be.
    """
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = _LABEL.sub("", line).strip().strip("\"'`“”").strip()
    limit = min(QUESTION_MAX_CHARS, max(_MIN_LENGTH_LIMIT, _GROWTH_LIMIT * len(original)))
    return line if len(line) <= limit else ""


async def rewrite_question(question: str, history: Sequence[tuple[str, str]]) -> str:
    """A standalone version of ``question`` given ``history`` as (role, content) pairs."""
    if not history:
        return question
    parts: list[str] = []
    try:
        async for item in llm.stream(prompts.build_rewrite_messages(question, history), "rewrite"):
            if isinstance(item, llm.Piece):
                parts.append(item.text)
    except llm.LlmError as exc:
        logger.warning(
            "follow-up rewrite failed; using the original question", extra={"code": exc.code}
        )
        return question
    return _clean("".join(parts), question) or question
