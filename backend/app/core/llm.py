"""The Gemini chat model: the only place the LLM client is built (CLAUDE.md rule 2).

* Temperature and the **thinking budget (0)** come from settings (rule 18). Gemini Flash
  models are reasoning models; unbounded thinking adds latency and eats the output token limit.
* The library's own retries are switched off (``max_retries=1``): they ignore the server's retry
  hint. ``stream`` retries instead, honouring the hint, for a short bounded time, because a user
  is waiting. A rate limit that outlasts that becomes a friendly ``LlmError`` (never a hang).
* ``stream`` yields ``Wait`` (rate limited, retrying shortly) and ``Piece`` items. Errors are
  raised as ``LlmError`` with a message that is safe to show to the user.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import BaseMessage

from app.config import get_settings
from app.constants import REWRITE_MAX_TOKENS
from app.core.gemini_errors import Classified, classify

logger = logging.getLogger(__name__)

RATE_LIMITED_MESSAGE = "Rate limit reached, please try again in a minute."
_BACKOFF_BASE_SECONDS = 2.0

# Indirection so tests can run without really sleeping.
_sleep = asyncio.sleep
_jitter = random.uniform


class LlmError(Exception):
    """The model could not answer. ``code`` is machine-readable; ``message`` is for the user."""

    def __init__(self, code: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after = retry_after


@dataclass(frozen=True)
class Wait:
    """The call was rate limited; the next attempt starts in ``seconds``."""

    seconds: float


@dataclass(frozen=True)
class Piece:
    """One streamed chunk of model output."""

    text: str
    finish_reason: str | None
    blocked: bool
    usage: dict[str, int]


@lru_cache
def _build(model: str, max_tokens: int) -> Any:
    settings = get_settings()
    api_key = settings.google_api_key.get_secret_value()
    if not api_key:
        raise LlmError(
            "not_configured", "GOOGLE_API_KEY is not set, so questions cannot be answered."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=max_tokens,
        thinking_budget=settings.llm_thinking_budget,
        max_retries=1,
        timeout=settings.llm_timeout_seconds,
    )


Kind = Literal["answer", "rewrite", "analysis"]


def get_chat_model(kind: Kind = "answer") -> Any:
    """The chat model for answering, rewriting follow-ups or analysing (built once per setting)."""
    settings = get_settings()
    if kind == "rewrite":
        return _build(settings.rewrite_model_id, REWRITE_MAX_TOKENS)
    if kind == "analysis":  # summaries, tables and comparisons: same model, a larger output budget
        return _build(settings.llm_model, settings.analysis_max_tokens)
    return _build(settings.llm_model, settings.llm_max_tokens)


def clear_cache() -> None:
    """Forget built models (tests)."""
    _build.cache_clear()


def message_text(content: Any) -> str:
    """Text of a model message whose content is a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _is_blocked(reason: Any) -> bool:
    name = str(getattr(reason, "name", reason) or "")
    return name not in ("", "0", "None", "BLOCK_REASON_UNSPECIFIED")


def _piece(chunk: Any) -> Piece:
    metadata = getattr(chunk, "response_metadata", None) or {}
    feedback = metadata.get("prompt_feedback") or {}
    usage = getattr(chunk, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    return Piece(
        text=message_text(chunk.content),
        finish_reason=metadata.get("finish_reason"),
        blocked=_is_blocked(feedback.get("block_reason")),
        usage={
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "reasoning_tokens": int(details.get("reasoning") or 0),
        },
    )


def _to_error(exc: BaseException, info: Classified) -> LlmError:
    if info.kind == "rate_limit":
        return LlmError("rate_limited", RATE_LIMITED_MESSAGE, info.retry_after)
    if info.kind == "quota":
        return LlmError(
            "quota_exhausted",
            "The daily AI usage limit has been reached. Please try again tomorrow.",
        )
    if info.kind == "auth":
        return LlmError(
            "auth", "The AI service rejected the API key. Please contact your administrator."
        )
    if info.kind == "transient":
        return LlmError(
            "unavailable",
            "The AI service is temporarily unavailable. Please try again in a moment.",
        )
    return LlmError("failed", f"The AI service could not answer ({type(exc).__name__}).")


async def stream(
    messages: Sequence[BaseMessage], kind: Kind = "answer", max_retries: int | None = None
) -> AsyncIterator[Wait | Piece]:
    """Stream the model's reply, retrying a rate limit briefly before any output has been sent.

    ``max_retries`` replaces ``INTERACTIVE_MAX_RETRIES`` for work that can afford to wait longer
    (analysis); each wait is still capped by ``INTERACTIVE_MAX_WAIT_SECONDS``.
    """
    settings = get_settings()
    model = get_chat_model(kind)
    limit = settings.interactive_max_retries if max_retries is None else max_retries
    attempt = 0
    while True:
        started = False
        try:
            async for chunk in model.astream(list(messages)):
                started = True
                yield _piece(chunk)
            return
        except LlmError:
            raise
        except Exception as exc:
            info = classify(exc)
            if started:  # part of the answer is already out; it cannot be restarted cleanly
                raise _to_error(exc, info) from exc
            attempt += 1
            if info.retry_after is not None:
                delay = info.retry_after + 1.0
            else:
                delay = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            delay *= _jitter(1.0, 1.1)
            retriable = info.kind in ("rate_limit", "transient")
            if not retriable or attempt > limit or delay > settings.interactive_max_wait_seconds:
                raise _to_error(exc, info) from exc
            logger.warning(
                "llm call will be retried",
                extra={
                    "kind": info.kind,
                    "code": info.code,
                    "attempt": attempt,
                    "wait_s": round(delay, 1),
                },
            )
            yield Wait(delay)
            await _sleep(delay)


BLOCKED_MESSAGE = "The AI service declined to process this content."
EMPTY_MESSAGE = "The AI service returned no answer. Please try again."


async def complete(
    messages: Sequence[BaseMessage],
    kind: Kind = "analysis",
    on_wait: Callable[[float], None] | None = None,
    max_retries: int | None = None,
) -> tuple[str, dict[str, int]]:
    """The whole reply of one call: ``(text, usage)``. For work whose output is read as a unit
    (JSON, a summary) rather than streamed to a person.

    A rate limit is retried briefly exactly like ``stream`` does (``on_wait`` is told how long the
    wait is, so a screen can say so). A safety block, a reply cut off by a stop reason other than
    the normal ones, or an empty reply raises ``LlmError``: a blank result is never returned.
    """
    parts: list[str] = []
    usage: dict[str, int] = {}
    finish: str | None = None
    async for item in stream(messages, kind, max_retries):
        if isinstance(item, Wait):
            if on_wait is not None:
                on_wait(item.seconds)
            continue
        if item.blocked:
            raise LlmError("blocked", BLOCKED_MESSAGE)
        parts.append(item.text)
        finish = item.finish_reason or finish
        for key, value in item.usage.items():
            usage[key] = usage.get(key, 0) + value
    if finish not in (None, "STOP", "MAX_TOKENS"):  # SAFETY, RECITATION, BLOCKLIST, ...
        raise LlmError("blocked", BLOCKED_MESSAGE)
    text = "".join(parts)
    if not text.strip():
        raise LlmError("empty", EMPTY_MESSAGE)
    if finish == "MAX_TOKENS":
        logger.warning("analysis reply hit the output limit", extra={"kind": kind})
    return text, usage
