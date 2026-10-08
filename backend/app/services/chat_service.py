"""One chat turn (CLAUDE.md 4.2): rewrite -> filter -> retrieve -> guard -> generate -> verify -> persist.

``prepare_chat`` runs first and can fail with an ordinary HTTP error (unknown conversation).
``run_chat`` then yields the SSE events: ``status``, ``token``, ``citations``, ``done`` and
``error``. Whatever goes wrong while streaming becomes an ``error`` event with a message that is
safe to show; nothing here logs questions, answers or document text (rule 21).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select

from app.config import get_settings
from app.constants import (
    CONVERSATION_TITLE_MAX_CHARS,
    REFUSAL_MESSAGE,
    REWRITE_HISTORY_MESSAGES,
    MessageRole,
)
from app.core import embeddings, generation, guard, llm, query_rewrite, retrieval
from app.core.filters import FilterRequest, SearchFilter, build_filter
from app.db.database import get_sessionmaker
from app.db.models import Conversation, Message

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Something went wrong while answering. Please try again."


class ConversationNotFound(Exception):
    """The conversation does not exist or belongs to someone else."""


class ChatUser(Protocol):
    """The parts of the ``User`` model a chat turn needs."""

    id: uuid.UUID
    role: str


@dataclass(frozen=True)
class ChatPlan:
    """Everything resolved before streaming starts."""

    user_id: uuid.UUID
    question: str
    conversation_id: uuid.UUID
    new_conversation: bool
    history: list[tuple[str, str]]
    search_filter: SearchFilter


@dataclass(frozen=True)
class SseEvent:
    """One server-sent event: a name and a JSON-serializable payload."""

    event: str
    data: dict[str, Any]


def prepare_chat(
    user: ChatUser,
    question: str,
    conversation_id: uuid.UUID | None,
    filters: FilterRequest | None,
) -> ChatPlan:
    """Resolve the conversation (owner only), its recent history and the search filter."""
    history: list[tuple[str, str]] = []
    new_conversation = conversation_id is None
    resolved_id = conversation_id or uuid.uuid4()
    if conversation_id is not None:
        with get_sessionmaker()() as db:
            owned = db.scalar(
                select(Conversation.id).where(
                    Conversation.id == conversation_id, Conversation.user_id == user.id
                )
            )
            if owned is None:
                raise ConversationNotFound
            rows = db.execute(
                select(Message.role, Message.content)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(REWRITE_HISTORY_MESSAGES)
            ).all()
            history = [(row.role, row.content) for row in reversed(rows)]
    return ChatPlan(
        user_id=user.id,
        question=question.strip(),
        conversation_id=resolved_id,
        new_conversation=new_conversation,
        history=history,
        search_filter=build_filter(user, filters),
    )


def _title(question: str) -> str:
    return " ".join(question.split())[:CONVERSATION_TITLE_MAX_CHARS]


def _persist(
    plan: ChatPlan,
    answer: str,
    citations: list[dict[str, Any]],
    guard_score: float | None,
    timings: dict[str, Any],
    flagged: bool,
) -> uuid.UUID:
    """Store the question and the answer together; returns the assistant message id."""
    with get_sessionmaker()() as db:
        now = datetime.now(UTC)
        if plan.new_conversation:
            db.add(
                Conversation(
                    id=plan.conversation_id, user_id=plan.user_id, title=_title(plan.question)
                )
            )
            db.flush()
        else:
            conversation = db.get(Conversation, plan.conversation_id)
            if conversation is not None:
                conversation.updated_at = now
        # Explicit timestamps: both rows share one transaction, and `now()` would tie them.
        db.add(
            Message(
                conversation_id=plan.conversation_id,
                role=MessageRole.USER.value,
                content=plan.question,
                created_at=now,
            )
        )
        assistant = Message(
            conversation_id=plan.conversation_id,
            role=MessageRole.ASSISTANT.value,
            content=answer,
            citations=citations,
            guard_score=guard_score,
            timings=timings,
            flagged=flagged,
            created_at=now + timedelta(microseconds=1),
        )
        db.add(assistant)
        db.commit()
        return assistant.id


def _citation(number: int, candidate: retrieval.Candidate) -> dict[str, Any]:
    return {
        "n": number,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "document_name": candidate.document_name,
        "page_number": candidate.page_number,
        "page_end": candidate.page_end,
        "section_title": candidate.section_title,
        "text": candidate.text,
    }


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def _turn(plan: ChatPlan) -> AsyncIterator[SseEvent]:
    settings = get_settings()
    started = time.perf_counter()
    timings: dict[str, Any] = {}

    yield SseEvent("status", {"stage": "searching"})
    question = plan.question
    if plan.history:
        yield SseEvent("status", {"stage": "rewriting"})
        step = time.perf_counter()
        question = await query_rewrite.rewrite_question(plan.question, plan.history)
        timings["rewrite_ms"] = _ms(step)

    result = await asyncio.to_thread(retrieval.retrieve, question, plan.search_filter)
    timings.update(result.timings)

    decision = guard.evaluate(
        question, result.best_similarity, result.top_keyword_text, settings.relevance_threshold
    )
    guard_info = {
        "passed": decision.passed,
        "reason": decision.reason,
        "score": None if decision.score is None else round(decision.score, 4),
        "threshold": decision.threshold,
    }
    flags = {
        "partial": False,
        "citations_removed": 0,
        "uncited": False,
        "retried": False,
        "rewritten": question != plan.question,
    }
    usage: dict[str, int] = {}
    citations: list[dict[str, Any]] = []

    if not decision.passed:
        # Below the threshold: the LLM is never called (CLAUDE.md rule 8).
        answer = REFUSAL_MESSAGE
        yield SseEvent("token", {"text": answer})
        refused = True
    else:
        yield SseEvent("status", {"stage": "generating"})
        final: generation.AnswerResult | None = None
        async for item in generation.generate_answer(question, result.candidates):
            if isinstance(item, llm.Wait):
                yield SseEvent("status", {"stage": "rate_limited", "retry_in": round(item.seconds)})
            elif isinstance(item, generation.Restart):
                # The answer streamed so far is discarded and written again with stricter rules.
                yield SseEvent("status", {"stage": "retrying", "reset": True})
            elif isinstance(item, generation.Token):
                yield SseEvent("token", {"text": item.text})
            else:
                final = item
        if final is None:  # generate_answer always ends with a result; never fall through silently
            raise llm.LlmError("empty", generation.EMPTY_MESSAGE)
        answer, refused, usage = final.text, final.refused, final.usage
        citations = [_citation(n, result.candidates[n - 1]) for n in final.cited]
        flags.update(
            partial=final.truncated,
            citations_removed=final.invalid_removed,
            uncited=final.uncited,
            retried=final.retried,
        )
        timings["ttft_ms"] = final.ttft_ms
        timings["generate_ms"] = final.generate_ms

    flagged = bool(flags["partial"] or flags["citations_removed"] or flags["uncited"])
    timings["total_ms"] = _ms(started)
    stored_timings = {**timings, "usage": usage}
    message_id = await asyncio.to_thread(
        _persist, plan, answer, citations, decision.score, stored_timings, flagged
    )
    logger.info(
        "chat turn complete",
        extra={
            "refused": refused,
            "guard_reason": decision.reason,
            "guard_score": guard_info["score"],
            "citations": len(citations),
            "flagged": flagged,
            "total_ms": timings["total_ms"],
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        },
    )
    yield SseEvent("citations", {"citations": citations})
    yield SseEvent(
        "done",
        {
            "conversation_id": str(plan.conversation_id),
            "message_id": str(message_id),
            "answer": answer,
            "refused": refused,
            "guard": guard_info,
            "timings": timings,
            "usage": usage,
            "flags": flags,
        },
    )


def error_event(exc: BaseException) -> SseEvent:
    """Map a failure to a user-safe ``error`` event."""
    code, message, retry_after = "internal", INTERNAL_ERROR_MESSAGE, None
    if isinstance(exc, llm.LlmError):
        code, message, retry_after = exc.code, exc.message, exc.retry_after
    elif isinstance(exc, retrieval.IndexUnavailableError):
        code, message = "index_unavailable", exc.message
    elif isinstance(exc, embeddings.EmbeddingRateLimitError):
        code, message = "rate_limited", llm.RATE_LIMITED_MESSAGE
    elif isinstance(exc, embeddings.EmbeddingQuotaError):
        code, message = "quota_exhausted", str(exc)
    elif isinstance(exc, (embeddings.EmbeddingAuthError, embeddings.EmbeddingConfigError)):
        code, message = (
            "auth",
            "The AI service is not configured correctly. Please contact your administrator.",
        )
    elif isinstance(exc, embeddings.EmbeddingError):
        code, message = (
            "unavailable",
            "The AI service is temporarily unavailable. Please try again in a moment.",
        )
    else:
        logger.error("chat turn failed unexpectedly", extra={"error": type(exc).__name__})
        return SseEvent("error", {"code": code, "message": message, "retry_after": retry_after})
    logger.warning("chat turn failed", extra={"error": type(exc).__name__, "code": code})
    return SseEvent("error", {"code": code, "message": message, "retry_after": retry_after})


async def run_chat(plan: ChatPlan) -> AsyncIterator[SseEvent]:
    """Run one turn; any failure ends the stream with an ``error`` event instead of an exception."""
    try:
        async for event in _turn(plan):
            yield event
    except Exception as exc:  # CancelledError (client went away) is not an Exception: it propagates
        yield error_event(exc)
