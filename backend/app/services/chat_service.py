"""One chat turn (CLAUDE.md 4.2): rewrite -> filter -> retrieve -> guard -> generate -> verify -> persist.

``prepare_chat`` runs first and can fail with an ordinary HTTP error (unknown conversation).
``run_chat`` then yields the SSE events: ``status``, ``token``, ``citations``, ``done`` and
``error``. Whatever goes wrong while streaming becomes an ``error`` event with a message that is
safe to show; nothing here logs questions, answers or document text (rule 21).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import numpy as np
from sqlalchemy import func, select

from app.config import get_settings
from app.constants import (
    CONVERSATION_TITLE_MAX_CHARS,
    REFUSAL_MESSAGE,
    REWRITE_HISTORY_MESSAGES,
    DocumentStatus,
    EventKind,
    EventOutcome,
    MessageRole,
    QuestionSource,
)
from app.core import (
    analysis,
    answer_cache,
    embeddings,
    generation,
    graph_retrieval,
    guard,
    llm,
    query_rewrite,
    retrieval,
)
from app.core.filters import FilterRequest, SearchFilter, build_filter
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Conversation, Document, Message
from app.services import metrics_service

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Something went wrong while answering. Please try again."
CONVERSATION_DELETED_MESSAGE = "This conversation was deleted while the answer was being written."


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
    source: str = QuestionSource.TYPED.value  # typed, or picked from a suggestion (dashboard)


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
    source: str = QuestionSource.TYPED.value,
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
        source=source,
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
    conflicts: list[dict[str, Any]] | None = None,
    suggestions: list[dict[str, Any]] | None = None,
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
            if conversation is None:  # deleted while the answer was being written
                raise ConversationNotFound
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
            conflicts=conflicts or None,
            suggestions=suggestions or None,
            guard_score=guard_score,
            timings=timings,
            flagged=flagged,
            created_at=now + timedelta(microseconds=1),
        )
        db.add(assistant)
        db.commit()
        return assistant.id


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _cache_entry_valid(entry: answer_cache.CacheEntry, search_filter: SearchFilter) -> bool:
    """A cached answer is served only if every passage it cites is still there: its document is
    indexed and the passage passes this reader's filter. Invalidation should already have dropped
    such an entry; this is the check that does not rely on it (rules 6 and 24)."""
    wanted = set(entry.cited_chunk_ids)
    if not wanted:
        return False
    with get_sessionmaker()() as db:
        found = db.scalar(
            select(func.count(func.distinct(Chunk.id)))
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.id.in_(wanted),
                Document.status == DocumentStatus.INDEXED.value,
                *search_filter.sql_conditions(Chunk),
            )
        )
    return found == len(wanted)


async def _serve_cached(
    plan: ChatPlan,
    event: dict[str, Any],
    hit: answer_cache.CacheHit,
    rewritten: bool,
    started: float,
) -> AsyncIterator[SseEvent]:
    """Answer from a saved turn: no search, no model call. The answer is stored as a normal
    message and measured like any other turn, so history and the dashboard stay complete."""
    entry = hit.entry
    timings: dict[str, Any] = event["timings"]
    citations = copy.deepcopy(entry.citations)
    conflicts = copy.deepcopy(entry.conflicts)
    suggestions = copy.deepcopy(entry.suggestions)

    yield SseEvent("status", {"stage": "cache_hit"})
    yield SseEvent("token", {"text": entry.answer})

    timings.update(
        cache="hit", cache_similarity=round(hit.similarity, 4), saved_tokens=entry.saved_tokens
    )
    timings["total_ms"] = _ms(started)
    event.update(
        guard_reason=entry.guard.get("reason"),
        best_similarity=entry.guard.get("score"),
        cited=len(citations),
        cited_documents=len(entry.cited_document_ids),
        cited_document_ids=sorted(entry.cited_document_ids),
        conflicts=len(conflicts),
        suggestions=len(suggestions),
        total_ms=timings["total_ms"],
    )
    message_id = await asyncio.to_thread(
        _persist,
        plan,
        entry.answer,
        citations,
        entry.guard.get("score"),
        {**timings, "usage": {}},
        False,
        conflicts,
        suggestions,
    )
    answer_cache.get_cache().register_message(hit.entry_id, str(message_id))
    event["message_id"] = message_id
    await asyncio.to_thread(metrics_service.record, event)
    logger.info(
        "chat turn served from cache",
        extra={"similarity": round(hit.similarity, 4), "total_ms": timings["total_ms"]},
    )

    yield SseEvent("citations", {"citations": citations})
    if conflicts:
        yield SseEvent("conflicts", {"conflicts": conflicts})
    if suggestions:
        yield SseEvent("suggestions", {"suggestions": public_suggestions(suggestions)})
    yield SseEvent(
        "done",
        {
            "conversation_id": str(plan.conversation_id),
            "message_id": str(message_id),
            "answer": entry.answer,
            "refused": False,
            "guard": entry.guard,
            "timings": timings,
            "usage": {},
            "flags": {
                "partial": False,
                "citations_removed": 0,
                "uncited": False,
                "retried": False,
                "rewritten": rewritten,
            },
            "cached": True,
        },
    )


async def _turn(plan: ChatPlan, event: dict[str, Any]) -> AsyncIterator[SseEvent]:
    """The turn itself. ``event`` (the dashboard measurement) is filled in as stages complete,
    so a turn that fails half-way still reports what it got to (``run_chat`` records it)."""
    settings = get_settings()
    started = time.perf_counter()
    timings: dict[str, Any] = event["timings"]

    yield SseEvent("status", {"stage": "searching"})
    question = plan.question
    if plan.history:
        yield SseEvent("status", {"stage": "rewriting"})
        step = time.perf_counter()
        question = await query_rewrite.rewrite_question(plan.question, plan.history)
        timings["rewrite_ms"] = _ms(step)

    rewritten = question != plan.question
    event["rewritten"] = rewritten

    # Saved answers (core/answer_cache.py). Not for a follow-up whose rewrite changed nothing: it
    # may still lean on the conversation ("and the second one?"), so its words alone say too little.
    cache = answer_cache.get_cache()
    use_cache = (
        settings.semantic_cache
        and not plan.search_filter.matches_nothing
        and (rewritten or not plan.history)
    )
    vector: list[float] | None = None
    scope = ""
    generation_seen = 0
    if use_cache:
        generation_seen = cache.generation  # read before searching: see AnswerCache.store
        step = time.perf_counter()
        vector = await asyncio.to_thread(embeddings.embed_query, question, interactive=True)
        timings["embed_ms"] = _ms(step)
        scope = answer_cache.scope_key(plan.search_filter)
        hit = cache.lookup(scope, vector)
        if hit is not None:
            if await asyncio.to_thread(_cache_entry_valid, hit.entry, plan.search_filter):
                async for item in _serve_cached(plan, event, hit, rewritten, started):
                    yield item
                return
            cache.evict(hit.entry_id)  # a passage it cites is gone or no longer readable
            generation_seen = cache.generation

    result = await asyncio.to_thread(retrieval.retrieve, question, plan.search_filter, vector)
    timings.update(result.timings)
    metrics_service.add_retrieval(event, [c.similarity for c in result.candidates])

    decision = guard.evaluate(
        question,
        result.best_similarity,
        result.top_keyword_text,
        settings.relevance_threshold,
        settings.lexical_override_margin,
    )
    event["guard_reason"] = decision.reason
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
    extra_usage: dict[str, int] = {}  # the conflict check and the follow-up suggestions
    citations: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    extras_failed = False  # an extra that failed would be missing from a saved answer for good

    if not decision.passed:
        # Below the threshold: the LLM is never called (CLAUDE.md rule 8).
        answer = REFUSAL_MESSAGE
        yield SseEvent("token", {"text": answer})
        refused = True
    else:
        # The guard has passed on the seed passages. Now the knowledge graph may add passages
        # connected to them (other documents included); the guard never sees those, and a failure
        # here costs nothing but the extra context.
        candidates = result.candidates
        if settings.graph_retrieval_active:
            try:
                expansion = await asyncio.to_thread(
                    graph_retrieval.expand, result.candidates, plan.search_filter
                )
                candidates = [*result.candidates, *expansion.candidates]
                timings.update(graph_ms=expansion.ms, graph_added=len(expansion.candidates))
            except Exception as exc:
                logger.warning("graph expansion failed", extra={"error": type(exc).__name__})
        yield SseEvent("status", {"stage": "generating"})
        final: generation.AnswerResult | None = None
        async for item in generation.generate_answer(question, candidates):
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
        citations = [retrieval.citation_dict(n, candidates[n - 1]) for n in final.cited]
        flags.update(
            partial=final.truncated,
            citations_removed=final.invalid_removed,
            uncited=final.uncited,
            retried=final.retried,
        )
        timings["ttft_ms"] = final.ttft_ms
        timings["generate_ms"] = final.generate_ms
        metrics_service.add_usage(event, usage)
        event.update(
            cited=len(citations),
            cited_documents=len({c["document_id"] for c in citations}),
            cited_document_ids=sorted({c["document_id"] for c in citations}),
            citations_removed=final.invalid_removed,
            uncited=final.uncited,
            retried=final.retried,
            partial=final.truncated,
        )

        # Two extras after a complete answer, run side by side so the wait is the longer of the
        # two: a conflict check when the answer draws on two or more documents (single-document
        # answers never pay for it), and follow-up questions. Either may fail without effect.
        passages = [analysis.Passage(n, candidates[n - 1]) for n in final.cited]
        check_conflicts = (
            settings.conflict_detection
            and not refused
            and len({p.candidate.document_id for p in passages}) >= 2
        )
        offer_followups = settings.followup_suggestions and not refused

        async def conflict_check() -> None:
            nonlocal extras_failed
            step = time.perf_counter()
            try:
                found, used = await analysis.detect_conflicts(question, passages)
                conflicts.extend(found)
                metrics_service.add_usage(extra_usage, used)
            except Exception as exc:  # the answer is complete; the check is an extra
                extras_failed = True
                logger.warning("conflict check failed", extra={"error": type(exc).__name__})
            timings["conflict_ms"] = _ms(step)

        async def followups() -> None:
            nonlocal extras_failed
            step = time.perf_counter()
            try:
                found, used = await analysis.suggest_followups(question, answer, candidates)
                suggestions.extend(found)
                metrics_service.add_usage(extra_usage, used)
            except Exception as exc:  # an extra, like the conflict check
                extras_failed = True
                logger.warning("follow-up suggestions failed", extra={"error": type(exc).__name__})
            timings["followup_ms"] = _ms(step)

        extras = [
            job()
            for job, wanted in ((conflict_check, check_conflicts), (followups, offer_followups))
            if wanted
        ]
        if extras:
            yield SseEvent(
                "status", {"stage": "checking_sources" if check_conflicts else "suggesting"}
            )
            await asyncio.gather(*extras)

    if refused:
        event["outcome"] = EventOutcome.REFUSED.value
    metrics_service.add_usage(event, extra_usage)
    flagged = bool(flags["partial"] or flags["citations_removed"] or flags["uncited"])
    timings["total_ms"] = _ms(started)
    stored_timings = {**timings, "usage": usage}
    if extra_usage:
        stored_timings["extra_usage"] = extra_usage
    message_id = await asyncio.to_thread(
        _persist,
        plan,
        answer,
        citations,
        decision.score,
        stored_timings,
        flagged,
        conflicts,
        suggestions,
    )
    event.update(
        message_id=message_id,
        conflicts=len(conflicts),
        suggestions=len(suggestions),
        total_ms=timings["total_ms"],
    )
    await asyncio.to_thread(metrics_service.record, event)
    if (
        use_cache
        and vector is not None
        and decision.passed
        and not refused
        and not flagged
        and not extras_failed
        and citations
    ):
        entry_id = cache.store(
            answer_cache.CacheEntry(
                scope=scope,
                vector=np.asarray(vector, dtype=np.float32),
                search_filter=plan.search_filter,
                answer=answer,
                citations=copy.deepcopy(citations),
                conflicts=copy.deepcopy(conflicts),
                suggestions=copy.deepcopy(suggestions),
                guard=guard_info,
                cited_chunk_ids=tuple(c["chunk_id"] for c in citations),
                cited_document_ids=frozenset(c["document_id"] for c in citations),
                saved_tokens=int(usage.get("total_tokens") or 0)
                + int(extra_usage.get("total_tokens") or 0),
            ),
            generation_seen,
        )
        if entry_id is not None:
            cache.register_message(entry_id, str(message_id))
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
            "conflicts": len(conflicts),
        },
    )
    yield SseEvent("citations", {"citations": citations})
    if conflicts:
        yield SseEvent("conflicts", {"conflicts": conflicts})
    if suggestions:
        yield SseEvent("suggestions", {"suggestions": public_suggestions(suggestions)})
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
            "cached": False,
        },
    )


def error_event(exc: BaseException) -> SseEvent:
    """Map a failure to a user-safe ``error`` event."""
    code, message, retry_after = "internal", INTERNAL_ERROR_MESSAGE, None
    if isinstance(exc, llm.LlmError):
        code, message, retry_after = exc.code, exc.message, exc.retry_after
    elif isinstance(exc, analysis.AnalysisError):
        code, message = exc.code, exc.message
    elif isinstance(exc, ConversationNotFound):
        code, message = "conversation_deleted", CONVERSATION_DELETED_MESSAGE
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


def public_suggestions(suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the screen gets of a follow-up suggestion: its type, the question and its document."""
    return [
        {"type": s["type"], "question": s["question"], "document_id": s.get("document_id")}
        for s in suggestions
    ]


async def run_chat(plan: ChatPlan) -> AsyncIterator[SseEvent]:
    """Run one turn; any failure ends the stream with an ``error`` event instead of an exception.

    Every turn is measured once (``query_events``): answered and refused turns by ``_turn``, a
    failed one here, with the stages it completed. A client that went away is not a failure.
    """
    event = metrics_service.new_event(
        EventKind.CHAT,
        plan.user_id,
        question=plan.question,
        source=plan.source,
        conversation_id=plan.conversation_id,
    )
    started = time.perf_counter()
    try:
        async for item in _turn(plan, event):
            yield item
    except Exception as exc:  # CancelledError (client went away) is not an Exception: it propagates
        failure = error_event(exc)
        event.update(
            outcome=EventOutcome.FAILED.value,
            error_code=failure.data["code"],
            total_ms=_ms(started),
        )
        await asyncio.to_thread(metrics_service.record, event)
        yield failure
