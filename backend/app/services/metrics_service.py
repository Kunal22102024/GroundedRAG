"""Measurements behind the evaluation dashboard, and the "often asked" suggestions built on them.

Every chat turn and every analysis run records exactly one ``query_events`` row, whatever its
outcome: answered, refused or failed. Recording never fails the request it measures: an error is
logged by class name and dropped. The dashboard reads the rows back in aggregate (admins only);
``popular_questions`` turns successful questions asked by several people into suggestions.

Definitions (TDD 17):

* **abstention rate**: refused / (answered + refused); a refusal is a guard refusal (no model call:
  ``below_threshold``, ``no_candidates``) or a model refusal (the guard passed, the model said the
  documents do not contain the answer);
* **citation precision**: valid citations / (valid + invalid markers removed);
* **failure rate**: failed / all.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Float, and_, delete, func, or_, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import (
    DocumentStatus,
    EventKind,
    EventOutcome,
    QuestionSource,
    allowed_access_levels,
)
from app.db.database import get_sessionmaker
from app.db.models import Feedback, QueryEvent, User

logger = logging.getLogger(__name__)

ANALYSIS_KINDS = (
    EventKind.SUMMARY.value,
    EventKind.EXTRACT.value,
    EventKind.COMPARE.value,
    EventKind.SUGGEST.value,
)
GUARD_REFUSALS = ("below_threshold", "no_candidates")
LATENCY_STAGES = ("rewrite_ms", "embed_ms", "dense_ms", "keyword_ms", "ttft_ms", "generate_ms")
HISTOGRAM_BINS = 20  # best dense similarity, 0.05 wide
QUESTION_MAX_CHARS_STORED = 2000
_COLUMNS = {column.name for column in QueryEvent.__table__.columns}


# --- Recording -------------------------------------------------------------------------------------


def new_event(kind: EventKind, user_id: uuid.UUID | None, **fields: Any) -> dict[str, Any]:
    """A measurement being filled in as a request runs; ``record`` stores it."""
    return {
        "kind": kind.value,
        "user_id": user_id,
        "outcome": EventOutcome.ANSWERED.value,
        "model": get_settings().llm_model,
        "timings": {},
        **fields,
    }


def add_usage(event: dict[str, Any], usage: dict[str, int] | None) -> None:
    """Add a model call's token counts to the event."""
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        event[key] = int(event.get(key) or 0) + int((usage or {}).get(key) or 0)


def add_retrieval(event: dict[str, Any], similarities: Sequence[float | None]) -> None:
    """Best and mean dense similarity of the retrieved candidates (retrieval relevance)."""
    known = [s for s in similarities if s is not None]
    event["retrieved"] = len(similarities)
    event["best_similarity"] = round(max(known), 4) if known else None
    event["mean_similarity"] = round(sum(known) / len(known), 4) if known else None


def record(event: dict[str, Any]) -> None:
    """Store one measurement. Never raises: telemetry must not break what it measures."""
    try:
        row = {key: value for key, value in event.items() if key in _COLUMNS}
        question = row.get("question")
        if isinstance(question, str):
            row["question"] = question[:QUESTION_MAX_CHARS_STORED]
        with get_sessionmaker()() as db:
            db.add(QueryEvent(**row))
            db.commit()
    except Exception as exc:  # noqa: BLE001 - deliberately swallowed, see the docstring
        logger.warning("query event not recorded", extra={"error": type(exc).__name__})


def prune(db: Session) -> int:
    """Delete measurements older than ``METRICS_RETENTION_DAYS``. Returns the count."""
    cutoff = datetime.now(UTC) - timedelta(days=get_settings().metrics_retention_days)
    deleted = db.execute(delete(QueryEvent).where(QueryEvent.created_at < cutoff)).rowcount or 0
    db.commit()
    if deleted:
        logger.info("old query events pruned", extra={"deleted": deleted})
    return int(deleted)


# --- Dashboard -------------------------------------------------------------------------------------


def _ratio(part: float | int | None, whole: float | int | None) -> float | None:
    return round(float(part or 0) / float(whole), 4) if whole else None


def _round(value: Any, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def _scope(kind: str) -> Any:
    if kind == "chat":
        return QueryEvent.kind == EventKind.CHAT.value
    if kind == "analysis":
        return QueryEvent.kind.in_(ANALYSIS_KINDS)
    return True


def _usd(input_tokens: int, output_tokens: int) -> float | None:
    settings = get_settings()
    if not (settings.llm_input_usd_per_mtok or settings.llm_output_usd_per_mtok):
        return None
    return round(
        input_tokens / 1e6 * settings.llm_input_usd_per_mtok
        + output_tokens / 1e6 * settings.llm_output_usd_per_mtok,
        4,
    )


def summary(db: Session, days: int, kind: str = "all") -> dict[str, Any]:
    """Every dashboard figure for the last ``days`` days, ``kind`` being all, chat or analysis."""
    settings = get_settings()
    since = datetime.now(UTC) - timedelta(days=days)
    window = and_(QueryEvent.created_at >= since, _scope(kind))
    chat = and_(QueryEvent.created_at >= since, QueryEvent.kind == EventKind.CHAT.value)
    e = QueryEvent
    answered = e.outcome == EventOutcome.ANSWERED.value
    refused = e.outcome == EventOutcome.REFUSED.value
    failed = e.outcome == EventOutcome.FAILED.value

    def count(condition: Any = True) -> Any:
        return func.count().filter(condition)

    totals = db.execute(
        select(count(), count(answered), count(refused), count(failed)).where(window)
    ).one()
    total, n_answered, n_refused, n_failed = (int(v or 0) for v in totals)

    # Retrieval relevance and abstention are about questions, so they read chat turns only.
    relevance = db.execute(
        select(
            func.avg(e.best_similarity).filter(answered),
            func.percentile_cont(0.5).within_group(e.best_similarity).filter(answered),
            func.avg(e.best_similarity).filter(refused),
            func.avg(e.mean_similarity).filter(answered),
            count(and_(refused, e.guard_reason.in_(GUARD_REFUSALS))),
            count(and_(refused, e.guard_reason.notin_(GUARD_REFUSALS))),
            count(e.outcome != EventOutcome.FAILED.value),
            count(e.guard_reason == "lexical_override"),
        ).where(chat)
    ).one()
    reasons = dict(
        db.execute(
            select(e.guard_reason, func.count())
            .where(chat, e.guard_reason.is_not(None))
            .group_by(e.guard_reason)
        ).all()
    )
    bin_index = func.least(func.floor(e.best_similarity * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)
    bins = dict(
        db.execute(
            select(bin_index, func.count())
            .where(chat, e.best_similarity.is_not(None))
            .group_by(bin_index)
        ).all()
    )
    histogram = [
        {"from": round(i / HISTOGRAM_BINS, 2), "count": int(bins.get(i, 0))}
        for i in range(HISTOGRAM_BINS)
    ]

    grounded = db.execute(
        select(
            count(e.cited > 0),
            func.sum(e.cited),
            func.sum(e.citations_removed),
            count(e.uncited),
            count(e.retried),
            count(e.partial),
            func.avg(e.cited),
            func.avg(e.cited_documents),
            count(e.conflicts > 0),
        ).where(chat, answered)
    ).one()
    chat_answered = db.scalar(select(count(answered)).where(chat)) or 0
    votes = dict(
        db.execute(
            select(Feedback.rating, func.count())
            .where(Feedback.created_at >= since)
            .group_by(Feedback.rating)
        ).all()
    )

    stage_columns = []
    for stage in LATENCY_STAGES:
        value = e.timings[stage].astext.cast(Float)
        stage_columns += [
            func.percentile_cont(0.5).within_group(value),
            func.percentile_cont(0.95).within_group(value),
        ]
    latency_row = db.execute(
        select(
            func.percentile_cont(0.5).within_group(e.total_ms),
            func.percentile_cont(0.95).within_group(e.total_ms),
            func.max(e.total_ms),
            *stage_columns,
        ).where(window, e.outcome != EventOutcome.FAILED.value)
    ).one()
    stages = [
        {
            "stage": stage.removesuffix("_ms"),
            "p50": _round(latency_row[3 + 2 * i]),
            "p95": _round(latency_row[4 + 2 * i]),
        }
        for i, stage in enumerate(LATENCY_STAGES)
        if latency_row[3 + 2 * i] is not None
    ]

    by_kind = [
        {
            "kind": row.kind,
            "events": int(row.events),
            "input_tokens": int(row.input or 0),
            "output_tokens": int(row.output or 0),
            "usd": _usd(int(row.input or 0), int(row.output or 0)),
        }
        for row in db.execute(
            select(
                e.kind,
                func.count().label("events"),
                func.sum(e.input_tokens).label("input"),
                func.sum(e.output_tokens).label("output"),
            )
            .where(window)
            .group_by(e.kind)
            .order_by(e.kind)
        )
    ]
    input_total = sum(k["input_tokens"] for k in by_kind)
    output_total = sum(k["output_tokens"] for k in by_kind)

    day = func.date_trunc("day", e.created_at)
    daily = [
        {
            "day": row.day.date().isoformat(),
            "queries": int(row.queries),
            "refused": int(row.refused or 0),
            "failed": int(row.failed or 0),
            "abstention_rate": _ratio(row.refused, (row.answered or 0) + (row.refused or 0)),
            "failure_rate": _ratio(row.failed, row.queries),
            "p95_ms": _round(row.p95),
            "input_tokens": int(row.input or 0),
            "output_tokens": int(row.output or 0),
        }
        for row in db.execute(
            select(
                day.label("day"),
                func.count().label("queries"),
                count(answered).label("answered"),
                count(refused).label("refused"),
                count(failed).label("failed"),
                func.percentile_cont(0.95).within_group(e.total_ms).label("p95"),
                func.sum(e.input_tokens).label("input"),
                func.sum(e.output_tokens).label("output"),
            )
            .where(window)
            .group_by(day)
            .order_by(day)
        )
    ]

    suggestion = db.execute(
        select(
            count(),
            count(e.source == QuestionSource.SUGGESTION.value),
            count(e.source == QuestionSource.FOLLOWUP.value),
            count(e.suggestions > 0),
        ).where(chat)
    ).one()

    return {
        "days": days,
        "kind": kind,
        "threshold": settings.relevance_threshold,
        "totals": {
            "queries": total,
            "answered": n_answered,
            "refused": n_refused,
            "failed": n_failed,
            "abstention_rate": _ratio(n_refused, n_answered + n_refused),
            "failure_rate": _ratio(n_failed, total),
        },
        "relevance": {
            "best_similarity_answered_mean": _round(relevance[0], 4),
            "best_similarity_answered_p50": _round(relevance[1], 4),
            "best_similarity_refused_mean": _round(relevance[2], 4),
            "mean_similarity_answered": _round(relevance[3], 4),
            "guard_reasons": {str(k): int(v) for k, v in reasons.items()},
            "lexical_override_rate": _ratio(relevance[7], relevance[6]),
            "histogram": histogram,
        },
        "abstention": {
            "guard_refusals": int(relevance[4] or 0),
            "model_refusals": int(relevance[5] or 0),
            "rate": _ratio(
                (relevance[4] or 0) + (relevance[5] or 0),
                chat_answered + (relevance[4] or 0) + (relevance[5] or 0),
            ),
        },
        "groundedness": {
            "answers": int(chat_answered),
            "with_citations_rate": _ratio(grounded[0], chat_answered),
            "citation_precision": _ratio(grounded[1], (grounded[1] or 0) + (grounded[2] or 0)),
            "invalid_citations_removed": int(grounded[2] or 0),
            "uncited_rate": _ratio(grounded[3], chat_answered),
            "retried_rate": _ratio(grounded[4], chat_answered),
            "truncated_rate": _ratio(grounded[5], chat_answered),
            "avg_citations": _round(grounded[6], 2),
            "avg_documents": _round(grounded[7], 2),
            "conflict_rate": _ratio(grounded[8], chat_answered),
            "feedback_up": int(votes.get("up", 0)),
            "feedback_down": int(votes.get("down", 0)),
        },
        "latency": {
            "p50_ms": _round(latency_row[0]),
            "p95_ms": _round(latency_row[1]),
            "max_ms": _round(latency_row[2]),
            "stages": stages,
        },
        "cost": {
            "input_tokens": input_total,
            "output_tokens": output_total,
            "usd": _usd(input_total, output_total),
            "priced": _usd(1, 1) is not None,
            "by_kind": by_kind,
        },
        "suggestions": {
            "chat_questions": int(suggestion[0] or 0),
            "from_suggestion": int(suggestion[1] or 0),
            "from_followup": int(suggestion[2] or 0),
            "answers_with_followups": int(suggestion[3] or 0),
            "usage_rate": _ratio((suggestion[1] or 0) + (suggestion[2] or 0), suggestion[0]),
            "followup_click_rate": _ratio(suggestion[2], suggestion[3]),
        },
        "daily": daily,
    }


def flagged_condition() -> Any:
    """An answer whose citations had to be repaired, or that was cut short."""
    return and_(
        QueryEvent.outcome == EventOutcome.ANSWERED.value,
        or_(QueryEvent.citations_removed > 0, QueryEvent.uncited, QueryEvent.partial),
    )


def list_events(
    db: Session,
    outcome: str,
    kind: str = "all",
    limit: int = 50,
    before: datetime | None = None,
) -> list[dict[str, Any]]:
    """Recent problem queries (failed, refused or flagged), newest first."""
    condition = (
        flagged_condition()
        if outcome == "flagged"
        else QueryEvent.outcome == outcome  # failed/refused
    )
    query = (
        select(QueryEvent, User.email)
        .outerjoin(User, User.id == QueryEvent.user_id)
        .where(condition, _scope(kind))
        .order_by(QueryEvent.created_at.desc())
        .limit(limit)
    )
    if before is not None:
        query = query.where(QueryEvent.created_at < before)
    return [
        {
            "id": event.id,
            "created_at": event.created_at,
            "kind": event.kind,
            "outcome": event.outcome,
            "source": event.source,
            "user_email": email,
            "question": event.question,
            "error_code": event.error_code,
            "guard_reason": event.guard_reason,
            "best_similarity": event.best_similarity,
            "cited": event.cited,
            "citations_removed": event.citations_removed,
            "uncited": event.uncited,
            "retried": event.retried,
            "partial": event.partial,
            "total_ms": event.total_ms,
            "total_tokens": event.total_tokens,
            "conversation_id": event.conversation_id,
            "document_ids": event.document_ids,
        }
        for event, email in db.execute(query)
    ]


# --- "Often asked" suggestions -----------------------------------------------------------------------

_POPULAR_SQL = text(
    """
    WITH ok AS (
        SELECT e.user_id,
               e.question,
               e.created_at,
               regexp_replace(lower(regexp_replace(btrim(e.question), '\\s+', ' ', 'g')),
                              '[?.!\\s]+$', '') AS norm,
               (SELECT count(*) FROM feedback f
                 WHERE f.message_id = e.message_id AND f.rating = 'up') AS ups
          FROM query_events e
         WHERE e.kind = 'chat'
           AND e.outcome = 'answered'
           AND e.cited > 0
           AND e.citations_removed = 0
           AND NOT e.uncited
           AND NOT e.partial
           AND e.user_id IS NOT NULL
           AND e.question IS NOT NULL
           AND e.cited_document_ids IS NOT NULL
           AND jsonb_array_length(e.cited_document_ids) > 0
           AND e.created_at >= :since
           -- every cited document still exists, is searchable and is readable by this reader
           AND NOT EXISTS (
                 SELECT 1
                   FROM jsonb_array_elements_text(e.cited_document_ids) AS cited(id)
                   LEFT JOIN documents d
                          ON d.id::text = cited.id
                         AND d.status = :indexed
                         AND d.access_level = ANY(:levels)
                  WHERE d.id IS NULL)
           AND (cast(:selected AS text[]) IS NULL OR EXISTS (
                 SELECT 1 FROM jsonb_array_elements_text(e.cited_document_ids) AS cited(id)
                  WHERE cited.id = ANY(cast(:selected AS text[]))))
    )
    SELECT mode() WITHIN GROUP (ORDER BY question) AS question,
           count(DISTINCT user_id) AS users,
           sum(ups) AS ups,
           max(created_at) AS last_asked
      FROM ok
     WHERE norm <> ''
     GROUP BY norm
    HAVING count(DISTINCT user_id) >= :min_users
     ORDER BY users DESC, ups DESC, last_asked DESC
     LIMIT :limit
    """
)


def popular_questions(
    db: Session, role: str, document_ids: Sequence[str] | None, limit: int = 6
) -> list[dict[str, Any]]:
    """Questions several people asked and got a cleanly cited answer to, that this reader may see.

    A question is offered only when at least ``POPULAR_MIN_USERS`` different users asked it (so one
    person's question is never shown to others) and every document its answer cited is indexed and
    readable at this reader's access level. With ``document_ids`` it must cite one of them.
    """
    settings = get_settings()
    levels = [level.value for level in allowed_access_levels(role)]
    if not levels:
        return []
    rows = db.execute(
        _POPULAR_SQL,
        {
            "since": datetime.now(UTC) - timedelta(days=settings.metrics_retention_days),
            "indexed": DocumentStatus.INDEXED.value,
            "levels": levels,
            "selected": list(document_ids) if document_ids else None,
            "min_users": settings.popular_min_users,
            "limit": limit,
        },
    ).all()
    return [{"question": row.question.strip(), "asked_by": int(row.users)} for row in rows]
