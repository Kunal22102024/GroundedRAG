"""Analysis over cited sources: summaries, field extraction and conflict detection.

Everything here is grounded the way an answer is (CLAUDE.md rules 9 and 10):

* the model sees numbered ``<source>`` blocks framed as untrusted data and must cite them;
* every number it cites is checked against the sources it was given (``guard.filter_citations``):
  a number that maps to no source is removed, and an item left with no valid citation is dropped
  rather than shown as an unsupported claim;
* structured output is parsed and validated with Pydantic, retried once with a stricter reminder,
  and a second failure is an error, never a half-read result;
* a document longer than ``ANALYSIS_MAX_SOURCE_CHARS`` is read in batches whose sources keep their
  *global* numbers, so a citation in the final result still points at the right chunk.

No FastAPI and no database here: the service loads the chunks and stores the results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.core import guard, llm, prompts
from app.core.retrieval import Candidate, citation_dict

logger = logging.getLogger(__name__)

SUMMARY_KINDS = ("executive", "chapters", "faq", "glossary", "key_changes")
MAX_FAQ_ITEMS = 12
MAX_GLOSSARY_ITEMS = 40
MAX_FIELD_CHARS = 60
UNREADABLE_MESSAGE = "The AI service returned a result that could not be read. Please try again."

Numbered = tuple[int, Candidate]
OnWait = Callable[[float], None]
OnProgress = Callable[[str], None]


class AnalysisError(Exception):
    """The analysis cannot be done (as opposed to the model failing). ``message`` is user-safe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --- Output shapes (anything the model adds beyond these is ignored) --------------------------


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChapterItem(_Loose):
    title: str = ""
    summary: str = ""


class FaqItem(_Loose):
    question: str = ""
    answer: str = ""


class GlossaryItem(_Loose):
    term: str = ""
    definition: str = ""


class ChangeItem(_Loose):
    change: str = ""
    impact: str = ""


class ChaptersOut(_Loose):
    items: list[ChapterItem] = Field(default_factory=list)


class FaqOut(_Loose):
    items: list[FaqItem] = Field(default_factory=list)


class GlossaryOut(_Loose):
    items: list[GlossaryItem] = Field(default_factory=list)


class ChangesOut(_Loose):
    items: list[ChangeItem] = Field(default_factory=list)


class ExtractedValue(_Loose):
    value: str = ""
    source: int = 0
    quote: str = ""


class ExtractedField(_Loose):
    field: str = ""
    values: list[ExtractedValue] = Field(default_factory=list)


class ExtractionOut(_Loose):
    fields: list[ExtractedField] = Field(default_factory=list)


class ConflictClaim(_Loose):
    source: int = 0
    claim: str = ""


class ConflictItem(_Loose):
    topic: str = ""
    claims: list[ConflictClaim] = Field(default_factory=list)


class ConflictsOut(_Loose):
    conflicts: list[ConflictItem] = Field(default_factory=list)


class QuestionItem(_Loose):
    question: str = ""
    source: int = 0


class QuestionsOut(_Loose):
    questions: list[QuestionItem] = Field(default_factory=list)


class FollowupItem(_Loose):
    type: str = ""
    question: str = ""
    source: int = 0


class FollowupsOut(_Loose):
    suggestions: list[FollowupItem] = Field(default_factory=list)


# kind -> (output model, the field that must carry a citation, the other text fields)
_ITEM_KINDS: dict[str, tuple[type[BaseModel], str, tuple[str, ...]]] = {
    "chapters": (ChaptersOut, "summary", ("title",)),
    "faq": (FaqOut, "answer", ("question",)),
    "glossary": (GlossaryOut, "definition", ("term",)),
    "key_changes": (ChangesOut, "change", ("impact",)),
}
# The field that identifies an item, to drop repeats when batches are merged.
_LABEL = {"chapters": "title", "faq": "question", "glossary": "term", "key_changes": "change"}


# --- Reading the model's reply ----------------------------------------------------------------


def parse_json(text: str, model: type[BaseModel]) -> Any:
    """The reply as ``model``. Tolerates a Markdown fence and text around the JSON object.

    Raises ``ValueError`` (``ValidationError`` is one) when there is no valid object.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return model.model_validate(json.loads(cleaned[start : end + 1]))


def _add(total: dict[str, int], more: dict[str, int]) -> None:
    for key, value in more.items():
        total[key] = total.get(key, 0) + value


def _retries(patient: bool) -> int | None:
    """How often a busy model is retried: analysis runs wait longer than a chat turn; the extra
    conflict check (``patient=False``) gives up as quickly as chat does."""
    return get_settings().analysis_max_retries if patient else None


async def _run_json(
    messages: Sequence[BaseMessage],
    model: type[BaseModel],
    on_wait: OnWait | None,
    patient: bool = True,
    kind: llm.Kind = "analysis",
) -> tuple[Any, dict[str, int]]:
    """One model call whose reply must be valid ``model`` JSON; retried once with a reminder."""
    usage: dict[str, int] = {}
    attempt_messages = list(messages)
    for _ in range(2):
        text, used = await llm.complete(attempt_messages, kind, on_wait, _retries(patient))
        _add(usage, used)
        try:
            return parse_json(text, model), usage
        except ValueError:  # includes json.JSONDecodeError and pydantic.ValidationError
            attempt_messages = [*messages, HumanMessage(content=prompts.JSON_RETRY_REMINDER)]
    raise llm.LlmError("failed", UNREADABLE_MESSAGE)


# For sibling modules (core/graph.py) that run their own JSON tasks the same way.
run_json = _run_json
add_usage = _add


async def _cited_text(
    messages: Sequence[BaseMessage], source_count: int, on_wait: OnWait | None
) -> tuple[str, list[int], dict[str, int]]:
    """A text reply with its citations checked. A reply that cites nothing is retried once."""
    usage: dict[str, int] = {}
    attempt_messages = list(messages)
    for _ in range(2):
        text, used = await llm.complete(attempt_messages, "analysis", on_wait, _retries(True))
        _add(usage, used)
        clean, cited, _ = guard.filter_citations(text, source_count)
        if clean and cited:
            return clean, cited, usage
        attempt_messages = [*messages, HumanMessage(content=prompts.CITATION_RETRY_REMINDER)]
    raise llm.LlmError("failed", UNREADABLE_MESSAGE)


# --- Sources ------------------------------------------------------------------------------------


def make_batches(sources: Sequence[Candidate], limit: int, start: int = 1) -> list[list[Numbered]]:
    """Consecutive sources grouped so each group's text stays under ``limit`` characters.

    Numbers are global (``start``, ``start + 1``, ...), not per group: a citation produced from
    any group means the same chunk everywhere else.
    """
    batches: list[list[Numbered]] = []
    current: list[Numbered] = []
    size = 0
    for number, source in enumerate(sources, start=start):
        cost = len(source.text) + 160  # the <source> wrapper and its attributes
        if current and size + cost > limit:
            batches.append(current)
            current, size = [], 0
        current.append((number, source))
        size += cost
    if current:
        batches.append(current)
    return batches


def _citations(numbered: dict[int, Candidate], cited: set[int]) -> list[dict[str, Any]]:
    return [citation_dict(n, numbered[n]) for n in sorted(cited) if n in numbered]


# --- Summaries ----------------------------------------------------------------------------------


async def summarize(
    kind: str,
    sources: Sequence[Candidate],
    *,
    earlier: Sequence[Candidate] | None = None,
    on_wait: OnWait | None = None,
    on_progress: OnProgress | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """A summary of one kind: ``(content, usage)``.

    ``content`` is ``{"kind", "text" | "items", "citations"}``; the text of every item carries
    ``[n]`` markers that match ``citations``. ``earlier`` (the older version of the document) is
    only used by ``key_changes``, which then reports the differences between the two.
    """
    if kind not in SUMMARY_KINDS:
        raise AnalysisError("bad_kind", f"Unknown summary type '{kind[:30]}'.")
    if not sources:
        raise AnalysisError("empty", "This document has no indexed text to summarize.")
    limit = get_settings().analysis_max_source_chars
    progress = on_progress or (lambda _stage: None)

    if kind == "key_changes" and earlier:
        everything = [*sources, *earlier]
        if sum(len(s.text) + 160 for s in everything) > limit:
            raise AnalysisError(
                "too_large",
                "These two documents are too long to compare version against version. "
                "Summarize the changes of one document instead.",
            )
        numbered = dict(enumerate(everything, start=1))
        task = prompts.KEY_CHANGES_VERSIONS_TASK.format(
            current=sources[0].document_name, earlier=earlier[0].document_name
        )
        return await _item_kind(
            kind, [list(numbered.items())], numbered, len(everything), task, on_wait, progress
        )

    numbered = dict(enumerate(sources, start=1))
    batches = make_batches(sources, limit)
    if kind == "executive":
        return await _executive(batches, numbered, on_wait, progress)
    return await _item_kind(
        kind, batches, numbered, len(sources), prompts.SUMMARY_TASKS[kind], on_wait, progress
    )


async def _executive(
    batches: list[list[Numbered]],
    numbered: dict[int, Candidate],
    on_wait: OnWait | None,
    progress: OnProgress,
) -> tuple[dict[str, Any], dict[str, int]]:
    total = len(numbered)
    usage: dict[str, int] = {}
    if len(batches) == 1:
        messages = prompts.build_analysis_messages(
            prompts.SUMMARY_TASKS["executive"], prompts.format_sources(batches[0])
        )
        text, cited, used = await _cited_text(messages, total, on_wait)
        _add(usage, used)
    else:
        notes: list[str] = []
        for index, batch in enumerate(batches, start=1):
            progress(f"Reading part {index} of {len(batches)}")
            messages = prompts.build_analysis_messages(
                prompts.NOTES_TASK, prompts.format_sources(batch)
            )
            note, _, used = await _cited_text(messages, total, on_wait)
            _add(usage, used)
            notes.append(f"PART {index}\n{note}")
        progress("Combining the parts")
        messages = prompts.build_analysis_messages(
            prompts.REDUCE_EXECUTIVE_TASK, "\n\n".join(notes), heading="NOTES"
        )
        text, cited, used = await _cited_text(messages, total, on_wait)
        _add(usage, used)
    return {"kind": "executive", "text": text, "citations": _citations(numbered, set(cited))}, usage


async def _item_kind(
    kind: str,
    batches: list[list[Numbered]],
    numbered: dict[int, Candidate],
    total: int,
    task: str,
    on_wait: OnWait | None,
    progress: OnProgress,
) -> tuple[dict[str, Any], dict[str, int]]:
    model, required, others = _ITEM_KINDS[kind]
    usage: dict[str, int] = {}
    items: list[dict[str, str]] = []
    cited: set[int] = set()
    seen: set[str] = set()
    for index, batch in enumerate(batches, start=1):
        if len(batches) > 1:
            progress(f"Reading part {index} of {len(batches)}")
        messages = prompts.build_analysis_messages(task, prompts.format_sources(batch))
        parsed, used = await _run_json(messages, model, on_wait)
        _add(usage, used)
        for raw in parsed.items:
            record = raw.model_dump()
            text, item_cited, _ = guard.filter_citations(str(record.get(required, "")), total)
            if not text or not item_cited:  # an unsupported item is dropped, never shown
                continue
            item = {required: text}
            cited.update(item_cited)
            for name in others:
                value = str(record.get(name, "")).strip()
                if name == "title" or name in ("question", "term"):
                    item[name] = value  # a label, not a claim
                else:
                    clean, extra, _ = guard.filter_citations(value, total)
                    item[name] = clean
                    cited.update(extra)
            label = item.get(_LABEL[kind], text).casefold()
            if not label or label in seen:
                continue
            seen.add(label)
            items.append(item)
    cap = {"faq": MAX_FAQ_ITEMS, "glossary": MAX_GLOSSARY_ITEMS}.get(kind)
    if cap:
        items = items[:cap]
    # Only the citations of items that survived the cap.
    kept: set[int] = set()
    for item in items:
        for value in item.values():
            kept.update(guard.filter_citations(value, total)[1])
    return {"kind": kind, "items": items, "citations": _citations(numbered, kept)}, usage


# --- Structured extraction -----------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One field to extract: its label (how it is shown) and what it means (for the model)."""

    label: str
    description: str


PRESETS: dict[str, FieldSpec] = {
    "dates": FieldSpec(
        "Dates",
        "dates the document states: effective, signing, review dates and when events happen",
    ),
    "parties": FieldSpec("Parties", "people, companies and organisations named, with their role"),
    "amounts": FieldSpec(
        "Amounts", "money, quantities, percentages and limits, with their unit or currency"
    ),
    "requirements": FieldSpec("Requirements", "what must, shall or should be done, and by whom"),
    "deadlines": FieldSpec(
        "Deadlines", "due dates, notice periods, retention periods and other time limits"
    ),
    "contacts": FieldSpec("Contacts", "names, roles, email addresses and phone numbers"),
    "definitions": FieldSpec("Definitions", "terms the document defines, with their meaning"),
}


def normalize_fields(fields: Sequence[str]) -> list[FieldSpec]:
    """Presets (by key or label) and custom names, without repeats, within the configured limit."""
    limit = get_settings().extraction_max_fields
    by_label = {spec.label.casefold(): spec for spec in PRESETS.values()}
    specs: list[FieldSpec] = []
    seen: set[str] = set()
    for raw in fields:
        name = " ".join(raw.split())
        if not name:
            continue
        if len(name) > MAX_FIELD_CHARS:
            raise AnalysisError(
                "bad_field", f"A field name can have at most {MAX_FIELD_CHARS} characters."
            )
        spec = (
            PRESETS.get(name.casefold()) or by_label.get(name.casefold()) or FieldSpec(name, name)
        )
        if spec.label.casefold() in seen:
            continue
        seen.add(spec.label.casefold())
        specs.append(spec)
    if not specs:
        raise AnalysisError("no_fields", "Choose at least one field to extract.")
    if len(specs) > limit:
        raise AnalysisError("too_many_fields", f"Extract at most {limit} fields at once.")
    return specs


def fields_key(specs: Sequence[FieldSpec]) -> str:
    """A stable key for a set of fields (order and case do not matter), for saved results."""
    joined = "|".join(sorted(spec.label.casefold() for spec in specs))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


async def extract(
    specs: Sequence[FieldSpec],
    sources: Sequence[Candidate],
    *,
    on_wait: OnWait | None = None,
    on_progress: OnProgress | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fields pulled out of a document: ``{"kind": "extraction", "fields", "citations"}``.

    Each value has the number of the source it came from; a value without a valid source is
    dropped, and a quote that is not really in that source is blanked (the value is kept).
    """
    if not sources:
        raise AnalysisError("empty", "This document has no indexed text to extract from.")
    limit = get_settings().analysis_max_source_chars
    numbered = dict(enumerate(sources, start=1))
    batches = make_batches(sources, limit)
    progress = on_progress or (lambda _stage: None)
    listing = "\n".join(
        f"- {spec.label}" + ("" if spec.description == spec.label else f": {spec.description}")
        for spec in specs
    )
    task = prompts.EXTRACTION_TASK.format(fields=listing)

    usage: dict[str, int] = {}
    rows: dict[str, list[dict[str, Any]]] = {spec.label: [] for spec in specs}
    seen: set[tuple[str, str]] = set()
    cited: set[int] = set()
    wanted = {spec.label.casefold(): spec.label for spec in specs}
    for index, batch in enumerate(batches, start=1):
        if len(batches) > 1:
            progress(f"Reading part {index} of {len(batches)}")
        messages = prompts.build_analysis_messages(task, prompts.format_sources(batch))
        parsed, used = await _run_json(messages, ExtractionOut, on_wait)
        _add(usage, used)
        for field in parsed.fields:
            label = wanted.get(field.field.strip().casefold())
            if label is None:
                continue
            for value in field.values:
                text = " ".join(value.value.split())
                source = numbered.get(value.source)
                if not text or source is None:
                    continue
                key = (label.casefold(), text.casefold())
                if key in seen:
                    continue
                seen.add(key)
                quote = " ".join(value.quote.split())
                if quote and _squash(quote) not in _squash(source.text):
                    quote = ""
                rows[label].append({"value": text, "n": value.source, "quote": quote})
                cited.add(value.source)
    content = {
        "kind": "extraction",
        "fields": [{"field": spec.label, "values": rows[spec.label]} for spec in specs],
        "citations": _citations(numbered, cited),
    }
    return content, usage


# --- Conflict detection ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """A source as the model sees it: ``n`` is the number it is shown under and cites."""

    n: int
    candidate: Candidate


async def detect_conflicts(
    question: str, passages: Sequence[Passage], *, on_wait: OnWait | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Points on which sources from *different documents* contradict each other.

    Returns ``([{"topic", "claims": [{"n", "document_id", "document_name", "claim"}]}], usage)``.
    No model call is made when the passages come from fewer than two documents. A conflict is kept
    only if its claims name real passages and come from at least two documents; the model's idea
    of a conflict between two parts of one document is not one.
    """
    if len({p.candidate.document_id for p in passages}) < 2:
        return [], {}
    messages = prompts.build_analysis_messages(
        prompts.CONFLICT_TASK.format(question=question),
        prompts.format_sources((p.n, p.candidate) for p in passages),
    )
    parsed, usage = await _run_json(messages, ConflictsOut, on_wait, patient=False)
    by_number = {p.n: p for p in passages}
    conflicts: list[dict[str, Any]] = []
    for item in parsed.conflicts:
        topic = guard.filter_citations(" ".join(item.topic.split()), 0)[0][:200]
        claims: list[dict[str, Any]] = []
        used: set[int] = set()
        for claim in item.claims:
            passage = by_number.get(claim.source)
            text = guard.filter_citations(" ".join(claim.claim.split()), 0)[0][:400]
            if passage is None or not text or claim.source in used:
                continue
            used.add(claim.source)
            claims.append(
                {
                    "n": passage.n,
                    "document_id": passage.candidate.document_id,
                    "document_name": passage.candidate.document_name,
                    "claim": text,
                }
            )
        if topic and len({c["document_id"] for c in claims}) >= 2:
            conflicts.append({"topic": topic, "claims": claims})
    return conflicts, usage


# --- Suggested questions -----------------------------------------------------------------------------

QUESTION_MIN_CHARS = 8
QUESTION_MAX_CHARS = 200
SUGGESTION_TYPES = ("follow_up", "sub_question", "related")
MAX_FOLLOWUPS = 5


def clean_question(raw: str) -> str | None:
    """A suggested question fit to show and to send back as a question, or ``None``.

    One line, no citation markers or wrapping quotes, 8 to 200 characters, ending in "?".
    """
    text = guard.filter_citations(" ".join(raw.split()), 0)[0].strip().strip("\"'“”‘’ ")
    text = re.sub(
        r"^(?:[-*•]\s+|\d{1,2}[.)]\s+)", "", text
    ).strip()  # a list marker, not "2026 ..."
    if not (QUESTION_MIN_CHARS <= len(text) <= QUESTION_MAX_CHARS) or not text.endswith("?"):
        return None
    return text


def _spread(sources: Sequence[Candidate], limit: int) -> list[Candidate]:
    """Sources evenly spaced through the document, within ``limit`` characters (one call)."""
    if sum(len(c.text) for c in sources) <= limit:
        return list(sources)
    average = max(1, sum(len(c.text) for c in sources) // len(sources))
    wanted = max(1, min(len(sources), limit // average))
    step = len(sources) / wanted
    picked: list[Candidate] = []
    used = 0
    for i in range(wanted):
        candidate = sources[min(len(sources) - 1, int(i * step))]
        if used + len(candidate.text) > limit:
            break
        picked.append(candidate)
        used += len(candidate.text)
    return picked or [sources[0]]


async def suggest_questions(
    sources: Sequence[Candidate], count: int, *, on_wait: OnWait | None = None
) -> tuple[dict[str, Any], dict[str, int]]:
    """Questions a reader would ask that this document answers.

    Returns ``({"kind": "questions", "items": [{"question", "n"}], "citations"}, usage)``. A
    question whose source number is not a real source is dropped (the document might not answer
    it), and so are repeats and anything that is not a well-formed question.
    """
    if not sources:
        raise AnalysisError("empty", "This document has no indexed text to suggest questions from.")
    picked = _spread(sources, get_settings().analysis_max_source_chars)
    numbered = dict(enumerate(picked, start=1))
    messages = prompts.build_analysis_messages(
        prompts.SUGGEST_QUESTIONS_TASK.format(count=count),
        prompts.format_sources(numbered.items()),
    )
    parsed, usage = await _run_json(messages, QuestionsOut, on_wait)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed.questions:
        question = clean_question(item.question)
        if question is None or item.source not in numbered or _squash(question) in seen:
            continue
        seen.add(_squash(question))
        items.append({"question": question, "n": item.source})
        if len(items) >= count:
            break
    if not items:
        raise llm.LlmError("failed", UNREADABLE_MESSAGE)
    cited = {item["n"] for item in items}
    return {"kind": "questions", "items": items, "citations": _citations(numbered, cited)}, usage


async def suggest_followups(
    question: str, answer: str, candidates: Sequence[Candidate], *, count: int = 4
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Follow-up, sub- and related questions after an answer, each answerable from a passage.

    ``candidates`` are all the passages retrieved for the question, numbered from 1 as in the
    answer. Returns ``([{"type", "question", "n", "document_id", "chunk_id"}], usage)``. A
    suggestion naming no real passage, of an unknown type, repeating the question or another
    suggestion, or not shaped like a question is dropped. Gives up as quickly as a chat turn does:
    it is an extra after an answer that is already complete.
    """
    if not candidates:
        return [], {}
    numbered = dict(enumerate(candidates, start=1))
    messages = prompts.build_analysis_messages(
        prompts.FOLLOWUP_TASK.format(question=question, answer=answer, count=count),
        prompts.format_sources(numbered.items()),
    )
    parsed, usage = await _run_json(messages, FollowupsOut, None, patient=False)
    seen = {_squash(question.rstrip("?")) + "?"}
    suggestions: list[dict[str, Any]] = []
    for item in parsed.suggestions:
        text = clean_question(item.question)
        kind = item.type.strip().lower().replace("-", "_").replace(" ", "_")
        candidate = numbered.get(item.source)
        if text is None or candidate is None or kind not in SUGGESTION_TYPES:
            continue
        if _squash(text) in seen:
            continue
        seen.add(_squash(text))
        suggestions.append(
            {
                "type": kind,
                "question": text,
                "n": item.source,
                "document_id": candidate.document_id,
                "chunk_id": candidate.chunk_id,
            }
        )
        if len(suggestions) >= MAX_FOLLOWUPS:
            break
    return suggestions, usage
