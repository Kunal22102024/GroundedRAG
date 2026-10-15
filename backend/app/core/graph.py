"""Knowledge-graph extraction: entities and the relationships between them, grounded in chunks.

The model reads a document in batches of numbered ``<source>`` blocks (framed as untrusted data,
CLAUDE.md rule 10) and lists entities and relations as JSON. Nothing it says is kept on trust
(rule 39):

* an **entity** survives only if its name really occurs in at least one chunk read, and its
  *mentions* are not what the model claimed but every chunk read that contains the name;
* a **relation** survives only if the source number it names is a real source of that batch, both
  entity names occur in that very chunk, and it joins two different entities; its quote is kept
  only when it is a passage of that chunk (otherwise it is blanked, as in extraction);
* labels and types are normalised, and a type outside the vocabulary becomes ``other``.

Entities are merged by a normalised name (``normalize_key``), within the document here and across
documents later, which is what links documents to each other. A document longer than
``KG_MAX_BATCHES`` batches is read at evenly spaced parts and ``coverage`` says how much was read.

No FastAPI and no database: ``services/graph_service.py`` loads the chunks and stores the result.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.constants import EntityType
from app.core import analysis, prompts
from app.core.analysis import OnProgress, OnWait
from app.core.retrieval import Candidate

logger = logging.getLogger(__name__)

ENTITY_TYPES = frozenset(t.value for t in EntityType)
MAX_NAME_CHARS = 200  # kg_entities.name / key
MAX_LABEL_CHARS = 64  # kg_relations.label
MAX_DESCRIPTION_CHARS = 400
MAX_QUOTE_CHARS = 300
MIN_NAME_CHARS = 2
MAX_MENTIONS_PER_ENTITY = 200

_NON_WORD = re.compile(r"[\W_]+")
_MARKERS = re.compile(r"\s*\[\d+(?:\s*[,;–-]\s*\d+)*\]")  # [3] or [1, 2] the model added anyway


# --- What the model returns (anything else is ignored) ------------------------------------------


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RawEntity(_Loose):
    name: str | None = None
    type: str | None = None
    description: str | None = None
    # A hint to the model to ground itself; mentions come from the text, not from this.
    sources: list[Any] = Field(default_factory=list)


class RawRelation(_Loose):
    subject: str | None = None
    object: str | None = None
    label: str | None = None
    source: int | None = None
    quote: str | None = None


class GraphOut(_Loose):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)


# --- The result ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A merged entity: its mentions are chunk ids, in reading order."""

    key: str
    name: str
    type: str
    description: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class Relation:
    """``subject --label--> object`` stated in ``chunk_id``; subject and object are entity keys."""

    subject: str
    object: str
    label: str
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class Graph:
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class GraphResult:
    graph: Graph
    usage: dict[str, int]
    batches: int  # model calls made
    coverage: float  # share of the document's chunks that were read (1.0 = all)


# --- Names, labels and quotes -----------------------------------------------------------------------


def squash(text: str) -> str:
    """Casefolded text with punctuation and whitespace collapsed to single spaces: the form in
    which names and quotes are compared with a chunk (linear in the text)."""
    return " ".join(_NON_WORD.sub(" ", text.casefold()).split())


def normalize_key(name: str) -> str:
    """The identity of an entity: ``"The Annual-Leave Policy"`` and ``"annual leave policy"`` agree."""
    squashed = squash(name)
    if squashed.startswith("the "):
        squashed = squashed[4:]
    return squashed[:MAX_NAME_CHARS]


def occurs(key: str, squashed_text: str) -> bool:
    """Whether a normalised name appears in a squashed text as whole words."""
    return bool(key) and f" {key} " in f" {squashed_text} "


def normalize_label(label: str | None) -> str:
    """``"Reports To"`` -> ``"reports_to"``; empty when nothing usable is left."""
    return "_".join(_NON_WORD.sub(" ", (label or "").casefold()).split())[:MAX_LABEL_CHARS]


def normalize_type(value: str | None) -> str:
    cleaned = (value or "").strip().casefold()
    return cleaned if cleaned in ENTITY_TYPES else EntityType.OTHER.value


def _clean_text(value: str | None, limit: int) -> str:
    return " ".join(_MARKERS.sub("", value or "").split())[:limit]


def _display_name(value: str | None) -> str:
    return " ".join((value or "").split())[:MAX_NAME_CHARS]


# --- Reading a document ------------------------------------------------------------------------------


def choose_batches(batches: list[list[Any]], limit: int) -> list[list[Any]]:
    """At most ``limit`` batches, evenly spaced through the document (all when it is shorter)."""
    if len(batches) <= limit:
        return batches
    if limit == 1:
        return [batches[0]]
    positions = sorted({round(i * (len(batches) - 1) / (limit - 1)) for i in range(limit)})
    return [batches[i] for i in positions]


class _Entry:
    """What the batches said about one entity key, before merging."""

    def __init__(self) -> None:
        self.names: Counter[str] = Counter()
        self.types: Counter[str] = Counter()
        self.descriptions: list[str] = []

    def add(self, name: str, type_: str | None = None, description: str = "") -> None:
        self.names[name] += 1
        if type_ and type_ != EntityType.OTHER.value:
            self.types[type_] += 1
        if description:
            self.descriptions.append(description)


async def extract_graph(
    sources: Sequence[Candidate],
    *,
    on_wait: OnWait | None = None,
    on_progress: OnProgress | None = None,
) -> GraphResult:
    """Entities and relations of one document, grounded in its chunks (``sources`` in reading
    order). Raises ``llm.LlmError`` when the model fails; nothing is half-returned."""
    settings = get_settings()
    batches = choose_batches(
        analysis.make_batches(sources, settings.kg_batch_chars), settings.kg_max_batches
    )
    read = [pair for batch in batches for pair in batch]
    texts = {n: squash(f"{c.section_title} {c.text}") for n, c in read}
    quotes_in = {n: squash(c.text) for n, c in read}

    entries: dict[str, _Entry] = {}
    relations: dict[
        tuple[str, str, str, int], str
    ] = {}  # (subject, object, label, source) -> quote
    usage: dict[str, int] = {}
    task = prompts.GRAPH_TASK.format(
        max_entities=settings.kg_max_entities_per_batch,
        max_relations=settings.kg_max_relations_per_batch,
    )

    for index, batch in enumerate(batches, start=1):
        if on_progress is not None:
            on_progress(f"Reading part {index} of {len(batches)}")
        messages = prompts.build_analysis_messages(task, prompts.format_sources(batch))
        out, used = await analysis.run_json(messages, GraphOut, on_wait, kind="graph")
        analysis.add_usage(usage, used)
        numbers = {n for n, _ in batch}

        for raw in out.entities[: settings.kg_max_entities_per_batch * 2]:
            name = _display_name(raw.name)
            key = normalize_key(name)
            if len(key) < MIN_NAME_CHARS:
                continue
            entries.setdefault(key, _Entry()).add(
                name, normalize_type(raw.type), _clean_text(raw.description, MAX_DESCRIPTION_CHARS)
            )

        for rel in out.relations[: settings.kg_max_relations_per_batch * 2]:
            label = normalize_label(rel.label)
            subject_name, object_name = _display_name(rel.subject), _display_name(rel.object)
            subject, object_ = normalize_key(subject_name), normalize_key(object_name)
            if (
                not label
                or rel.source not in numbers
                or len(subject) < MIN_NAME_CHARS
                or len(object_) < MIN_NAME_CHARS
                or subject == object_
                # Both ends must be named in the very chunk the relation is attributed to.
                or not occurs(subject, texts[rel.source])
                or not occurs(object_, texts[rel.source])
            ):
                continue
            quote = _clean_text(rel.quote, MAX_QUOTE_CHARS)
            if squash(quote) not in quotes_in[rel.source] or not squash(quote):
                quote = ""  # not a passage of that chunk: never shown as one
            entries.setdefault(subject, _Entry()).add(subject_name)
            entries.setdefault(object_, _Entry()).add(object_name)
            relations.setdefault((subject, object_, label, rel.source), quote)

    # Mentions are every chunk read that contains the name, whatever the model listed.
    chunk_order = [(n, c.chunk_id) for n, c in read]
    mentions: dict[str, tuple[str, ...]] = {}
    for key in entries:
        found = tuple(chunk_id for n, chunk_id in chunk_order if occurs(key, texts[n]))
        if found:
            mentions[key] = found[:MAX_MENTIONS_PER_ENTITY]

    keep = sorted(mentions, key=lambda k: (-len(mentions[k]), k))[: settings.kg_max_entities]
    kept = set(keep)
    chunk_of = dict(chunk_order)
    entities = tuple(
        Entity(
            key=key,
            name=entries[key].names.most_common(1)[0][0],
            type=entries[key].types.most_common(1)[0][0]
            if entries[key].types
            else EntityType.OTHER.value,
            description=max(entries[key].descriptions, key=len, default=""),
            chunk_ids=mentions[key],
        )
        for key in keep
    )
    graph_relations = tuple(
        Relation(subject, object_, label, chunk_of[source], quote)
        for (subject, object_, label, source), quote in relations.items()
        if subject in kept and object_ in kept
    )
    coverage = 1.0 if not sources else round(len(read) / len(sources), 4)
    logger.info(
        "knowledge graph extracted",
        extra={
            "entities": len(entities),
            "relations": len(graph_relations),
            "batches": len(batches),
            "coverage": coverage,
        },
    )
    return GraphResult(Graph(entities, graph_relations), usage, len(batches), coverage)
