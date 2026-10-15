"""GraphRAG: widen the retrieved passages through the knowledge graph (multi-hop retrieval).

Plain retrieval scores every chunk on its own, so an answer that needs two documents ("who approves
X?" in one, "what does X require?" in another) only works if both happen to resemble the question.
Here the entities named in the retrieved ("seed") chunks are looked up in the graph and followed
along their relations, up to ``GRAPH_HOPS`` steps and across documents (entities with the same
normalised name are the same node). The chunks that state those relations, and then the chunks that
mention the neighbouring entities, are added behind the seeds, at most
``GRAPH_EXPANSION_MAX_CHUNKS`` of them, nearest first.

Guarantees (CLAUDE.md rules 6, 7, 8 and 24):

* It runs after the relevance guard has passed, on the seeds only; added chunks carry no similarity,
  so the guard and the dashboard's relevance figures never see them, and a refused question is never
  widened.
* Every relation traversed, and every chunk returned, must pass the caller's search filter and
  belong to an indexed document, checked in the query itself and again by ``load_chunks``. A relation
  stated in a passage the caller may not read is not followed, so nothing is learned from it.
* Entities are followed by name only, so a very common name (``KG_MAX_ENTITY_CHUNKS``) is skipped
  instead of linking everything to everything.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import DocumentStatus
from app.core import retrieval
from app.core.filters import SearchFilter
from app.core.retrieval import Candidate
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document, KgEntity, KgMention, KgRelation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Expansion:
    """The passages the graph added, and what the walk saw (for the timings)."""

    candidates: list[Candidate] = field(default_factory=list)
    seed_entities: int = 0
    reached_entities: int = 0
    ms: float = 0.0


def _generic_keys(db: Session, keys: Sequence[str]) -> set[str]:
    """Entities mentioned in so many chunks that following them says nothing."""
    if not keys:
        return set()
    rows = db.execute(
        select(KgEntity.key)
        .join(KgMention, KgMention.entity_id == KgEntity.id)
        .where(KgEntity.key.in_(keys))
        .group_by(KgEntity.key)
        .having(func.count() > get_settings().kg_max_entity_chunks)
    )
    return {key for (key,) in rows}


def _seed_keys(db: Session, chunk_ids: Sequence[str]) -> set[str]:
    rows = db.execute(
        select(KgEntity.key)
        .join(KgMention, KgMention.entity_id == KgEntity.id)
        .where(KgMention.chunk_id.in_(chunk_ids))
        .distinct()
    )
    return {key for (key,) in rows}


def expand(seeds: Sequence[Candidate], search_filter: SearchFilter) -> Expansion:
    """Passages connected to ``seeds`` through the graph, nearest first. Blocking (a few small
    queries); call it from a worker thread. Empty when the graph feature is off or has nothing."""
    settings = get_settings()
    if not settings.graph_retrieval_active or not seeds or search_filter.matches_nothing:
        return Expansion()
    started = time.perf_counter()
    seed_ids = {c.chunk_id for c in seeds}
    visible = [
        Document.status == DocumentStatus.INDEXED.value,
        *search_filter.sql_conditions(Chunk),
    ]
    source = KgEntity.__table__.alias("src")
    target = KgEntity.__table__.alias("tgt")

    with get_sessionmaker()() as db:
        start = _seed_keys(db, list(seed_ids))
        generic = _generic_keys(db, list(start))
        frontier = start - generic
        if not frontier:
            return Expansion(seed_entities=len(start), ms=_ms(started))

        visited = set(frontier) | generic
        # chunk id -> [hops from a seed entity, relations that lead to it at that distance]
        evidence: dict[str, list[int]] = {}
        direct: set[str] = set()  # entities one step from a seed: their mentions are the fallback
        for hop in range(1, settings.graph_hops + 1):
            if not frontier:
                break
            rows = db.execute(
                select(KgRelation.chunk_id, source.c.key, target.c.key)
                .join(source, source.c.id == KgRelation.source_id)
                .join(target, target.c.id == KgRelation.target_id)
                .join(Chunk, Chunk.id == KgRelation.chunk_id)
                .join(Document, Document.id == Chunk.document_id)
                .where(or_(source.c.key.in_(frontier), target.c.key.in_(frontier)), *visible)
            ).all()
            reached: set[str] = set()
            for chunk_id, from_key, to_key in rows:
                if chunk_id not in seed_ids:
                    best = evidence.setdefault(chunk_id, [hop, 0])
                    if hop < best[0]:
                        best[:] = [hop, 0]
                    if best[0] == hop:
                        best[1] += 1
                reached |= {from_key, to_key} - visited
            reached -= _generic_keys(db, list(reached))
            if hop == 1:
                direct = set(reached)
            visited |= reached
            frontier = reached

        ranked = sorted(evidence, key=lambda c: (evidence[c][0], -evidence[c][1], c))
        wanted = settings.graph_expansion_max_chunks
        chosen = ranked[:wanted]
        if len(chosen) < wanted and direct:
            # Room left: passages that mention a neighbour, the ones naming most neighbours first.
            mentions = db.execute(
                select(KgMention.chunk_id, func.count(func.distinct(KgEntity.key)).label("n"))
                .join(KgEntity, KgEntity.id == KgMention.entity_id)
                .join(Chunk, Chunk.id == KgMention.chunk_id)
                .join(Document, Document.id == Chunk.document_id)
                .where(KgEntity.key.in_(direct), *visible)
                .group_by(KgMention.chunk_id)
                .order_by(func.count(func.distinct(KgEntity.key)).desc(), KgMention.chunk_id)
            ).all()
            for chunk_id, _ in mentions:
                if chunk_id not in seed_ids and chunk_id not in chosen:
                    chosen.append(chunk_id)
                if len(chosen) == wanted:
                    break

        rows_by_id = retrieval.load_chunks(db, chosen, search_filter)  # the filter, once more
        added = [
            retrieval.candidate_from_row(rows_by_id[chunk_id])
            for chunk_id in chosen
            if chunk_id in rows_by_id
        ]
    return Expansion(added, len(start), len(visited) - len(start), _ms(started))


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
