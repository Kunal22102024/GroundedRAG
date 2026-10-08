"""Retrieval and refusal evaluation over eval/questions.jsonl.

Run inside the backend container (it needs the indexed documents and the Gemini key):

    docker compose exec backend python -m eval.run_eval            # retrieval + guard, no LLM calls
    docker compose exec backend python -m eval.run_eval --verbose  # one line per question

For each question it embeds the question once (one embedding request per question) and reports:

* hit-rate@5 and MRR of the expected document and page, for dense-only, keyword-only and hybrid
  (RRF) retrieval, over the answerable questions;
* refusal accuracy at ``RELEVANCE_THRESHOLD``: the share of unanswerable questions the guard
  stops before the LLM, and the share of answerable ones it wrongly stops;
* the best dense similarity per question kind, which is how the threshold was chosen;
* p50 and p95 of the embedding, search and fusion stages.

Near-miss questions (topic present, fact absent) are expected to pass the guard: the model's own
refusal handles them. No LLM is called, so the run costs only embedding requests.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.core import bm25, embeddings, fusion, guard, vectorstore
from app.core.filters import build_filter
from app.db.database import get_sessionmaker
from app.db.models import Chunk, Document
from app.services import index_service

QUESTIONS = Path(__file__).with_name("questions.jsonl")
TOP_K = 5


@dataclass
class Row:
    kind: str
    question: str
    answerable: bool
    expected: tuple[str, int] | None
    best_similarity: float
    guard_passed: bool
    ranks: dict[str, int | None]
    stages: dict[str, float]


class _Admin:
    role = "admin"


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="print one line per question")
    args = parser.parse_args()

    settings = get_settings()
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    search_filter = build_filter(_Admin())

    with get_sessionmaker()() as db:
        index_service.load_bm25(db)
        located = {
            row.id: (row.name, row.page_number, row.page_end)
            for row in db.execute(
                select(Chunk.id, Document.name, Chunk.page_number, Chunk.page_end).join(
                    Document, Document.id == Chunk.document_id
                )
            )
        }
        texts = dict(db.execute(select(Chunk.id, Chunk.content)).all())

    def is_expected(chunk_id: str, expected: tuple[str, int]) -> bool:
        name, first, last = located[chunk_id]
        return name == expected[0] and first <= expected[1] <= last

    rows: list[Row] = []
    for q in questions:
        expected = (q["expected_document"], q["expected_page"]) if q["answerable"] else None
        started = time.perf_counter()
        vector = embeddings.embed_query(q["question"])
        embed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense = vectorstore.query(vector, settings.retrieval_candidates, search_filter.to_chroma())
        dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        keyword = bm25.get_index().search(q["question"], settings.retrieval_candidates, allow=search_filter.allows)
        keyword_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fused = fusion.rrf([[h.chunk_id for h in dense], [h.chunk_id for h in keyword]], settings.rrf_k)
        fusion_ms = (time.perf_counter() - started) * 1000

        orderings = {
            "dense": [h.chunk_id for h in dense],
            "keyword": [h.chunk_id for h in keyword],
            "hybrid": [chunk_id for chunk_id, _ in fused],
        }
        ranks: dict[str, int | None] = {}
        for name, ordered in orderings.items():
            ranks[name] = next((i for i, c in enumerate(ordered[:TOP_K], 1) if expected and is_expected(c, expected)), None)

        best = max((h.similarity for h in dense), default=0.0)
        top_keyword_text = texts.get(keyword[0].chunk_id) if keyword else None
        decision = guard.evaluate(q["question"], best, top_keyword_text, settings.relevance_threshold)
        rows.append(
            Row(q["kind"], q["question"], q["answerable"], expected, best, decision.passed, ranks,
                {"embed_ms": embed_ms, "dense_ms": dense_ms, "keyword_ms": keyword_ms, "fusion_ms": fusion_ms})
        )

    if args.verbose:
        for r in rows:
            mark = "pass " if r.guard_passed else "REFUSE"
            print(f"{r.kind:12s} {r.best_similarity:.3f} {mark} hybrid@{r.ranks['hybrid']} | {r.question}")
        print()

    answerable = [r for r in rows if r.answerable]
    print(f"Retrieval over {len(answerable)} answerable questions (top {TOP_K})")
    print(f"{'channel':10s} {'hit-rate@5':>11s} {'MRR':>7s}")
    for channel in ("dense", "keyword", "hybrid"):
        hits = [r.ranks[channel] for r in answerable]
        hit_rate = sum(1 for h in hits if h) / len(hits)
        mrr = statistics.mean(1 / h if h else 0.0 for h in hits)
        print(f"{channel:10s} {hit_rate:11.2f} {mrr:7.2f}")

    unanswerable = [r for r in rows if not r.answerable]
    stopped = sum(1 for r in unanswerable if not r.guard_passed)
    wrongly = sum(1 for r in answerable if not r.guard_passed)
    print(f"\nGuard at RELEVANCE_THRESHOLD={settings.relevance_threshold}")
    print(f"  unanswerable stopped before the LLM: {stopped}/{len(unanswerable)}")
    print(f"  answerable wrongly stopped:          {wrongly}/{len(answerable)}")
    by_kind: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_kind[r.kind].append(r)
    print("\nBest dense similarity by kind")
    for kind, group in by_kind.items():
        sims = [r.best_similarity for r in group]
        stopped_k = sum(1 for r in group if not r.guard_passed)
        print(f"  {kind:12s} n={len(group)} min={min(sims):.3f} mean={statistics.mean(sims):.3f} max={max(sims):.3f} stopped={stopped_k}")

    print("\nStage latency (ms)")
    for stage in ("embed_ms", "dense_ms", "keyword_ms", "fusion_ms"):
        values = [r.stages[stage] for r in rows]
        print(f"  {stage[:-3]:8s} p50={_percentile(values, 50):7.1f} p95={_percentile(values, 95):7.1f}")
    return 0 if wrongly == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
