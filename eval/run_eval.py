"""Retrieval and refusal evaluation over eval/questions.jsonl.

Run inside the backend container (it needs the indexed documents and the Gemini key):

    docker compose exec backend python -m eval.run_eval            # retrieval + guard, no LLM calls
    docker compose exec backend python -m eval.run_eval --verbose  # one line per question

(`eval/` is only in the container with the development override; see eval/README.md.)

For each question it embeds the question once (one embedding request per question; the production
query cache serves the repeat) and reports:

* hit-rate@5 and MRR of the expected document and page, over the answerable questions, for
  dense-only and keyword-only search (the two channels read directly) and for the hybrid result
  that production returns (``retrieval.retrieve``: RRF fusion, then the top ``RETRIEVAL_TOP_K``);
* the guard at ``RELEVANCE_THRESHOLD``, decided by the production ``guard.evaluate`` on the
  production retrieval result: the share of off-topic questions it stops before the LLM, and the
  share of answerable ones it wrongly stops;
* the best dense similarity per question kind, which is how the threshold was chosen;
* p50 and p95 of the embedding, search and fusion stages.

Near-miss questions (topic present, fact absent) are expected to pass the guard: the model's own
refusal handles them, and they are reported but not judged. No LLM is called, so the run costs only
embedding requests. A question whose document is not indexed is skipped and listed (the three
public SANS PDFs are not part of the repository).

The exit status is 1 when an answerable question is wrongly stopped, an off-topic question gets
through the guard, or the hybrid hit-rate falls below ``--min-hit-rate``.
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
from app.core import bm25, embeddings, fusion, guard, retrieval, vectorstore
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
    """What ``build_filter`` needs of a user: the eval searches everything an administrator can."""

    role = "admin"


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def _rank(ordered: list[str], expected: tuple[str, int] | None, matches) -> int | None:
    """1-based position of the first result for the expected document and page, within TOP_K."""
    if expected is None:
        return None
    return next((i for i, chunk_id in enumerate(ordered[:TOP_K], 1) if matches(chunk_id)), None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per question")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.8,
        help="fail when the hybrid hit-rate@5 is below this (default 0.8)",
    )
    args = parser.parse_args()

    settings = get_settings()
    all_questions = [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    search_filter = build_filter(_Admin())

    with get_sessionmaker()() as db:
        index_service.load_bm25(db)
        indexed = set(db.scalars(select(Document.name).where(Document.status == "indexed")).all())
        located = {
            row.id: (row.name, row.page_number, row.page_end)
            for row in db.execute(
                select(Chunk.id, Document.name, Chunk.page_number, Chunk.page_end).join(
                    Document, Document.id == Chunk.document_id
                )
            )
        }

    def needs(question: dict) -> list[str]:
        required = list(question.get("requires", []))
        if question["answerable"]:
            required.append(question["expected_document"])
        return required

    questions = [q for q in all_questions if all(name in indexed for name in needs(q))]
    skipped = [q for q in all_questions if q not in questions]
    if not questions:
        print("None of the questions' documents are indexed. See eval/README.md.", file=sys.stderr)
        return 1

    def is_expected(chunk_id: str, expected: tuple[str, int]) -> bool:
        place = located.get(chunk_id)  # a stale vector has no row: it is simply not a hit
        return place is not None and place[0] == expected[0] and place[1] <= expected[1] <= place[2]

    rows: list[Row] = []
    for q in questions:
        expected = (q["expected_document"], q["expected_page"]) if q["answerable"] else None

        def matches(chunk_id: str, expected: tuple[str, int] | None = expected) -> bool:
            return expected is not None and is_expected(chunk_id, expected)

        started = time.perf_counter()
        vector = embeddings.embed_query(q["question"])
        embed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense = vectorstore.query(vector, settings.retrieval_candidates, search_filter.to_chroma())
        dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        keyword = bm25.get_index().search(
            q["question"], settings.retrieval_candidates, allow=search_filter.allows
        )
        keyword_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fusion.rrf([[h.chunk_id for h in dense], [h.chunk_id for h in keyword]], settings.rrf_k)
        fusion_ms = (time.perf_counter() - started) * 1000

        # Exactly what a chat turn does: the production retrieval, then the production guard.
        result = retrieval.retrieve(q["question"], search_filter)
        decision = guard.evaluate(
            q["question"],
            result.best_similarity,
            result.top_keyword_text,
            settings.relevance_threshold,
            settings.lexical_override_margin,
        )
        ranks = {
            "dense": _rank([h.chunk_id for h in dense], expected, matches),
            "keyword": _rank([h.chunk_id for h in keyword], expected, matches),
            "hybrid": _rank([c.chunk_id for c in result.candidates], expected, matches),
        }
        rows.append(
            Row(
                q["kind"],
                q["question"],
                q["answerable"],
                expected,
                result.best_similarity or 0.0,
                decision.passed,
                ranks,
                {
                    "embed_ms": embed_ms,
                    "dense_ms": dense_ms,
                    "keyword_ms": keyword_ms,
                    "fusion_ms": fusion_ms,
                },
            )
        )

    if args.verbose:
        for r in rows:
            mark = "pass " if r.guard_passed else "REFUSE"
            print(
                f"{r.kind:12s} {r.best_similarity:.3f} {mark} hybrid@{r.ranks['hybrid']} | "
                f"{r.question}"
            )
        print()

    answerable = [r for r in rows if r.answerable]
    print(f"Retrieval over {len(answerable)} answerable questions (top {TOP_K})")
    print(f"{'channel':10s} {'hit-rate@5':>11s} {'MRR':>7s}")
    hit_rates: dict[str, float] = {}
    for channel in ("dense", "keyword", "hybrid"):
        hits = [r.ranks[channel] for r in answerable]
        hit_rates[channel] = sum(1 for h in hits if h) / len(hits)
        mrr = statistics.mean(1 / h if h else 0.0 for h in hits)
        print(f"{channel:10s} {hit_rates[channel]:11.2f} {mrr:7.2f}")

    off_topic = [r for r in rows if r.kind == "off_topic"]
    near_miss = [r for r in rows if r.kind == "near_miss"]
    wrongly = [r for r in answerable if not r.guard_passed]
    leaked = [r for r in off_topic if r.guard_passed]
    print(
        f"\nGuard at RELEVANCE_THRESHOLD={settings.relevance_threshold} "
        f"(lexical override margin {settings.lexical_override_margin})"
    )
    print(
        f"  off-topic stopped before the LLM:        {len(off_topic) - len(leaked)}/{len(off_topic)}"
    )
    print(f"  answerable wrongly stopped:              {len(wrongly)}/{len(answerable)}")
    print(
        f"  near-miss passed on to the model:        "
        f"{sum(1 for r in near_miss if r.guard_passed)}/{len(near_miss)} (expected; the model refuses)"
    )
    by_kind: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_kind[r.kind].append(r)
    print("\nBest dense similarity by kind")
    for kind, group in by_kind.items():
        sims = [r.best_similarity for r in group]
        stopped = sum(1 for r in group if not r.guard_passed)
        print(
            f"  {kind:12s} n={len(group)} min={min(sims):.3f} mean={statistics.mean(sims):.3f} "
            f"max={max(sims):.3f} stopped={stopped}"
        )

    print("\nStage latency (ms)")
    for stage in ("embed_ms", "dense_ms", "keyword_ms", "fusion_ms"):
        values = [r.stages[stage] for r in rows]
        print(
            f"  {stage[:-3]:8s} p50={_percentile(values, 50):7.1f} p95={_percentile(values, 95):7.1f}"
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} question(s) whose documents are not indexed:")
        for q in skipped:
            missing = [name for name in needs(q) if name not in indexed]
            print(f"  - {q['question']}  (needs {', '.join(missing)})")

    problems = []
    if wrongly:
        problems.append(f"{len(wrongly)} answerable question(s) wrongly stopped")
    if leaked:
        problems.append(f"{len(leaked)} off-topic question(s) got through the guard")
    if hit_rates["hybrid"] < args.min_hit_rate:
        problems.append(f"hybrid hit-rate {hit_rates['hybrid']:.2f} < {args.min_hit_rate}")
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
