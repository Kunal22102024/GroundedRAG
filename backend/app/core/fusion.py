"""Reciprocal Rank Fusion (TDD 5.3). A pure function."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked id lists: ``score(d) = sum over lists of 1 / (k + rank)`` with 1-based ranks.

    A chunk missing from a list contributes nothing for it. The scores are for ordering only: with
    ``k = 60`` and two lists the maximum is about 0.033, so they must never be compared to a
    relevance threshold (CLAUDE.md rule 7). Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
