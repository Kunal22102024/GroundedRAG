"""Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.core.fusion import rrf


def test_scores_follow_the_formula_and_agreement_wins() -> None:
    fused = rrf([["a", "b", "c"], ["b", "d"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in one list, rank 1 in the other
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [chunk for chunk, _ in fused][0] == "b"


def test_a_chunk_missing_from_a_channel_contributes_nothing_for_it() -> None:
    assert dict(rrf([["a"], []]))["a"] == pytest.approx(1 / 61)


def test_scores_are_tiny_so_they_can_never_be_a_relevance_threshold() -> None:
    best = rrf([["a"], ["a"]])[0][1]
    assert best < 0.034  # CLAUDE.md rule 7: gate on cosine similarity, not on this


def test_ties_keep_first_seen_order() -> None:
    assert [chunk for chunk, _ in rrf([["a"], ["b"]])] == ["a", "b"]


def test_nothing_in_nothing_out() -> None:
    assert rrf([]) == [] and rrf([[], []]) == []
