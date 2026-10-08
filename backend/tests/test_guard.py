"""The relevance gate and the lexical override (TDD 6.3)."""

from __future__ import annotations

import pytest

from app.core.guard import contains_token, evaluate, identifier_tokens

THRESHOLD = 0.60


def test_gate_passes_at_and_above_the_threshold_and_refuses_below() -> None:
    assert evaluate("q", 0.80, None, THRESHOLD).reason == "relevant"
    assert evaluate("q", THRESHOLD, None, THRESHOLD).passed
    refused = evaluate("q", 0.59, None, THRESHOLD)
    assert not refused.passed and refused.reason == "below_threshold" and refused.score == 0.59


def test_no_candidates_is_a_refusal() -> None:
    decision = evaluate("q", None, None, THRESHOLD)
    assert not decision.passed and decision.reason == "no_candidates"


def test_exact_token_passes_despite_a_low_score() -> None:
    chunk = "When the gateway returns ERR_404_NET the upstream service could not be reached."
    decision = evaluate("so like what is that err_404_net thing??", 0.31, chunk, THRESHOLD)
    assert decision.passed and decision.reason == "lexical_override"


def test_override_needs_the_token_in_the_top_keyword_chunk() -> None:
    chunk = "Emergency reboots are allowed on Sundays only."
    assert not evaluate("what is ERR_404_NET", 0.31, chunk, THRESHOLD).passed
    assert not evaluate("what is ERR_404_NET", 0.31, None, THRESHOLD).passed


def test_ordinary_words_never_trigger_the_override() -> None:
    chunk = "Annual leave is 25 days. Backups are retained for 35 days."
    assert not evaluate("how many sick days are there", 0.30, chunk, THRESHOLD).passed


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does ERR_404_NET mean?", ["ERR_404_NET"]),
        ("What does safeguard EM-09 require?", ["EM-09"]),
        ("What runs at 01:30?", ["01:30"]),
        ("meaning of error E2", ["E2"]),
        ("code 404 in the gateway", ["404"]),
        ("Who won the 2018 football world cup?", []),  # a year is not an identifier
        ("a well-known e-mail policy", []),
        ("Which version, v2.3.1 or 2.3?", ["v2.3.1", "2.3"]),
    ],
)
def test_identifier_tokens(question: str, expected: list[str]) -> None:
    assert identifier_tokens(question) == expected


def test_token_must_match_as_a_whole_token_case_insensitively() -> None:
    assert contains_token("see err_404_net.", "ERR_404_NET")
    assert contains_token("E2 | Quota exceeded", "e2")
    assert not contains_token("ERR_404_NETWORK", "ERR_404_NET")
    assert not contains_token("at 101:30", "01:30")
