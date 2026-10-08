"""Citation verification: markers are checked while streaming; a bad answer is retried once."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.constants import REFUSAL_MESSAGE
from app.core import generation, guard, llm
from app.core.retrieval import Candidate
from tests.conftest import FakeChatModel


def feed_all(text: str, sources: int, piece: int | None = None) -> tuple[str, list[int], int]:
    citation_filter = guard.CitationFilter(sources)
    pieces = [text] if piece is None else [text[i : i + piece] for i in range(0, len(text), piece)]
    shown = "".join(citation_filter.feed(p) for p in pieces) + citation_filter.finish()
    return shown, citation_filter.cited, citation_filter.invalid


def test_valid_markers_pass_through() -> None:
    assert feed_all("Leave is 25 days [1]. Parents get 26 weeks [2].", 2) == (
        "Leave is 25 days [1]. Parents get 26 weeks [2].",
        [1, 2],
        0,
    )


def test_markers_that_map_to_no_source_are_removed_and_counted() -> None:
    shown, cited, invalid = feed_all("A [1] and B [7] and C [0].", 3)
    assert "[7]" not in shown and "[0]" not in shown and "[1]" in shown
    assert (cited, invalid) == ([1], 2)


def test_lists_are_normalized_and_partially_valid_lists_keep_the_valid_part() -> None:
    assert feed_all("Fact [1, 3].", 3)[0] == "Fact [1][3]."
    shown, cited, invalid = feed_all("Fact [2, 9].", 3)
    assert (shown, cited, invalid) == ("Fact [2].", [2], 1)


@pytest.mark.parametrize("piece", [1, 2, 3, 5])
def test_a_marker_split_across_stream_pieces_is_handled_the_same(piece: int) -> None:
    text = "Alpha [1][12] beta [2, 3]. Done."
    assert feed_all(text, 3, piece) == feed_all(text, 3)


def test_brackets_that_are_not_citations_are_left_alone() -> None:
    text = "Use [a-z] or [] and see [the guide](http://x) for [ERR_404_NET]."
    assert feed_all(text, 3)[0] == text


def test_a_marker_cut_off_by_the_token_limit_is_dropped() -> None:
    assert feed_all("The limit is 3 attempts [", 2)[0] == "The limit is 3 attempts "
    assert feed_all("The limit is 3 attempts [1", 2)[0] == "The limit is 3 attempts "


def test_tidy_up_after_removed_markers() -> None:
    assert guard.filter_citations("Done [9] . Next  point [8].", 2)[0] == "Done. Next point."


def test_refusal_sentence_is_recognised_even_when_quoted_or_cited() -> None:
    assert guard.is_refusal(REFUSAL_MESSAGE)
    assert guard.is_refusal(f'"{REFUSAL_MESSAGE}" [1]')
    assert not guard.is_refusal("Leave is 25 days [1].")


# --- Regeneration: one retry, then strip and flag -----------------------------------------------


def candidates(count: int = 2) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=f"c{i}",
            document_id="d",
            document_name="policy.md",
            text=f"text {i}",
            page_number=1,
            page_end=1,
            section_title="s",
            department="IT",
            document_type="Policy",
            access_level="Internal",
            similarity=0.9,
            keyword_score=None,
            rrf_score=0.03,
        )
        for i in range(1, count + 1)
    ]


def run(question: str = "q") -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in generation.generate_answer(question, candidates())]

    return asyncio.run(collect())


def test_a_good_answer_takes_one_call(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("Leave is ", "25 days [1].")
    events = run()
    result = events[-1]
    assert len(fake_llm.calls) == 1
    assert result.text == "Leave is 25 days [1]." and result.cited == [1] and not result.retried
    assert not any(isinstance(e, generation.Restart) for e in events)
    assert result.usage["reasoning_tokens"] == 0 and result.usage["input_tokens"] == 120


def test_an_invalid_citation_triggers_exactly_one_retry(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("Wrong source [9].").reply("Right source [2].")
    events = run()
    assert len(fake_llm.calls) == 2
    assert any(isinstance(e, generation.Restart) for e in events)
    assert "Reminder: cite only" in str(fake_llm.calls[1][-1].content)  # the stricter second prompt
    result = events[-1]
    assert result.text == "Right source [2]." and result.retried and result.invalid_removed == 0


def test_still_invalid_after_the_retry_strips_the_markers_and_flags(
    fake_llm: FakeChatModel,
) -> None:
    fake_llm.reply("Claim [9].").reply("Claim again [8] and [1].")
    result = run()[-1]
    assert len(fake_llm.calls) == 2  # never a third
    assert "[8]" not in result.text and "[1]" in result.text
    assert result.invalid_removed == 1 and result.retried


def test_an_answer_with_no_citation_at_all_is_retried_then_flagged(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("Leave is 25 days.").reply("Leave is still 25 days.")
    result = run()[-1]
    assert len(fake_llm.calls) == 2 and result.uncited and result.cited == []


def test_the_model_refusing_by_itself_is_a_refusal_not_an_uncited_answer(
    fake_llm: FakeChatModel,
) -> None:
    fake_llm.reply(REFUSAL_MESSAGE)
    result = run()[-1]
    assert len(fake_llm.calls) == 1
    assert result.refused and result.text == REFUSAL_MESSAGE and not result.uncited


# --- Gemini finish reasons --------------------------------------------------------------------


def test_max_tokens_keeps_the_partial_answer_and_flags_it(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("Leave is 25 days [1] and", finish="MAX_TOKENS")
    result = run()[-1]
    assert result.truncated and result.text.startswith("Leave is 25 days [1]")


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
def test_a_safety_stop_is_an_error_not_a_blank_answer(fake_llm: FakeChatModel, finish: str) -> None:
    fake_llm.reply("partial", finish=finish)
    with pytest.raises(llm.LlmError) as caught:
        run()
    assert caught.value.code == "blocked"


def test_a_blocked_prompt_is_an_error(fake_llm: FakeChatModel) -> None:
    fake_llm.block()
    with pytest.raises(llm.LlmError) as caught:
        run()
    assert caught.value.code == "blocked"


def test_an_empty_response_is_an_error(fake_llm: FakeChatModel) -> None:
    fake_llm.reply()
    with pytest.raises(llm.LlmError) as caught:
        run()
    assert caught.value.code == "empty"
