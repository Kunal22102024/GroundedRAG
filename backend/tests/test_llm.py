"""The chat model is built with thinking disabled and without the library's own retries."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from app.config import get_settings
from app.core import llm


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(get_settings(), "google_api_key", SecretStr("test-key-not-real"))
    llm.clear_cache()
    yield
    llm.clear_cache()


def test_answer_model_uses_the_configured_generation_settings(built: None) -> None:
    model = llm.get_chat_model("answer")
    settings = get_settings()
    assert model.model.endswith(settings.llm_model)
    assert model.thinking_budget == 0 == settings.llm_thinking_budget  # rule 18
    assert model.temperature == settings.llm_temperature == 0.0
    assert model.max_output_tokens == settings.llm_max_tokens
    assert model.max_retries == 1  # the library's own retries ignore the server's retry hint


def test_rewrite_model_is_small_and_also_thinks_zero(built: None) -> None:
    model = llm.get_chat_model("rewrite")
    assert model.thinking_budget == 0 and model.max_output_tokens == 256


def test_each_event_loop_gets_a_client_of_its_own(built: None) -> None:
    """A client carried into another event loop fails its first call ("Event loop is closed"), and
    every background job runs in a loop of its own, so the client is cached per loop."""

    async def two_calls() -> tuple[object, object]:
        return llm.get_chat_model("answer"), llm.get_chat_model("answer")

    first_a, first_b = asyncio.run(two_calls())
    second_a, second_b = asyncio.run(two_calls())
    assert first_a is first_b and second_a is second_b  # one per loop, not one per call
    assert first_a is not second_a  # a new loop never gets the previous loop's client

    plain = llm.get_chat_model("answer")  # no loop running: its own entry
    assert plain is llm.get_chat_model("answer") and plain is not first_a
    assert not any(loop is not None and loop.is_closed() for loop in llm._models)  # closed: dropped


def test_a_background_job_closes_its_clients_before_its_loop_ends(built: None) -> None:
    """Left to the garbage collector, a client tries to close itself on the finished loop."""
    closed: list[str] = []

    class Aio:
        async def aclose(self) -> None:
            closed.append("aclose")

    class Client:
        aio = Aio()

    async def job() -> str:
        llm.get_chat_model("answer").client = Client()  # what the real model carries
        assert len(llm._models) == 1
        return "result"

    assert llm.run_background(job()) == "result"
    assert closed == ["aclose"]
    assert not llm._models  # and the finished loop's entry is gone


def test_a_failing_background_job_still_closes_its_clients(built: None) -> None:
    closed: list[str] = []

    class Aio:
        async def aclose(self) -> None:
            closed.append("aclose")

    class Client:
        aio = Aio()

    async def job() -> None:
        llm.get_chat_model("answer").client = Client()
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        llm.run_background(job())
    assert closed == ["aclose"]


def test_without_a_key_the_error_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "google_api_key", SecretStr(""))
    llm.clear_cache()
    with pytest.raises(llm.LlmError) as caught:
        llm.get_chat_model()
    assert caught.value.code == "not_configured"
