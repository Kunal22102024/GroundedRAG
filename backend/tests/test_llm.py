"""The chat model is built with thinking disabled and without the library's own retries."""

from __future__ import annotations

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


def test_without_a_key_the_error_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "google_api_key", SecretStr(""))
    llm.clear_cache()
    with pytest.raises(llm.LlmError) as caught:
        llm.get_chat_model()
    assert caught.value.code == "not_configured"
