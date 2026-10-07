"""Gemini embeddings wrapper: task types, normalization, batching, retries and error handling."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from app.config import get_settings
from app.core import embeddings
from app.core.embeddings import (
    EmbeddingAuthError,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingQuotaError,
    EmbeddingRateLimitError,
)

DIM = 768
RATE_LIMIT_MESSAGE = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota. Quota exceeded for metric: "
    "embed_content_free_tier_requests, limit: 100. Please retry in 12.5s."
)


class FakeClientError(Exception):
    """Mimics google.genai.errors.ClientError / ServerError: the HTTP status is ``code``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


def wrapped(code: int, message: str) -> Exception:
    """LangChain wraps the SDK error: the interesting one is only reachable via ``__cause__``."""
    outer = RuntimeError(f"Error embedding content: {message}")
    outer.__cause__ = FakeClientError(code, message)
    return outer


@dataclass
class Call:
    kind: str
    texts: list[str]
    task_type: str
    dimension: int


class FakeClient:
    """Stands in for GoogleGenerativeAIEmbeddings. ``script`` lists what each call does in order."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.script: list[Exception | None] = []
        self.dimension_override: int | None = None
        self.zero_vectors = False
        self.short_batches = False
        self.active = 0
        self.max_active = 0
        self.hold = 0.0
        self._lock = threading.Lock()

    def _step(self) -> None:
        with self._lock:
            step = self.script.pop(0) if self.script else None
        if step is not None:
            raise step

    def _vector(self, text: str, dimension: int) -> list[float]:
        if self.zero_vectors:
            return [0.0] * dimension
        vector = [0.0] * dimension
        vector[0], vector[1], vector[2] = (
            3.0,
            4.0,
            float(sum(map(ord, text)) % 97),
        )  # not unit length
        return vector

    def embed_documents(
        self, texts: list[str], *, batch_size: int, task_type: str, output_dimensionality: int
    ) -> list[list[float]]:
        with self._lock:
            self.calls.append(Call("docs", list(texts), task_type, output_dimensionality))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.hold:
                time.sleep(self.hold)
            self._step()
            size = self.dimension_override or output_dimensionality
            out = [self._vector(t, size) for t in texts]
            return out[:-1] if self.short_batches else out
        finally:
            with self._lock:
                self.active -= 1

    def embed_query(self, text: str, *, task_type: str, output_dimensionality: int) -> list[float]:
        with self._lock:
            self.calls.append(Call("query", [text], task_type, output_dimensionality))
        self._step()
        return self._vector(text, self.dimension_override or output_dimensionality)


class Clock:
    """A fake clock: ``sleep`` advances time instantly and is recorded."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(embeddings, "_client", lambda: fake)
    return fake


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake_clock = Clock()
    monkeypatch.setattr(embeddings, "_sleep", fake_clock.sleep)
    monkeypatch.setattr(embeddings, "_monotonic", fake_clock.monotonic)
    monkeypatch.setattr(embeddings, "_jitter", lambda low, high: 1.0)
    return fake_clock


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


# --- Task types, dimension, normalization ---------------------------------------------------------


def test_documents_use_the_document_task_type_and_the_configured_dimension(
    client: FakeClient,
) -> None:
    vectors = embeddings.embed_documents(["alpha", "beta"])
    call = client.calls[0]
    assert (call.kind, call.task_type, call.dimension) == ("docs", "RETRIEVAL_DOCUMENT", DIM)
    assert all(len(v) == DIM for v in vectors)


def test_queries_use_the_query_task_type(client: FakeClient) -> None:
    embeddings.embed_query("what is the retry limit?")
    call = client.calls[0]
    assert (call.kind, call.task_type, call.dimension) == ("query", "RETRIEVAL_QUERY", DIM)


def test_every_vector_is_normalized_to_unit_length(client: FakeClient) -> None:
    for vector in embeddings.embed_documents(["alpha", "beta", "gamma"]) + [
        embeddings.embed_query("delta")
    ]:
        assert norm(vector) == pytest.approx(1.0, abs=1e-9)


def test_wrong_dimension_from_the_service_is_an_error(client: FakeClient) -> None:
    client.dimension_override = 3072
    with pytest.raises(EmbeddingError, match="unexpected size"):
        embeddings.embed_documents(["alpha"])


def test_zero_vector_is_an_error(client: FakeClient) -> None:
    client.zero_vectors = True
    with pytest.raises(EmbeddingError, match="empty vector"):
        embeddings.embed_query("alpha")


def test_wrong_number_of_vectors_is_an_error(client: FakeClient) -> None:
    client.short_batches = True
    with pytest.raises(EmbeddingError, match="wrong number"):
        embeddings.embed_documents(["a1", "b2"])


# --- Deduplication and batching ----------------------------------------------------------------------


def test_identical_texts_are_embedded_once_and_results_keep_input_order(client: FakeClient) -> None:
    result = embeddings.embed_documents(["same text", "other text", "same text"])
    assert sorted(client.calls[0].texts) == ["other text", "same text"]
    assert result[0] == result[2] and result[0] != result[1]


def test_texts_are_batched_and_order_is_preserved(client: FakeClient) -> None:
    texts = [f"distinct text number {n}" for n in range(70)]
    result = embeddings.embed_documents(texts)
    assert sorted(len(c.texts) for c in client.calls) == [6, 32, 32]
    expected = {t: client._vector(t, DIM) for t in texts}
    for text, vector in zip(texts, result, strict=True):
        raw = expected[text]
        assert vector == pytest.approx([x / norm(raw) for x in raw])


def test_concurrency_is_bounded(client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "embedding_batch_size", 1)
    monkeypatch.setattr(get_settings(), "embedding_max_concurrency", 2)
    client.hold = 0.05
    embeddings.embed_documents([f"text {n}" for n in range(8)])
    assert client.max_active == 2


def test_empty_input_makes_no_calls(client: FakeClient) -> None:
    assert embeddings.embed_documents([]) == []
    assert client.calls == []


def test_one_failing_batch_fails_the_whole_call(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "embedding_batch_size", 1)
    client.script = [None, wrapped(400, "API key not valid. Please pass a valid API key.")]
    with pytest.raises(EmbeddingAuthError):
        embeddings.embed_documents(["one", "two", "three"])


# --- Retries --------------------------------------------------------------------------------------------


def test_rate_limit_waits_for_the_servers_retry_hint_then_succeeds(
    client: FakeClient, clock: Clock
) -> None:
    client.script = [wrapped(429, RATE_LIMIT_MESSAGE), None]
    assert len(embeddings.embed_documents(["alpha"])) == 1
    assert clock.sleeps == [13.5]  # hint 12.5 s + 1 s margin
    assert len(client.calls) == 2


def test_transient_errors_back_off_exponentially(
    client: FakeClient, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "embedding_max_retries", 5)
    client.script = [
        wrapped(503, "UNAVAILABLE"),
        wrapped(500, "INTERNAL"),
        wrapped(503, "UNAVAILABLE"),
        None,
    ]
    embeddings.embed_query("alpha")
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_network_errors_are_retried(client: FakeClient, clock: Clock) -> None:
    client.script = [ConnectionError("reset"), TimeoutError("slow"), None]
    embeddings.embed_query("alpha")
    assert len(client.calls) == 3 and len(clock.sleeps) == 2


def test_rate_limit_that_never_clears_raises_a_helpful_error(
    client: FakeClient, clock: Clock
) -> None:
    client.script = [wrapped(429, RATE_LIMIT_MESSAGE)] * 10
    with pytest.raises(EmbeddingRateLimitError, match="free-tier"):
        embeddings.embed_documents(["alpha"])
    assert len(client.calls) == get_settings().embedding_max_retries + 1


def test_a_server_suggested_wait_that_is_too_long_is_reported_instead_of_slept(
    client: FakeClient, clock: Clock
) -> None:
    client.script = [wrapped(429, "RESOURCE_EXHAUSTED. Please retry in 600.0s.")]
    with pytest.raises(EmbeddingRateLimitError):
        embeddings.embed_query("alpha")
    assert clock.sleeps == []


def test_outage_that_never_clears_is_an_error_without_blaming_rate_limits(
    client: FakeClient, clock: Clock
) -> None:
    client.script = [wrapped(503, "UNAVAILABLE")] * 10
    with pytest.raises(EmbeddingError, match="unavailable") as caught:
        embeddings.embed_query("alpha")
    assert not isinstance(caught.value, EmbeddingRateLimitError)


def test_a_rate_limit_pauses_all_callers_not_just_the_one_that_hit_it(
    client: FakeClient, clock: Clock
) -> None:
    embeddings._start_cooldown(7.0)  # another thread just received a 429
    embeddings.embed_query("alpha")
    assert clock.sleeps == [7.0] and len(client.calls) == 1


# --- Errors that must not be retried ----------------------------------------------------------------------------


def test_invalid_key_is_reported_clearly_and_not_retried(client: FakeClient, clock: Clock) -> None:
    client.script = [
        wrapped(400, "INVALID_ARGUMENT. API key not valid. Please pass a valid API key.")
    ]
    with pytest.raises(EmbeddingAuthError, match="GOOGLE_API_KEY"):
        embeddings.embed_documents(["alpha"])
    assert len(client.calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("code", [401, 403])
def test_auth_status_codes(client: FakeClient, code: int) -> None:
    client.script = [wrapped(code, "denied")]
    with pytest.raises(EmbeddingAuthError):
        embeddings.embed_query("alpha")


def test_daily_quota_is_not_retried(client: FakeClient, clock: Clock) -> None:
    client.script = [
        wrapped(
            429,
            "RESOURCE_EXHAUSTED. quotaId: EmbedContentRequestsPerDayPerProjectPerModel-FreeTier",
        )
    ]
    with pytest.raises(EmbeddingQuotaError, match="exhausted"):
        embeddings.embed_query("alpha")
    assert len(client.calls) == 1 and clock.sleeps == []


def test_other_bad_requests_are_input_errors(client: FakeClient) -> None:
    client.script = [wrapped(400, "INVALID_ARGUMENT. content is empty")]
    with pytest.raises(EmbeddingInputError):
        embeddings.embed_query("alpha")


def test_unknown_failures_are_reported_without_their_message(client: FakeClient) -> None:
    client.script = [ValueError("something about alpha document text")]
    with pytest.raises(EmbeddingError) as caught:
        embeddings.embed_query("alpha")
    assert str(caught.value) == "Embedding failed (ValueError)."


# --- Input validation ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_blank_text_is_rejected_before_any_call(client: FakeClient, bad: str) -> None:
    with pytest.raises(EmbeddingInputError, match="empty"):
        embeddings.embed_documents(["fine", bad])
    with pytest.raises(EmbeddingInputError):
        embeddings.embed_query(bad)
    assert client.calls == []


def test_text_that_would_be_silently_truncated_is_rejected(client: FakeClient) -> None:
    too_long = "x" * 2001
    with pytest.raises(EmbeddingInputError, match="truncated") as caught:
        embeddings.embed_documents(["ok", too_long])
    assert "xxxx" not in str(caught.value), "the error must not echo document text"
    assert "item 1" in str(caught.value)
    with pytest.raises(EmbeddingInputError):
        embeddings.embed_query(too_long)
    assert client.calls == []
    embeddings.embed_documents(["y" * 2000])  # exactly at the limit is fine


# --- Configuration ----------------------------------------------------------------------------------------------------


def test_missing_api_key_is_a_clear_configuration_error() -> None:
    embeddings.clear_caches()
    assert get_settings().google_api_key.get_secret_value() == ""
    with pytest.raises(EmbeddingConfigError, match="GOOGLE_API_KEY is not set"):
        embeddings.embed_documents(["alpha"])
    with pytest.raises(EmbeddingConfigError):
        embeddings.embed_query("alpha")


# --- Query cache ----------------------------------------------------------------------------------------------------------


def test_repeated_questions_are_served_from_the_cache(client: FakeClient) -> None:
    first = embeddings.embed_query("what is the retry limit?")
    second = embeddings.embed_query("what is the retry limit?")
    assert first == second and len(client.calls) == 1
    second.append(99.0)  # callers cannot corrupt the cache
    assert len(embeddings.embed_query("what is the retry limit?")) == DIM


def test_cache_evicts_the_least_recently_used_question(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embeddings, "QUERY_CACHE_SIZE", 2)
    for question in ("aa", "bb", "aa", "cc"):  # "bb" is the least recently used when "cc" arrives
        embeddings.embed_query(question)
    assert len(client.calls) == 3
    embeddings.embed_query("aa")
    embeddings.embed_query("cc")
    assert len(client.calls) == 3, "aa and cc are still cached"
    embeddings.embed_query("bb")
    assert len(client.calls) == 4, "bb was evicted"


# --- Readiness probe ----------------------------------------------------------------------------------------------------------


def test_probe_reports_ok(client: FakeClient) -> None:
    assert embeddings.probe() == "ok"


def test_probe_treats_rate_limiting_as_reachable_and_does_not_retry(
    client: FakeClient, clock: Clock
) -> None:
    client.script = [wrapped(429, RATE_LIMIT_MESSAGE)]
    assert embeddings.probe() == "rate_limited"
    assert len(client.calls) == 1 and clock.sleeps == []


def test_probe_raises_for_a_rejected_key(client: FakeClient) -> None:
    client.script = [wrapped(400, "API key not valid")]
    with pytest.raises(EmbeddingAuthError):
        embeddings.probe()


# --- Privacy ----------------------------------------------------------------------------------------------------------------------


def test_retry_logs_contain_no_text_or_message(
    client: FakeClient, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    secret_text = "CONFIDENTIAL-SALARY-TABLE"
    client.script = [wrapped(429, f"RESOURCE_EXHAUSTED {secret_text} Please retry in 3s.")]
    with caplog.at_level(logging.DEBUG):
        embeddings.embed_documents([secret_text])
    rendered = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert secret_text not in rendered
    assert "embedding call will be retried" in rendered


# --- Classification against the real SDK error classes ---------------------------------------------------------------------


def classify(exc: BaseException) -> Any:
    return embeddings._classify(exc)


def test_real_sdk_errors_are_classified_correctly() -> None:
    from google.genai import errors

    rate = errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": RATE_LIMIT_MESSAGE}}
    )
    info = classify(wrapped_from(rate))
    assert (info.kind, info.retry_after, info.code) == ("rate_limit", 12.5, 429)

    key = errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": "API key not valid. Please pass a valid API key.",
            }
        },
    )
    assert classify(wrapped_from(key)).kind == "auth"

    server = errors.ServerError(
        503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}}
    )
    assert classify(wrapped_from(server)).kind == "transient"


def wrapped_from(inner: BaseException) -> Exception:
    outer = RuntimeError("Error embedding content")
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (wrapped(429, "RESOURCE_EXHAUSTED"), "rate_limit"),
        (wrapped(429, "RESOURCE_EXHAUSTED EmbedContentRequestsPerDay"), "quota"),
        (wrapped(403, "PERMISSION_DENIED"), "auth"),
        (wrapped(400, "INVALID_ARGUMENT bad"), "input"),
        (wrapped(500, "INTERNAL"), "transient"),
        (ConnectionError("x"), "transient"),
        (ValueError("x"), "fatal"),
    ],
)
def test_classification_table(exc: BaseException, kind: str) -> None:
    assert classify(exc).kind == kind
