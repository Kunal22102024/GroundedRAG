"""Gemini embeddings.

The only module that calls the embedding API (CLAUDE.md rule 2). It guarantees:

* chunks are embedded with ``RETRIEVAL_DOCUMENT`` and queries with ``RETRIEVAL_QUERY``,
  always passed explicitly (rule 12);
* every vector has the configured dimension and unit length: Google only returns
  unit-length vectors at the model's default 3072 dimensions, so reduced-size vectors
  are normalized here (rule 13);
* no input is long enough to be silently truncated by the model (rule 16);
* identical texts in one call are embedded once, and repeated queries come from a
  small LRU cache;
* rate limits and transient errors are retried with exponential backoff and jitter,
  honouring the server's own "retry in N seconds" hint, with bounded concurrency
  and a cooldown shared by all threads so one 429 does not trigger a burst of
  further 429s.

Error messages are written for the end user and never contain document text or keys.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypeVar

import numpy as np

from app.config import get_settings
from app.constants import EMBEDDING_INPUT_MAX_CHARS

logger = logging.getLogger(__name__)

T = TypeVar("T")

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"

QUERY_CACHE_SIZE = 256
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0
# A server-suggested wait longer than this is reported to the caller instead of blocking a worker.
_MAX_SINGLE_WAIT_SECONDS = 120.0

# Indirection so tests can run without really sleeping.
_sleep: Callable[[float], None] = time.sleep
_monotonic: Callable[[], float] = time.monotonic
_jitter: Callable[[float, float], float] = random.uniform


class EmbeddingError(Exception):
    """Embedding failed. The message is safe to show to the user."""


class EmbeddingConfigError(EmbeddingError):
    """The embedding service is not configured (no API key)."""


class EmbeddingAuthError(EmbeddingError):
    """The API key was rejected."""


class EmbeddingRateLimitError(EmbeddingError):
    """Still rate limited after all retries (or the required wait is too long)."""


class EmbeddingQuotaError(EmbeddingError):
    """A daily or billing quota is used up; retrying will not help."""


class EmbeddingInputError(EmbeddingError):
    """The text cannot be embedded (empty or too long)."""


Kind = Literal["rate_limit", "quota", "auth", "input", "transient", "fatal"]


@dataclass(frozen=True)
class _Classified:
    kind: Kind
    retry_after: float | None = None
    code: int | None = None


# --- Error classification ---------------------------------------------------------

_RETRY_IN = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_TRANSIENT_STATUSES = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "ABORTED")


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and its causes (LangChain wraps the google-genai error)."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 6:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _classify(exc: BaseException) -> _Classified:
    chain = _chain(exc)
    text = " ".join(str(e) for e in chain)
    code = next(
        (
            int(c)
            for e in chain
            for c in (getattr(e, "code", None), getattr(e, "status_code", None))
            if isinstance(c, int)
        ),
        None,
    )

    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        if re.search(r"PerDay|per_day|daily", text, re.IGNORECASE):
            return _Classified("quota", code=code)
        hint = _RETRY_IN.search(text) or _RETRY_DELAY.search(text)
        return _Classified("rate_limit", float(hint.group(1)) if hint else None, code)
    if (
        code in (401, 403)
        or "API key not valid" in text
        or "API_KEY_INVALID" in text
        or "PERMISSION_DENIED" in text
        or "UNAUTHENTICATED" in text
    ):
        return _Classified("auth", code=code)
    if code == 400 or "INVALID_ARGUMENT" in text:
        return _Classified("input", code=code)
    if (code is not None and code >= 500) or any(s in text for s in _TRANSIENT_STATUSES):
        return _Classified("transient", code=code)
    for e in chain:
        module = type(e).__module__
        if isinstance(e, (TimeoutError, ConnectionError, OSError)) or module.startswith(
            ("httpx", "httpcore", "httpx2")
        ):
            return _Classified("transient", code=code)
    return _Classified("fatal", code=code)


# --- Client, rate-limit cooldown and cache ---------------------------------------------


@lru_cache
def _client() -> Any:
    """The LangChain Gemini embeddings client, built on first use."""
    settings = get_settings()
    api_key = settings.google_api_key.get_secret_value()
    if not api_key:
        raise EmbeddingConfigError("GOOGLE_API_KEY is not set, so documents cannot be embedded.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=api_key)


_cooldown_lock = threading.Lock()
_cooldown_until = 0.0


def _wait_for_cooldown() -> None:
    with _cooldown_lock:
        remaining = _cooldown_until - _monotonic()
    if remaining > 0:
        _sleep(remaining)


def _start_cooldown(seconds: float) -> None:
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, _monotonic() + seconds)


_cache_lock = threading.Lock()
_query_cache: OrderedDict[str, list[float]] = OrderedDict()


def clear_caches() -> None:
    """Forget the client, the query cache and any rate-limit cooldown (used by tests)."""
    global _cooldown_until
    _client.cache_clear()
    with _cache_lock:
        _query_cache.clear()
    with _cooldown_lock:
        _cooldown_until = 0.0


# --- Calls with retries ------------------------------------------------------------------


def _with_retries(call: Callable[[], T], max_retries: int | None = None) -> T:
    """Run ``call``, retrying rate limits and transient failures with backoff.

    Non-retryable problems (bad key, bad input, exhausted quota) raise immediately.
    """
    retries = get_settings().embedding_max_retries if max_retries is None else max_retries
    attempt = 0
    while True:
        _wait_for_cooldown()
        try:
            return call()
        except EmbeddingError:
            raise
        except Exception as exc:
            info = _classify(exc)
            if info.kind == "auth":
                raise EmbeddingAuthError(
                    "The embedding service rejected the API key. Check GOOGLE_API_KEY."
                ) from exc
            if info.kind == "quota":
                raise EmbeddingQuotaError(
                    "The embedding quota is exhausted. Wait for it to reset or enable billing."
                ) from exc
            if info.kind == "input":
                raise EmbeddingInputError("The embedding service rejected the input.") from exc
            if info.kind == "fatal":
                raise EmbeddingError(f"Embedding failed ({type(exc).__name__}).") from exc

            attempt += 1
            if info.retry_after is not None:
                delay = info.retry_after + 1.0
            else:
                delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
            delay *= _jitter(1.0, 1.25)
            too_long = delay > _MAX_SINGLE_WAIT_SECONDS
            if attempt > retries or too_long:
                if info.kind == "rate_limit":
                    raise EmbeddingRateLimitError(
                        "The embedding service is rate limited (free-tier keys allow about 100 "
                        "requests per minute). Try again later or use a billing-enabled key."
                    ) from exc
                raise EmbeddingError(
                    "The embedding service is unavailable. Try again later."
                ) from exc
            if info.kind == "rate_limit":
                _start_cooldown(delay)
            logger.warning(
                "embedding call will be retried",
                extra={
                    "kind": info.kind,
                    "code": info.code,
                    "attempt": attempt,
                    "wait_s": round(delay, 1),
                },
            )
            _sleep(delay)


def _normalize(vectors: Sequence[Sequence[float]], expected_dim: int) -> list[list[float]]:
    """Check dimensions and scale every vector to unit length."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise EmbeddingError(
            f"The embedding service returned vectors of an unexpected size (expected {expected_dim})."
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise EmbeddingError("The embedding service returned an empty vector.")
    return (array / norms).tolist()


def _check_text(text: str, index: int | None = None) -> None:
    where = f" (item {index})" if index is not None else ""
    if not text or not text.strip():
        raise EmbeddingInputError(f"Cannot embed empty text{where}.")
    if len(text) > EMBEDDING_INPUT_MAX_CHARS:
        raise EmbeddingInputError(
            f"Text{where} is longer than {EMBEDDING_INPUT_MAX_CHARS} characters and would be truncated."
        )


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed chunk texts with ``RETRIEVAL_DOCUMENT``; returns unit-length vectors in input order."""
    if not texts:
        return []
    for index, text in enumerate(texts):
        _check_text(text, index)
    settings = get_settings()

    unique = list(dict.fromkeys(texts))  # identical chunks are embedded once
    size = settings.embedding_batch_size
    batches = [unique[i : i + size] for i in range(0, len(unique), size)]
    client = _client()
    dimension = settings.embedding_dim

    def run(batch: list[str]) -> list[list[float]]:
        raw = _with_retries(
            lambda: client.embed_documents(
                batch,
                batch_size=len(batch),
                task_type=DOCUMENT_TASK,
                output_dimensionality=dimension,
            )
        )
        if len(raw) != len(batch):
            raise EmbeddingError("The embedding service returned the wrong number of vectors.")
        return _normalize(raw, dimension)

    if len(batches) == 1:
        results = [run(batches[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=settings.embedding_max_concurrency, thread_name_prefix="embed"
        ) as pool:
            futures = [pool.submit(run, batch) for batch in batches]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    by_text = {
        text: vector
        for batch, vectors in zip(batches, results, strict=True)
        for text, vector in zip(batch, vectors, strict=True)
    }
    return [list(by_text[text]) for text in texts]


def embed_query(text: str) -> list[float]:
    """Embed a question with ``RETRIEVAL_QUERY``; repeated questions are served from an LRU cache."""
    _check_text(text)
    with _cache_lock:
        cached = _query_cache.get(text)
        if cached is not None:
            _query_cache.move_to_end(text)
            return list(cached)

    settings = get_settings()
    client = _client()
    raw = _with_retries(
        lambda: client.embed_query(
            text, task_type=QUERY_TASK, output_dimensionality=settings.embedding_dim
        )
    )
    vector = _normalize([raw], settings.embedding_dim)[0]
    with _cache_lock:
        _query_cache[text] = vector
        while len(_query_cache) > QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return list(vector)


def probe() -> str:
    """One cheap real call to prove the service is reachable and the key works.

    Single attempt (no retries) so readiness checks answer quickly. Returns ``"ok"``,
    or ``"rate_limited"`` when the service answered but is throttling us (the key is
    valid and the service is up). Raises ``EmbeddingError`` subclasses otherwise.
    """
    settings = get_settings()
    client = _client()
    try:
        raw = _with_retries(
            lambda: client.embed_query(
                "readiness probe",
                task_type=QUERY_TASK,
                output_dimensionality=settings.embedding_dim,
            ),
            max_retries=0,
        )
    except EmbeddingRateLimitError:
        return "rate_limited"
    _normalize([raw], settings.embedding_dim)
    return "ok"
