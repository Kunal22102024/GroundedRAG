"""Classify errors raised by the Gemini client (embeddings and chat share this).

LangChain wraps the ``google-genai`` error, so the classifier walks the exception chain and
looks at status codes and message text. The server's own retry hint ("Please retry in 12s")
is extracted so callers can wait exactly as long as asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["rate_limit", "quota", "auth", "input", "transient", "fatal"]


@dataclass(frozen=True)
class Classified:
    """What kind of failure this was, the server's retry hint in seconds, and the HTTP code."""

    kind: Kind
    retry_after: float | None = None
    code: int | None = None


_RETRY_IN = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_TRANSIENT_STATUSES = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "ABORTED")


def chain(exc: BaseException) -> list[BaseException]:
    """The exception and its causes (LangChain wraps the google-genai error)."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 6:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def classify(exc: BaseException) -> Classified:
    """Sort a Gemini client error into a retry decision."""
    errors = chain(exc)
    text = " ".join(str(e) for e in errors)
    code = next(
        (
            int(c)
            for e in errors
            for c in (getattr(e, "code", None), getattr(e, "status_code", None))
            if isinstance(c, int)
        ),
        None,
    )

    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        if re.search(r"PerDay|per_day|daily", text, re.IGNORECASE):
            return Classified("quota", code=code)
        hint = _RETRY_IN.search(text) or _RETRY_DELAY.search(text)
        return Classified("rate_limit", float(hint.group(1)) if hint else None, code)
    if (
        code in (401, 403)
        or "API key not valid" in text
        or "API_KEY_INVALID" in text
        or "PERMISSION_DENIED" in text
        or "UNAUTHENTICATED" in text
    ):
        return Classified("auth", code=code)
    if code == 400 or "INVALID_ARGUMENT" in text:
        return Classified("input", code=code)
    if (code is not None and code >= 500) or any(s in text for s in _TRANSIENT_STATUSES):
        return Classified("transient", code=code)
    for e in errors:
        module = type(e).__module__
        if isinstance(e, (TimeoutError, ConnectionError, OSError)) or module.startswith(
            ("httpx", "httpcore", "httpx2")
        ):
            return Classified("transient", code=code)
    return Classified("fatal", code=code)
