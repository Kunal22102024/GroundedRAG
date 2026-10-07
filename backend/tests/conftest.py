"""Shared test setup.

Tests never need network access or a Gemini key: the LLM and embeddings are
mocked in the tests that use them. Settings are forced to harmless dummy values
*before* any ``app`` module is imported, so a test can never reach the real
database even when run inside the Docker stack (where real values exist).
"""

from __future__ import annotations

import logging
import os

import pytest

# Force (not setdefault): inside the container the real DATABASE_URL is set.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@127.0.0.1:1/test"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret-0123456789"
os.environ["GOOGLE_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "INFO"


class LogCollector(logging.Handler):
    """Collects log records emitted on the logger it is attached to."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def request_logs() -> LogCollector:
    """Capture records from the ``app.request`` access logger."""
    collector = LogCollector()
    logger = logging.getLogger("app.request")
    previous_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    yield collector
    logger.removeHandler(collector)
    logger.setLevel(previous_level)
