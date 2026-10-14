"""The container entrypoint keeps the backend at ONE worker (CLAUDE.md rule 4).

The BM25 index lives in process memory: a second worker would have its own copy that never sees
uploads or deletions made through the first, and its startup would fail the first one's running
ingestions. uvicorn reads WEB_CONCURRENCY when --workers is absent, so the entrypoint both pins
``--workers 1`` and refuses to start when such a variable asks for more.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")


def run(**env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if k not in {"WEB_CONCURRENCY", "UVICORN_WORKERS"}}
    return subprocess.run(
        ["sh", str(ENTRYPOINT)],
        env={**clean, **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("variable", ["WEB_CONCURRENCY", "UVICORN_WORKERS"])
@pytest.mark.parametrize("value", ["2", "4", "auto"])
def test_more_than_one_worker_is_refused_before_anything_starts(variable: str, value: str) -> None:
    result = run(**{variable: value})
    assert result.returncode == 1
    assert f"{variable}={value}" in result.stderr and "ONE worker" in result.stderr
    assert "alembic" not in result.stderr.lower()  # it stopped before the migrations


def test_the_server_is_started_with_exactly_one_worker() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "--workers 1" in text and "exec uvicorn" in text
    assert "--reload" not in text
