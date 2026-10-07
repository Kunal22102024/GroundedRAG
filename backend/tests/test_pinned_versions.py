"""CLAUDE.md rule 28: the ChromaDB image and client library are pinned together.

The server's /version endpoint reports a fixed API version, so a runtime check is
impossible (see app/core/vectorstore.py). This test enforces the pin statically.
It needs the repository's ``chroma/`` directory, which is not part of the backend
image, so it is skipped when run inside the container.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
CHROMA_DOCKERFILE = BACKEND_DIR.parent / "chroma" / "Dockerfile"


def test_chroma_image_tag_matches_client_pin() -> None:
    if not CHROMA_DOCKERFILE.is_file():
        pytest.skip("chroma/Dockerfile not available (running inside the backend image)")

    requirements = (BACKEND_DIR / "requirements.txt").read_text()
    image = re.search(
        r"^FROM\s+chromadb/chroma:(\S+)\s*$", CHROMA_DOCKERFILE.read_text(), re.MULTILINE
    )
    client = re.search(r"^chromadb-client==(\S+?)(?:\s|$)", requirements, re.MULTILINE)

    assert image, "chroma/Dockerfile must start FROM chromadb/chroma:<exact version>"
    assert client, "requirements.txt must pin chromadb-client==<exact version>"
    assert image.group(1) == client.group(1)
