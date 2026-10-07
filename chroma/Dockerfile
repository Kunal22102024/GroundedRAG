# Thin wrapper around the official ChromaDB server image.
#
# Why a wrapper: the upstream image runs as root and its data directory (/data)
# does not exist in the image, so a named volume mounted there would be root
# owned. Creating /data owned by an unprivileged user lets the server run
# non-root (CLAUDE.md rule 30) and lets Docker copy that ownership into a fresh
# named volume.
#
# The version here MUST match chromadb-client in backend/requirements.txt
# (CLAUDE.md rule 28); backend/tests/test_pinned_versions.py enforces it. The
# server's own /version endpoint reports a fixed API version, so it cannot be
# used to verify this at runtime.
FROM chromadb/chroma:1.5.9

RUN mkdir -p /data && chown 10001:10001 /data

USER 10001:10001
