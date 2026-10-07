# GroundedRAG

Enterprise document Q&A with answers you can verify. Upload PDF, DOCX, Markdown, HTML or
text files, ask questions in a chat interface, and get streamed answers grounded **only** in
those documents, each claim carrying a clickable citation (document, page, section). When the
documents do not contain the answer, the app says so instead of guessing.

> **Build status:** Phase 1 of 6 (deployment skeleton) is complete. Everything below that
> describes later phases (upload, chat, citations) is the design target in [PRD.md](PRD.md) and
> [TDD.md](TDD.md), not yet implemented.

| Document | Defines |
| --- | --- |
| [PRD.md](PRD.md) | What the product must do and why |
| [TDD.md](TDD.md) | How it is built: architecture, schemas, thresholds, configuration |
| [CLAUDE.md](CLAUDE.md) | How to work in this repo: file purposes, flows, routes, the 30 rules |

## Architecture

```
                         host :8080 (HTTP_PORT)
                                  │
   ┌──────────────────────────────▼─────────────────────────────┐
   │ edge network                                               │
   │   ┌──────────────┐        ┌───────────────────────────┐    │
   │   │  frontend    │ /api   │  backend (FastAPI,        │────┼──► Gemini API
   │   │  nginx       ├───────►│  one uvicorn worker)      │    │    (embeddings + LLM,
   │   │  :8080       │        │  :8000                    │    │     outbound HTTPS)
   │   └──────────────┘        └──────┬──────────────┬─────┘    │
   └──────────────────────────────────┼──────────────┼──────────┘
   ┌──────────────────────────────────┼──────────────┼──────────┐
   │ internal network (no route out)  │              │          │
   │                          ┌───────▼─────┐  ┌─────▼──────┐   │
   │                          │  postgres   │  │  chroma    │   │
   │                          │  :5432      │  │  :8000     │   │
   │                          └──────┬──────┘  └─────┬──────┘   │
   └─────────────────────────────────┼───────────────┼──────────┘
                              pg_data volume   chroma_data volume
                                        (backend also mounts the `uploads` volume)
```

Only nginx publishes a port. Postgres is the source of truth; Chroma (dense vectors) and the
in-memory BM25 index are derived from it and rebuildable.

## Quick start

Requirements: Docker with Compose v2. A Gemini API key is needed from Phase 3 onwards (ingestion
and chat); Phase 1 runs without one.

```bash
cp .env.example .env     # then fill in GOOGLE_API_KEY, JWT_SECRET, POSTGRES_PASSWORD
docker compose up --build
```

Open <http://localhost:8080>. The page shows the live status of the backend, Postgres and
Chroma. API docs are at <http://localhost:8080/api/docs>.

Generate the two secrets with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"   # JWT_SECRET
python -c "import secrets; print(secrets.token_hex(16))"   # POSTGRES_PASSWORD
```

**Never put a real key in `.env.example`.** That file is committed; `.env` is git-ignored.

### Services

| Service | Image | Runs as | Health check |
| --- | --- | --- | --- |
| `frontend` | `nginxinc/nginx-unprivileged:1.27-alpine` | nginx (101) | `GET /healthz` |
| `backend` | `python:3.11-slim` (built here) | `app` (10001) | `GET /api/health` |
| `chroma` | `chromadb/chroma:1.5.9` (thin wrapper in `chroma/`) | 10001 | heartbeat over `/dev/tcp` |
| `postgres` | `postgres:16` | `postgres` | `pg_isready` |

Named volumes `pg_data`, `chroma_data` and `uploads` survive `docker compose down`; use
`docker compose down -v` only when you want to delete all data.

## Development

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up   # hot reload (added in Phase 6)

docker compose exec backend pytest                 # tests (no network or API key needed)
docker compose exec backend ruff check .           # lint
docker compose exec backend ruff format --check .  # formatting
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

`backend/tests/test_pinned_versions.py` needs the repo's `chroma/` directory, which is not in the
backend image, so it is skipped inside the container. Run it with the directory mounted:

```bash
docker compose run --rm --no-deps --entrypoint pytest -v "$PWD/chroma:/chroma:ro" backend tests/test_pinned_versions.py
```

## Configuration reference

All settings are environment variables, read only by `backend/app/config.py`. Blank values in
`.env` are treated as unset. Full rationale for each is in [TDD.md §10](TDD.md).

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | none | Gemini API key. Needed from Phase 3; use a paid-tier key for confidential documents |
| `JWT_SECRET` | none, **required** | Token signing key, at least 32 characters |
| `JWT_EXPIRE_MINUTES` | `60` | Login token lifetime |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `groundedrag` / **required** / `groundedrag` | Database credentials (compose builds `DATABASE_URL` from them; keep the password URL-safe) |
| `HTTP_PORT` | `8080` | Host port for the web UI |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `EMBEDDING_DIM` | `768` | Output dimensionality (128 to 3072) |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_MAX_CONCURRENCY` | `32` / `4` | Ingestion throughput versus rate limits |
| `LLM_MODEL` | `gemini-2.5-flash` | Answer generation |
| `REWRITE_MODEL` | empty (= `LLM_MODEL`) | Follow-up question rewriting |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_THINKING_BUDGET` | `0.0` / `1024` / `0` | Generation settings |
| `RETRIEVAL_CANDIDATES` / `RETRIEVAL_TOP_K` | `30` / `5` | Candidates per channel; chunks sent to the LLM |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RELEVANCE_THRESHOLD` | `0.45` (provisional) | Refusal cutoff on cosine similarity; calibrate with the eval set |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking |
| `CHUNK_TARGET_CHARS` / `CHUNK_OVERLAP_CHARS` | `800` / `100` | Chunking |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |
| `CHROMA_COLLECTION` | `chunks` | Collection prefix; model and dimension are appended (`chunks__gemini-embedding-001__768`) |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `DATABASE_URL`, `CHROMA_HOST`, `CHROMA_PORT`, `UPLOAD_DIR` | set by compose | Only set these yourself when running the backend outside Docker |

Changing `EMBEDDING_MODEL`, `EMBEDDING_DIM` or `LLM_MODEL` requires re-running the evaluation and
recalibrating `RELEVANCE_THRESHOLD` (CLAUDE.md rule 17).

## Decisions

Deliberate choices and deviations from the spec documents. Newest phase last.

**Phase 1**

* **Synchronous SQLAlchemy with psycopg 3.** Simpler to reason about with Alembic and the
  background ingestion job; async endpoints (chat streaming) will reach the database through
  the thread pool.
* **HTTP-only Chroma client (`chromadb-client`).** The backend only talks to the server, so the
  full `chromadb` package (and its local-embedding dependencies) is not needed.
* **Chroma runs from a thin wrapper image** (`chroma/Dockerfile`) pinned to the same version as the
  client. The upstream image runs as root and has no `/data` directory, so a fresh named volume
  would be root-owned; the wrapper makes non-root operation (rule 30) work.
* **Chroma/client version match is enforced by a test, not at runtime.** The server's
  `/api/v2/version` returns a fixed `"1.0.0"` (and the in-image CLI reports a third number), so a
  runtime comparison would be permanently red. This replaces the TDD's implied startup check for
  versions; the model/dimension startup check (collection name and metadata) is unaffected and
  arrives with ingestion.
* **Two Docker networks.** `internal: true` for Postgres/Chroma/backend and `edge` for
  nginx/backend, so the data stores have no route to the internet and the web tier cannot reach
  them. The backend sits on both because it must call Gemini.
* **`/ready` checks Postgres and Chroma only for now.** The embedding-model reachability probe from
  the TDD is added with `core/embeddings.py` (Phase 3) as a cached probe, so readiness polling does
  not burn API quota.
* **`GOOGLE_API_KEY` is optional at the settings level.** The backend (and `pytest`) boots without
  it; the embedding/LLM wrappers raise a clear error at the point of use.
* **Postgres uses the Debian `postgres:16` image**, as the TDD says, rather than Alpine, to avoid
  musl collation differences.
* **Placeholder UI.** Until the React app lands in Phase 5, nginx serves a small static page
  (`frontend/static/index.html`) that shows backend, Postgres and Chroma status. `nginx.conf` is
  already in its final form (SPA fallback, `/api` proxy, SSE buffering off for `/api/chat`).
* **Test and lint tooling ships in the image by default** (`INSTALL_DEV=true`) so
  `docker compose exec backend pytest` works as documented; build with
  `--build-arg INSTALL_DEV=false` for a lean image. Tests use `httpx2` because Starlette
  deprecated plain `httpx` for its test client.
* **uvicorn's own access log is off.** It prints the full request path including the query string;
  `app/middleware.py` writes one privacy-safe JSON line per request (method, path, status, duration).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Bind for 0.0.0.0:8080 failed: port is already allocated` | Another process or container uses 8080. Set `HTTP_PORT=8081` (or any free port) in `.env`, then `docker compose up -d`. Find the owner with `docker ps` or `netstat -ano \| findstr :8080`. |
| `required variable POSTGRES_PASSWORD is missing a value` | `.env` is missing or incomplete. Run `cp .env.example .env` and fill in the required values. |
| Backend restarts with `ValidationError ... jwt_secret` | `JWT_SECRET` is empty or shorter than 32 characters. |
| Backend cannot log in to Postgres after editing `POSTGRES_PASSWORD` | The password is fixed when the `pg_data` volume is first created. Restore the old value, or reset with `docker compose down -v` (**deletes all data**). |
| `/api/ready` returns 503 | The body names the failing dependency under `checks`. Look at `docker compose ps` and `docker compose logs <service>`. |
| `exec /app/docker-entrypoint.sh: no such file or directory` | The script was checked out with Windows (CRLF) line endings. `.gitattributes` forces LF; re-checkout the file (or run `git add --renormalize .`). |
| Chroma container is unhealthy | It has no curl; the healthcheck uses bash `/dev/tcp`. Check `docker compose logs chroma` for the real error (usually a permissions problem on `chroma_data`). |
| Logs are one JSON object per line | By design (structured logging with a `request_id` on every request line). Migration output at startup is plain text. |
