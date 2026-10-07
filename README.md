# GroundedRAG

Enterprise document Q&A with answers you can verify. Upload PDF, DOCX, Markdown, HTML or
text files, ask questions in a chat interface, and get streamed answers grounded **only** in
those documents, each claim carrying a clickable citation (document, page, section). When the
documents do not contain the answer, the app says so instead of guessing.

> **Build status:** Phases 1 to 3 of 6 are complete (deployment skeleton; database and
> authentication; document ingestion and the documents API). Chat with citations, the web UI and
> the evaluation tooling are the design target in [PRD.md](PRD.md) and [TDD.md](TDD.md), not yet
> implemented.

| Document | Defines |
| --- | --- |
| [PRD.md](PRD.md) | What the product must do and why |
| [TDD.md](TDD.md) | How it is built: architecture, schemas, thresholds, configuration |
| [CLAUDE.md](CLAUDE.md) | How to work in this repo: file purposes, flows, routes, the 30 rules |

## Architecture

```
                         host :8081 (HTTP_PORT)
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

Open <http://localhost:8081>. The page shows the live status of the backend, Postgres and
Chroma. API docs are at <http://localhost:8081/api/docs>.

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

## Accounts and roles

Anyone can sign up (`POST /api/auth/signup`). **The first account ever created becomes `admin`;
every later one is a `user`**, so create your own account first. Roles decide which documents a
person may read once uploads exist (PRD 4.4):

| Role | Reads | May assign `Restricted` on upload | May delete |
| --- | --- | --- | --- |
| `admin` | Public, Internal, Restricted | yes | any document |
| `user` | Public, Internal | no | their own documents |

Passwords are 8 to 72 bytes. Tokens last `JWT_EXPIRE_MINUTES` (default 60) and are sent as
`Authorization: Bearer <token>`. Try it from the interactive docs at `/api/docs` ("Authorize"), or:

```bash
curl -s -X POST localhost:8081/api/auth/signup -H 'Content-Type: application/json' \n     -d '{"email": "you@company.com", "password": "a long passphrase"}'
curl -s localhost:8081/api/auth/me -H "Authorization: Bearer $TOKEN"
```

There is no user-management screen yet: to promote someone, update the database directly
(`docker compose exec postgres psql -U groundedrag -c "UPDATE users SET role='admin' WHERE email='...'"`).
The change applies immediately, even to tokens issued before it.

## Documents

Supported formats: PDF, DOCX, Markdown (`.md`, `.markdown`), HTML (`.html`, `.htm`) and plain text,
up to `MAX_UPLOAD_MB` each. Upload one or many files in a single request (needs a signed-in user and
a `GOOGLE_API_KEY`, because indexing calls Gemini):

```bash
curl -s -X POST localhost:8081/api/documents -H "Authorization: Bearer $TOKEN" \
     -F "files=@policy.pdf" -F "files=@handbook.docx" -F "department=HR" -F "document_type=Policy"
curl -s "localhost:8081/api/documents?status=failed" -H "Authorization: Bearer $TOKEN"
curl -s -OJ localhost:8081/api/documents/<id>/file -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE localhost:8081/api/documents/<id> -H "Authorization: Bearer $TOKEN"
```

The upload returns `202` at once; files then move `queued → processing → indexed` (or `failed` with an
`error_message`) in the background, so poll `GET /api/documents`. Files are accepted or refused one by
one: refused files come back in `rejected` with a reason and never block the others.

**What is checked on upload:** extension, declared content type, the file's actual bytes (a `.pdf`
must really be a PDF, a `.docx` really a zip with `word/document.xml`, text must not be binary) and
size. Originals are stored under generated names, never under the name you uploaded.

**Metadata** is worked out while indexing. Anything you pass wins; otherwise the folder or file name
(`IT/Policy_2026-01-15.pdf`), then the file's own properties, then the upload date:

| Field | From the file name or folder | From the file |
| --- | --- | --- |
| `department` | a known department as a folder or first word (HR, Engineering, Legal, Operations, IT) | Markdown front matter |
| `document_type` | words such as policy/standard, manual/handbook, spec/rfc, contract/agreement | title, front matter |
| `creation_date` | `2026-01-15`, `20260115`, `April2025`, `2026` (a month or year is refined to the exact day by a consistent PDF date) | PDF creation date, DOCX/HTML/Markdown date |
| `access_level` | defaults to `Internal`; `Restricted` only for admins | |

**What happens to problem files:** a scanned or blank PDF fails with *"This looks like a scanned PDF,
OCR not supported"*; an empty, corrupt or password-protected file fails with a plain explanation. A
failed document leaves nothing in the search index and stays listed so you can see why and delete it.

**Only PDFs are shown inline** when you open an original; every other type downloads. Opening an
uploaded HTML page on the app's own origin would let it run script next to your session token.

**A deleted document is gone everywhere**: search index, vector store, database and file. If a step
fails, the document is parked as `failed` (already unsearchable) and you simply delete it again.

### Free-tier Gemini keys

A free-tier key allows about **100 embedding requests per minute** (every chunk counts), and may have
a daily cap. Indexing backs off and resumes automatically, but it runs at roughly 100 chunks a minute:
a 300-page manual is on the order of 15 minutes. A billing-enabled key removes the limit (and is the
right choice for confidential documents, TDD section 9). `GET /api/ready` reports whether the key works.

### Checking the search index

Chroma can occasionally hold a vector that its index never returns, so a document looks indexed but
searches miss it. The backend checks every new chunk before marking a document indexed and re-checks
everything at startup, repairing what it finds without calling Gemini. To check on demand:

```bash
docker compose exec backend python -m app.maintenance verify-index
```

### Recovering the search index

The backend **refuses to start** if the vector store does not match the database, instead of quietly
answering from an empty index. That happens after changing `EMBEDDING_MODEL`/`EMBEDDING_DIM` (the
collection name encodes both) or losing the Chroma volume. The log says so; fix it by rebuilding the
vectors from the database (resumable, so a rate limit or quota stop just means running it again):

```bash
docker compose run --rm --no-deps --entrypoint python backend -m app.maintenance reembed
docker compose up -d backend
```

## Development

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up   # hot reload (added in Phase 6)

docker compose exec backend pytest                 # tests (no API key needed; see the note below)
docker compose exec backend ruff check .           # lint
docker compose exec backend ruff format --check .  # formatting
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

Database tests create a throwaway database (`groundedrag_test_<random>`) on the running Postgres
container, migrate it with the real Alembic migrations, and drop it afterwards; your real data is
never touched. They are skipped when no Postgres is reachable, so run the tests inside the stack.

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
| `BCRYPT_ROUNDS` | `12` | Password hashing cost (4 to 15); about 250 ms per hash at 12 |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `groundedrag` / **required** / `groundedrag` | Database credentials (compose builds `DATABASE_URL` from them; keep the password URL-safe) |
| `HTTP_PORT` | `8081` | Host port for the web UI (nginx listens on 8080 inside the container) |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `EMBEDDING_DIM` | `768` | Output dimensionality (128 to 3072) |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_MAX_CONCURRENCY` | `32` / `4` | Ingestion throughput versus rate limits |
| `EMBEDDING_MAX_RETRIES` | `6` | Retries per batch on rate limits and transient errors (backoff honours the server's retry hint) |
| `LLM_MODEL` | `gemini-2.5-flash` | Answer generation |
| `REWRITE_MODEL` | empty (= `LLM_MODEL`) | Follow-up question rewriting |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_THINKING_BUDGET` | `0.0` / `1024` / `0` | Generation settings |
| `RETRIEVAL_CANDIDATES` / `RETRIEVAL_TOP_K` | `30` / `5` | Candidates per channel; chunks sent to the LLM |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RELEVANCE_THRESHOLD` | `0.45` (provisional) | Refusal cutoff on cosine similarity; calibrate with the eval set |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking |
| `CHUNK_TARGET_CHARS` / `CHUNK_OVERLAP_CHARS` | `800` / `100` | Chunking |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |
| `INGESTION_WORKERS` | `1` | Documents indexed at the same time |
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

**Phase 2**

* **UUID primary keys generated by the database**, `VARCHAR` + named `CHECK` constraints instead of
  PostgreSQL ENUM types, and cascades as `ON DELETE` in the schema (details in TDD 4.3). Deleting a
  document therefore cannot leave chunk rows behind, whatever code path performs the delete.
* **The migration is written by hand** and a test proves it equals the models (including `CHECK`
  constraints, which Alembic's autogenerate cannot compare). Run it after every model change.
* **Documents survive their uploader.** `documents.owner_id` becomes `NULL` if the user is deleted
  (shared knowledge outlives an account); a user's conversations and feedback are deleted with them.
* **Open signup, first account is admin.** This follows the spec. The "first user" decision is made
  under a PostgreSQL advisory lock, so two simultaneous first signups cannot both become admin.
* **Signup and login both return a token and the user**, so the UI needs one call, not two.
* **The role is not stored in the token.** It is re-read from the database on every request, so a
  demotion or deletion takes effect immediately instead of when the token expires. The cost is one
  indexed primary-key lookup per request.
* **Emails are validated loosely** (`something@something.tld`) rather than with an email library, so
  internal domains such as `company.local` are accepted. They are stored lower-cased.
* **Passwords above 72 bytes are rejected** (bcrypt would otherwise ignore the excess, and bcrypt 5
  refuses it). The limit is in bytes, so 37 accented characters already exceed it.
* **Not implemented, by scope:** login rate limiting or lockout, password reset, email
  verification, and user management. A shared-intranet deployment should add throttling at nginx.
* **Auth handlers are plain `def`**, not `async`, because bcrypt is CPU-bound; FastAPI runs them in
  a worker thread so the event loop stays free for streaming.
* **Test database per session, not per test.** Tables are truncated between tests; a test refuses to
  run if the connected database name does not start with `groundedrag_test_`.

**Phase 3**

* **Metadata is resolved during indexing**, as the PRD says, so `documents.creation_date` is `NULL` until
  then (migration `0002`) instead of holding a made-up placeholder.
* **Page numbers are physical PDF pages**, so `#page=N` opens the cited page. Formats without pages
  report page 1; the UI will hide the page for them.
* **Two-column PDFs are read column by column** (TDD 3.1). Without it, side-by-side text is stitched
  into lines that mix two sentences. Tables are deliberately not split.
* **Page chrome is stripped** (repeated headers, footers, banners, bare page labels), with a safety
  net that skips the removal if it would delete more than 30% of a document's text.
* **Chunk overlap is whole sentences on the same page**, so a chunk's page range always describes
  exactly its text. A hard size ceiling (1500 characters) keeps every chunk far under the embedding
  model's 2,048-token limit, validated at startup.
* **`bm25s` rather than `rank_bm25`** for the keyword index (maintained, faster, filter mask support).
  It cannot add documents incrementally, so each update rebuilds from stored tokens; searches use an
  immutable snapshot and never wait for an ingestion.
* **The backend refuses to start on an inconsistent index** (Postgres has indexed chunks, Chroma is
  empty), because serving from an empty index would silently return nothing. `reembed` repairs it.
* **Indexing runs one document at a time** in an in-process pool, so a free-tier key's rate limit is
  shared sensibly. A restart fails documents that were mid-indexing (re-queueing could loop forever on
  a document that crashes the process) and resubmits ones that never started.
* **Delete parks the document as `failed` first**, then removes it from BM25, Chroma, Postgres and disk.
  This differs slightly from the order in CLAUDE.md, and gives a stronger guarantee: it is excluded
  from retrieval immediately, whichever later step fails.
* **Documents above your access level answer 404**, never 403, so their existence is not revealed.
* **The embedding readiness probe is cached** (5 minutes when healthy, 30 seconds when failing)
  because every probe spends one request out of a free-tier 100 per minute.
* **The vector index is verified, not trusted.** ChromaDB 1.5.9 was seen to store vectors that no search
  returns (15 of 155 after deleting and re-uploading the same files). The trigger could not be
  reproduced, so every write is checked by querying each new vector with its own embedding, with automatic
  repair (re-insert, or rebuild from the stored vectors; no API cost), startup verification, and a
  `verify-index` command. Re-test this on every Chroma upgrade (TDD 4.1).
* **The provisional `RELEVANCE_THRESHOLD=0.45` is too low for this model.** Measured on a small corpus (62
  chunks): the best chunk for a relevant question scores 0.76 to 0.82 cosine, and the best chunk for an
  unrelated question 0.56 at most, so 0.45 would never refuse. A cutoff near 0.65 separates these
  samples, but the corpus is tiny: calibrate with the evaluation set (CLAUDE.md rule 17) before relying on it.
* **Known limitation:** PDFs whose text layer overprints other text (a watermark laid over the body)
  can interleave characters. The document still indexes and its page numbers are right.
* **Not implemented, by scope:** OCR, re-ingesting a failed document in place (delete and upload
  again), and per-user storage quotas.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Bind for 0.0.0.0:8081 failed: port is already allocated` | Another process or container uses host port 8081. Set `HTTP_PORT=8082` (or any free port) in `.env`, then `docker compose up -d`. Find the owner with `docker ps` or `netstat -ano \| findstr :8081`. |
| `required variable POSTGRES_PASSWORD is missing a value` | `.env` is missing or incomplete. Run `cp .env.example .env` and fill in the required values. |
| Backend restarts with `ValidationError ... jwt_secret` | `JWT_SECRET` is empty or shorter than 32 characters. |
| Backend cannot log in to Postgres after editing `POSTGRES_PASSWORD` | The password is fixed when the `pg_data` volume is first created. Restore the old value, or reset with `docker compose down -v` (**deletes all data**). |
| My account is not an admin | Someone else signed up first. Promote yourself with the `UPDATE users SET role='admin' ...` command in "Accounts and roles". |
| `401` right after logging in | Check the header is exactly `Authorization: Bearer <token>`, the token has not expired (`expires_in` seconds), and `JWT_SECRET` has not changed since it was issued (changing it invalidates every token). |
| Database tests are all skipped | No Postgres was reachable. Run `docker compose exec backend pytest` with the stack up. |
| A document is `failed` with "rate limited" or "quota" | A free-tier Gemini key allows about 100 embedding requests per minute (and may have a daily cap). Wait and upload again, or use a billing-enabled key. The error message and the logs say which limit was hit. |
| A document is `failed` with "rejected the API key" | `GOOGLE_API_KEY` is wrong or revoked. `GET /api/ready` shows `embeddings` as not ok. |
| Backend exits at startup: "PostgreSQL holds N indexed chunks but the Chroma collection ... is empty" | The embedding settings changed or the Chroma volume was lost. Run the `reembed` command in "Recovering the search index", or restore the old settings. |
| Backend exits at startup: "does not match the configured embedding setup" | A collection with the current name exists but was built with a different model, dimension or distance. Restore the settings, or remove that collection and run `reembed`. |
| A document is `indexed` but searches never find it | The vector index lost it. Run `docker compose exec backend python -m app.maintenance verify-index` (see "Checking the search index"). A log line `vectors stored but unreachable by search; repairing` at startup means this was found and fixed. |
| Deleting a document answers 409 | It is being indexed right now. Delete it again when it finishes. |
| Deleting answers 503 | A step failed. The document is now `failed` (unsearchable); delete it again to finish. |
| `/api/ready` returns 503 | The body names the failing dependency under `checks`. Look at `docker compose ps` and `docker compose logs <service>`. |
| `exec /app/docker-entrypoint.sh: no such file or directory` | The script was checked out with Windows (CRLF) line endings. `.gitattributes` forces LF; re-checkout the file (or run `git add --renormalize .`). |
| Chroma container is unhealthy | It has no curl; the healthcheck uses bash `/dev/tcp`. Check `docker compose logs chroma` for the real error (usually a permissions problem on `chroma_data`). |
| Logs are one JSON object per line | By design (structured logging with a `request_id` on every request line). Migration output at startup is plain text. |
