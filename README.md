# GroundedRAG

Enterprise document Q&A with answers you can verify. Upload PDF, DOCX, Markdown, HTML or
text files, ask questions in a chat interface, and get streamed answers grounded **only** in
those documents, each claim carrying a clickable citation (document, page, section). When the
documents do not contain the answer, the app says so instead of guessing.

> **Build status:** all six phases are complete: deployment, database and authentication, document
> ingestion, hybrid retrieval with the relevance guard, streamed chat with verified citations, the
> web UI and the evaluation script. A later audit pass fixed correctness, security and test gaps
> (see "Decisions", *Audit pass*). The design is in [PRD.md](PRD.md) and [TDD.md](TDD.md).

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

Requirements: Docker with Compose v2 and a Gemini API key (ingestion and chat call Gemini).

```bash
cp .env.example .env     # then fill in GOOGLE_API_KEY, JWT_SECRET, POSTGRES_PASSWORD
docker compose up --build
```

Open <http://localhost:8081>, then:

1. **Sign up.** The first account becomes the administrator (only an admin can mark a document
   `Restricted`). The app listens on this machine only (`BIND_ADDR=127.0.0.1`). Before you share it
   on a network, create your account, then set `SIGNUP_ENABLED=false` (see "Accounts and roles").
2. **Library:** upload PDF, DOCX, Markdown, HTML or text files. Each moves `queued`, `processing`,
   `indexed`; department, type and date are detected unless you set them. Delete removes the file, its
   text and its search index.
3. **Chat:** ask a question. The answer streams in with citation chips (document and page); click one to
   see the exact passage and the original PDF opened at that page. Use the filters above the chat
   (department, type, date range, specific documents) to narrow the search. If the documents do not
   contain the answer you get "Insufficient information available in the selected documents.".

API docs are at <http://localhost:8081/api/docs>.

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

Anyone who can reach the app can sign up (`POST /api/auth/signup`). **The first account ever created
becomes `admin`; every later one is a `user`**, so create your own account first. Once the people
who need one have an account, set `SIGNUP_ENABLED=false` in `.env` and recreate the backend
(`docker compose up -d backend`): signup then answers 403, while signing in is unaffected. (A fresh
install with no account at all can always create its first one, so it can never lock you out.)
Roles decide which documents a person may read once uploads exist (PRD 4.4):

| Role | Reads | May assign `Restricted` on upload | May delete |
| --- | --- | --- | --- |
| `admin` | Public, Internal, Restricted | yes | any document |
| `user` | Public, Internal | no | their own documents |

Passwords are 8 to 72 bytes. Tokens last `JWT_EXPIRE_MINUTES` (default 60) and are sent as
`Authorization: Bearer <token>`. Try it from the interactive docs at `/api/docs` ("Authorize"), or:

```bash
curl -s -X POST localhost:8081/api/auth/signup -H 'Content-Type: application/json' \
     -d '{"email": "you@company.com", "password": "a long passphrase"}'   # the reply holds access_token
curl -s -X POST localhost:8081/api/auth/login -H 'Content-Type: application/json' \
     -d '{"email": "you@company.com", "password": "a long passphrase"}'   # same reply, for an existing account
curl -s localhost:8081/api/auth/me -H "Authorization: Bearer $TOKEN"
```

nginx limits how fast sign-in and sign-up can be tried (`429 Too many requests` with a
`Retry-After` header) and how large their bodies can be. It does not lock accounts, and it cannot
tell clients apart behind Docker Desktop's network, so treat it as a brake on scripts and put
real exposure behind a reverse proxy with TLS.

There is no user-management screen yet: to promote someone, update the database directly
(`docker compose exec postgres psql -U groundedrag -c "UPDATE users SET role='admin' WHERE email='...'"`).
The change applies immediately, even to tokens issued before it.

## Documents

Supported formats: PDF, DOCX, Markdown (`.md`, `.markdown`), HTML (`.html`, `.htm`) and plain text,
up to `MAX_UPLOAD_MB` each. Upload one or many files in a single request (at most
`MAX_FILES_PER_REQUEST`; needs a signed-in user and a `GOOGLE_API_KEY`, because indexing calls
Gemini). nginx checks the sign-in before it accepts the body, and refuses a request over 256 MB:

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

A `creation_date` you pass must be in 1990 or later and at most a year ahead; anything else is
refused with 422 instead of being quietly replaced by a date from another source. A document whose
extracted text is longer than `MAX_DOCUMENT_CHARS` (5 million) fails with a message saying so.
| `access_level` | defaults to `Internal`; `Restricted` only for admins | |

**What happens to problem files:** a scanned or blank PDF fails with *"This looks like a scanned PDF,
OCR not supported"*; an empty, corrupt or password-protected file fails with a plain explanation. A
failed document leaves nothing in the search index and stays listed so you can see why and delete it.

**Only PDFs are shown inline** when you open an original; every other type downloads. Opening an
uploaded HTML page on the app's own origin would let it run script next to your session token.

**A deleted document is gone everywhere**: search index, vector store, database, file, and the
passages copied into saved conversations. If a step fails, the document is parked as `failed`
(already unsearchable) and you simply delete it again. Old answers keep their text as a record of
the conversation, but their citation chips for that document read "source no longer available" and
open nothing. The same applies to a citation of a document the reader may no longer see (for
example after an administrator is demoted).

### Free-tier Gemini keys

A free-tier key allows about **100 embedding requests per minute** (every chunk counts), and may have
a daily cap. Indexing spaces its requests out (`EMBEDDING_TEXTS_PER_MINUTE`, default 90, so chat
questions keep working during a large upload), backs off on a 429 and resumes automatically, but it
runs at roughly 90 chunks a minute: a 300-page manual is on the order of 15 minutes. A
billing-enabled key removes the limit (raise `EMBEDDING_TEXTS_PER_MINUTE`, or set it to 0) and is the
right choice for confidential documents, TDD section 9. `GET /api/ready` reports whether the key works.

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
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d   # mounted source, backend reloads on change
(cd frontend && npm install && npm run dev)        # UI on :5173, proxies /api to localhost:8081

docker compose exec backend pytest                 # tests (no API key needed; see the note below)
docker compose exec backend ruff check .           # lint
docker compose exec backend ruff format --check .  # formatting
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec backend python -m eval.run_eval --verbose
                                                   # retrieval + guard over eval/questions.jsonl (embedding calls only, no LLM); see eval/README.md
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

If `HTTP_PORT` is not 8081, start the UI dev server with
`VITE_PROXY_TARGET=http://localhost:<port> npm run dev`.

Database tests create a throwaway database (`groundedrag_test_<random>`) on the running Postgres
container, migrate it with the real Alembic migrations, and drop it afterwards; your real data is
never touched. They are skipped when no Postgres is reachable, so run the tests inside the stack, and
set `REQUIRE_STACK=1` (as CI should) to make an unreachable Postgres or Chroma a **failure** rather
than a skip: a run in which every database test skipped proves nothing. The tests pin every setting
to its documented default, so a `.env` override cannot change their outcome.

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
| `GOOGLE_API_KEY` | none | Gemini API key, needed to index documents and to chat; use a paid-tier key for confidential documents |
| `JWT_SECRET` | none, **required** | Token signing key, at least 32 characters |
| `JWT_EXPIRE_MINUTES` | `60` | Login token lifetime (1 to 10080) |
| `BCRYPT_ROUNDS` | `12` | Password hashing cost (4 to 15); about 250 ms per hash at 12 |
| `SIGNUP_ENABLED` | `true` | `false` closes signup once the first account exists |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `groundedrag` / **required** / `groundedrag` | Database credentials (compose builds `DATABASE_URL` from them; keep the password URL-safe) |
| `HTTP_PORT` | `8081` | Host port for the web UI (nginx listens on 8080 inside the container) |
| `BIND_ADDR` | `127.0.0.1` | Address that port is published on; `0.0.0.0` shares the app on your network (no TLS: see "Accounts and roles") |
| `BACKEND_MEM_LIMIT` / `CHROMA_MEM_LIMIT` / `POSTGRES_MEM_LIMIT` / `FRONTEND_MEM_LIMIT` | `3g` / `2g` / `1g` / `256m` | Container memory ceilings |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` / `DB_POOL_TIMEOUT_SECONDS` | `10` / `10` / `30` | Postgres connections the backend may hold, and how long a request waits for one |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `EMBEDDING_DIM` | `768` | Output dimensionality (128 to 3072) |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_MAX_CONCURRENCY` | `32` / `4` | Ingestion throughput versus rate limits |
| `EMBEDDING_MAX_RETRIES` | `6` | Retries per batch on rate limits and transient errors (backoff honours the server's retry hint) |
| `EMBEDDING_TEXTS_PER_MINUTE` | `90` | Document texts embedded per minute, so an upload cannot starve chat of quota; `0` turns it off (billing-enabled keys) |
| `LLM_MODEL` | `gemini-3.8-flash` | Answer generation (`gemini-2.5-flash` is closed to new API keys) |
| `LLM_TIMEOUT_SECONDS` | `60` | A model call is abandoned after this long |
| `INTERACTIVE_MAX_RETRIES` / `INTERACTIVE_MAX_WAIT_SECONDS` | `2` / `30` | Rate-limit retries while a user waits (LLM and query embedding) |
| `REWRITE_MODEL` | empty (= `LLM_MODEL`) | Follow-up question rewriting |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_THINKING_BUDGET` | `0.0` / `1024` / `0` | Generation settings |
| `RETRIEVAL_CANDIDATES` / `RETRIEVAL_TOP_K` | `30` / `5` | Candidates per channel; chunks sent to the LLM |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RELEVANCE_THRESHOLD` | `0.60` (calibrated) | Refusal cutoff on the best dense cosine similarity; re-calibrate after changing the embedding model |
| `LEXICAL_OVERRIDE_MARGIN` | `0.15` | How far below the threshold an exact identifier match (`ERR_404_NET`) may still let a question through; `0` turns the override off |
| `RERANK_ENABLED` | `false` | Must stay `false`: reranking is not implemented, and `true` is rejected at startup |
| `CHUNK_TARGET_CHARS` / `CHUNK_OVERLAP_CHARS` | `800` / `100` | Chunking; target plus 1.5 x overlap plus 1 must not exceed 1500 |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |
| `MAX_FILES_PER_REQUEST` | `20` | Files accepted in one upload request |
| `MAX_DOCUMENT_CHARS` | `5000000` | A document with more extracted text fails instead of tying up the worker |
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
* **`/ready` checks Postgres, Chroma, the embedding service and index consistency.** The embedding
  reachability probe (added with `core/embeddings.py` in Phase 3) is cached, so readiness polling does
  not burn API quota. Phase 1 shipped with the first two checks only.
* **`GOOGLE_API_KEY` is optional at the settings level.** The backend (and `pytest`) boots without
  it; the embedding/LLM wrappers raise a clear error at the point of use.
* **Postgres uses the Debian `postgres:16` image**, as the TDD says, rather than Alpine, to avoid
  musl collation differences.
* **The web UI is a small React app** (Vite, TypeScript, Tailwind, TanStack Query) built in the frontend
  image and served by nginx (SPA fallback, `/api` proxy, SSE buffering off for `/api/chat`). To keep it
  small, CLAUDE.md's fine-grained component tree was consolidated into a few files (`pages/`,
  `components/chat`, `components/source`, `api/client.ts`); there is no dark mode. Answers are read with
  `fetch`, not `EventSource`, because the stream needs the bearer token. The cited PDF is fetched with the
  token and shown from a blob URL (`#page=N`), since an `<iframe>` cannot send headers.
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
* **Not implemented, by scope:** account lockout, password reset, email verification, and user
  management. Request rate limits exist at nginx only (see *Audit pass*); a shared deployment still
  needs TLS in front.
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
* **`RELEVANCE_THRESHOLD` is 0.60**, calibrated on `eval/questions.jsonl` (15 questions, 62 chunks): off-topic
  questions score at most 0.56, answerable ones at least 0.67. **There is no clean gap between answerable
  and near-miss questions** (topic present, fact absent: 0.63 to 0.70 against 0.67 to 0.80), so a dense
  score cannot stop them. They reach the model, whose prompt tells it to reply with the refusal sentence;
  this worked live, at the cost of one LLM call. Raising the cutoff would refuse real answers. Details and
  the table are in TDD 6.3. Very short queries can also score high, so the model's refusal is the second
  line of defence.
* **Default model `gemini-3.8-flash`.** `gemini-2.5-flash` answers `404 ... no longer available to new
  users` for a new API key, so the default moved (TDD 6.2, 15). `LLM_MODEL` is configuration; confirm any
  other model with one real call. A lighter model (`gemini-3.1-flash-lite`) reached the first token in 1.9 s
  against 3.0 s on the same prompt with the same answer.
* **Measured speed** (`gemini-3.8-flash`, free tier): first token 1.6 to 6.8 s, whole answer 2.9 to 7.4 s;
  a refusal below the threshold takes 0.6 s because no LLM is called. The TDD targets (1.2 s, 3.5 s) are not met
  with this model. Query embedding takes 0.5 to 0.9 s, accepted.
* **Citations are verified while streaming.** `[n]` markers that map to no retrieved source are removed
  before they are shown; an answer with a bad or missing citation is regenerated once, then stripped and
  flagged. The final answer in the `done` event is authoritative (TDD 6.3, 7).
* **Rate limits.** Free-tier Gemini keys have per-minute and per-day limits for chat as well as for
  embeddings. A 429 is retried briefly with the server's hint (at most 2 times, 30 s each); then the chat
  shows "Rate limit reached, please try again in a minute." It never hangs.
* **Not implemented, by scope:** reranking (`RERANK_ENABLED` must stay `false`; hybrid retrieval ranked the
  expected chunk first for all 10 answerable draft questions), the optional exact-token `where_document`
  search (BM25 plus the lexical override cover it), dark mode, and automated UI tests (the UI is checked
  by hand and by build-time type checking; its logic that mattered, the SSE reader and answer rendering,
  was exercised under Node during the audit pass).
* **Known limitation:** PDFs whose text layer overprints other text (a watermark laid over the body)
  can interleave characters. The document still indexes and its page numbers are right.
* **Not implemented, by scope:** OCR, re-ingesting a failed document in place (delete and upload
  again), and per-user storage quotas.

**Audit pass** (every item below came from reading the code and was reproduced, or covered by a test
that fails without the fix)

*Answers and the guard*

* **The lexical override has a floor.** An exact identifier in the best keyword hit used to pass the
  guard at any similarity, so `What is 100 USD in EUR?` reached the model because a chunk contained
  `100`. It now needs `similarity >= RELEVANCE_THRESHOLD - LEXICAL_OVERRIDE_MARGIN` (0.45), and the
  matching text must come from a passage the model is actually given. On the eval corpus unrelated
  questions score about 0.55 and identifier questions 0.64 or more, so no real question is affected.
* **Section headings are keyword-searchable.** A heading such as `ERR_404_NET` lives in
  `chunks.section_title`, not in the chunk text, so a question about it found nothing. The keyword index
  (and the override check) now include the heading; embeddings do not, so nothing is re-embedded and the
  calibrated threshold stays valid. The keyword index is rebuilt from Postgres at startup, so restarting
  the backend is enough for existing documents. (The text parser also stops promoting a lone `ERR_404_NET`
  line to a heading.)
* **The citation filter leaves ordinary brackets alone.** `[0-9]`, `[2023-2025]` and `args[0]` were
  rewritten or deleted as if they were citations; a dash range now counts only between two real sources,
  and brackets inside code are code. A removed marker takes its own preceding space with it, and nothing
  else in the answer is rewritten (the old global tidy-up turned `a .env file` into `a.env file` in the
  final answer).
* **A blank answer is an error, not a saved message** (rule 19): a reply made only of removed markers is
  retried once and then reported.
* **Follow-up rewriting treats the conversation as untrusted** (earlier answers quote documents), caps
  the rewrite's length, and falls back to the original question when the model writes an essay.

*Documents and indexing*

* **Parsing time is linear.** Hostile or just odd files took far longer than their size suggests, with
  the GIL held so the whole server stalled for everyone (measured: 8 s for 40 KB of unclosed Markdown
  brackets, 52 s for 200 KB of unclosed links, 148 s for 400,000 short text lines, 50 s for 60 KB of
  `A. A. A.`). The patterns are replaced by single passes (the same inputs now take well under a
  second), and `MAX_DOCUMENT_CHARS` bounds the rest. A sandboxed
  parser subprocess was considered and **not** built: it would add a process boundary for a problem
  that linear-time code and the container memory limit already contain.
* **Ingestion logs say where an unexpected failure happened, not what the exception said**: library
  error messages can quote the text being processed (rule 21).
* **Vector store.** Writers take one lock, so a rebuild cannot lose or resurrect a concurrent write; the
  Chroma client has real timeouts (its own default is none); an id that was never stored counts as
  unreachable instead of passing the check; and a crash between dropping the old collection and
  renaming the rebuilt one is recovered at startup (it used to strand every vector). `reembed` proves
  its vectors are searchable and does not write back a document deleted while its batch was embedding.
* **Saved conversations do not outlive the right to read them.** Deleting a document blanks the
  passages copied into saved answers; reading a conversation marks citations the reader can no longer
  see (deleted document, lowered role) as unavailable. The answer text itself stays, as a record.

*Security and resilience*

* **A chat stream no longer holds a database connection for its whole length** (measured: one
  `idle in transaction` per stream), and a login releases its connection before the 250 ms password
  check, so a burst of either cannot exhaust the pool.
* **nginx is the first line of defence.** Per-route body limits (32 KB for sign-in, 64 KB for a question,
  1 MB elsewhere, 256 MB for uploads), request-rate limits for sign-in, chat and uploads, and the
  sign-in is checked **before** an upload body is accepted (an unauthenticated 256 MB POST used to be
  buffered in full). 413 and 429 answers are JSON like the API's own.
* **Defaults that matter for a first deployment.** The port is published on `127.0.0.1` only,
  `SIGNUP_ENABLED` can close signup, the single-worker rule is enforced by the entrypoint (a
  `WEB_CONCURRENCY` above 1 stops the container), containers drop capabilities, cannot gain privileges
  and have memory and log-size limits, and `RERANK_ENABLED=true` is rejected instead of ignored.
* **`/api/docs` stays reachable.** It was briefly planned to be hidden; the README documents it as a way
  to try the API, and it lists routes only.
* **Content-Security-Policy.** Because the sign-in token lives in `localStorage`, nginx now sends a
  policy that allows only scripts from this origin. The PDF drawer needs `object-src`/`frame-src`
  to allow `blob:`; if a browser refuses to show a PDF there, relax `object-src` in
  `frontend/nginx-snippets/security-headers.conf` first.
* **Not changed, on purpose:** the first signup still becomes administrator (a fresh install must be
  able to start), the token stays in `localStorage` (the CSP is the mitigation), and there are no
  per-user quotas.

*Interface*

* The Stop button used to leave the chat permanently busy, and switching conversations while an answer
  streamed could write the old answer into the new one; both are fixed with a per-question id. A stream
  that ends without `done` is reported instead of hanging, errors are readable sentences (not raw
  JSON), logging out clears the cached data, a server restart no longer signs people out, the answer
  renderer handles citations inside bold text and numbered lists, and the library and drawer are
  usable by keyboard.

*Evaluation*

* `eval/sample_docs/` and `eval/README.md` were referred to but missing; the five small generated
  documents are now committed, the set grew from 15 to 24 questions (12 unanswerable), and `run_eval`
  calls the production retrieval and guard, fails on off-topic leakage, and skips questions whose
  documents are not indexed instead of crashing. Result with the full corpus: hybrid hit-rate@5 1.00,
  off-topic stopped 5/5, answerable wrongly stopped 0/12. Off-topic questions score at most 0.544 and the
  weakest answerable one 0.638, so `RELEVANCE_THRESHOLD=0.60` stands.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Bind for 127.0.0.1:8081 failed: port is already allocated` | Another process or container uses host port 8081. Set `HTTP_PORT=8082` (or any free port) in `.env`, then `docker compose up -d`. Find the owner with `docker ps` or `netstat -ano \| findstr :8081`. |
| The app opens on this machine but not from another computer | The port is published on `127.0.0.1` only (`BIND_ADDR`). Set `BIND_ADDR=0.0.0.0` in `.env` and recreate the stack; read "Accounts and roles" first, because there is no TLS and signup is open until you close it. |
| `required variable POSTGRES_PASSWORD is missing a value` | `.env` is missing or incomplete. Run `cp .env.example .env` and fill in the required values. |
| Backend restarts with `ValidationError ... jwt_secret` | `JWT_SECRET` is empty or shorter than 32 characters. |
| Backend cannot log in to Postgres after editing `POSTGRES_PASSWORD` | The password is fixed when the `pg_data` volume is first created. Restore the old value, or reset with `docker compose down -v` (**deletes all data**). |
| My account is not an admin | Someone else signed up first. Promote yourself with the `UPDATE users SET role='admin' ...` command in "Accounts and roles". |
| Signup answers `403 Signup is disabled on this server.` | `SIGNUP_ENABLED=false` and an account already exists. Set it to `true` and run `docker compose up -d backend` to let people register again. |
| `429 Too many requests` | nginx's request-rate limit for sign-in/sign-up, chat or uploads. Wait a few seconds (`Retry-After`) and retry. Behind Docker Desktop every client can look like one address, so a busy office can hit it; raise the `rate`/`burst` values in `frontend/nginx.conf`. |
| `413 The request is too large` | A body over nginx's limit for that route: 32 KB for sign-in, 64 KB for a question, 256 MB for one upload request (the per-file limit is `MAX_UPLOAD_MB`, and at most `MAX_FILES_PER_REQUEST` files). |
| The backend container exits at once with `WEB_CONCURRENCY=...: GroundedRAG must run ONE worker` | A platform or `.env` variable asks uvicorn for several workers. The BM25 index is in-process memory, so remove the variable. |
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
| Chat says "The AI service is temporarily unavailable" | Google answered `503 ... high demand` for the chat model (the backend log shows `"code": 503`). It is Google's overload, not the app. Wait and retry, or set another model your key can use, e.g. `LLM_MODEL=gemini-3.1-flash-lite`, then `docker compose up -d backend`. |
| Chat says "Rate limit reached, please try again in a minute" | The free tier's per-minute (or per-day) Gemini limit. Wait a minute and ask again, or use a billing-enabled key. |
| Chat error "The AI service could not answer (GoogleModelNotFoundError)" | The configured `LLM_MODEL` is not available to your key (`404 ... no longer available to new users`). Set `LLM_MODEL=gemini-3.8-flash` (or another model your key lists) and recreate the backend: `docker compose up -d backend` (a plain restart does not re-read `.env`). |
| Chat error "The search index is unavailable or out of date" | PostgreSQL lists indexed chunks but a search channel returns nothing. Check `GET /api/ready` (`index`), then run `verify-index` or `reembed` (see "Checking the search index"). |
| Everything is refused with "Insufficient information..." | Either nothing is indexed yet (Library shows `indexed`?), your filters exclude the documents, or the question scores below `RELEVANCE_THRESHOLD`. `--verbose` in `python -m eval.run_eval` shows the scores. |
| The source drawer shows no PDF | Only PDFs are embedded; other types offer a download. The file is fetched with your token, so a logged-out tab shows nothing. If a PDF is blank or the browser reports it was blocked, the Content-Security-Policy may be too strict for that browser's PDF viewer: relax `object-src` in `frontend/nginx-snippets/security-headers.conf` and rebuild the frontend. |
| A citation chip reads "source no longer available" | The cited document was deleted, or you may no longer read it (for example your role was lowered). The answer text is kept; the passage is not. |
| A chat answer stops with "The connection closed before the answer finished" | The stream ended without a final event (the backend restarted, or a proxy cut the connection). Ask again. |
