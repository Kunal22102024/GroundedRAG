# CLAUDE.md — GroundedRAG Project Guide

This file is the working guide for anyone (human or AI agent) changing this repository. Read it fully before making changes. It defines **what each file is for, how the system works, how requests are routed, and the rules every change must follow.**

---

## 1. Project Summary

**GroundedRAG** is an enterprise document Q&A web app. Users upload documents (PDF, DOCX, MD, HTML, TXT), then ask questions in a chat interface. Answers are grounded **only** in the uploaded documents, carry clickable inline citations (document, page, section), and the app refuses to answer when the documents don't contain the answer.

### Source-of-truth documents

| File | Defines | Use it for |
| --- | --- | --- |
| `PRD.md` | What the product must do and why | Product behavior, user-facing requirements |
| `TDD.md` | How the system is built | Architecture, schemas, thresholds, configuration |
| `CLAUDE.md` (this file) | How to work in the repo | File purposes, flows, routes, rules |

If code and a document disagree, **do not silently pick one**: follow `TDD.md` for technical behavior and `PRD.md` for product behavior, and record any deliberate deviation in `README.md` under "Decisions" and in the TDD changelog.

### Stack at a glance

* **Backend:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic
* **Databases:** PostgreSQL 16 (source of truth), ChromaDB server mode (dense vectors, cosine distance)
* **Keyword search:** BM25 in the backend (`bm25s` or `rank_bm25`), fused with dense results by RRF in the backend
* **Embeddings:** Google `gemini-embedding-001` via `langchain-google-genai` (768 dims, L2-normalized)
* **LLM:** Google `gemini-2.5-flash` via `langchain-google-genai` (model ID comes from config)
* **Frontend:** React + TypeScript + Vite + Tailwind CSS + TanStack Query
* **Infra:** Docker Compose (frontend/nginx, backend, chroma, postgres)

---

## 2. Repository Structure

```
groundedrag/
├── CLAUDE.md                     # this file
├── PRD.md                        # product requirements
├── TDD.md                        # technical design
├── README.md                     # setup, configuration reference, decisions, troubleshooting
├── docker-compose.yml            # production-like stack (4 services, 2 networks, 3 volumes)
├── docker-compose.dev.yml        # dev override: mounted source, hot reload
├── .env.example                  # every environment variable, documented
├── .gitignore
├── .gitattributes                # forces LF endings so scripts/Dockerfiles work from Windows
│
├── chroma/
│   └── Dockerfile                # pinned chromadb/chroma image + non-root user and /data ownership
│
├── backend/
│   ├── Dockerfile                # python:3.11-slim, non-root, runs docker-entrypoint.sh
│   ├── docker-entrypoint.sh      # alembic upgrade head, then ONE uvicorn worker
│   ├── requirements.txt          # runtime deps (pinned)
│   ├── requirements-dev.txt      # pytest, httpx2, ruff (installed in the image by default)
│   ├── pyproject.toml            # pytest + ruff configuration
│   ├── alembic.ini
│   ├── alembic/                  # env.py, script.py.mako, versions/ (migrations)
│   ├── app/
│   │   ├── main.py               # app factory, startup/shutdown, router mounting
│   │   ├── config.py             # Settings loaded from env (single source of config)
│   │   ├── constants.py          # REFUSAL_MESSAGE, access-level map, enums
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── deps.py               # FastAPI dependencies: db session, current user, require_admin
│   │   ├── middleware.py         # request-ID, structured logging, timing
│   │   │
│   │   ├── api/                  # HTTP layer only: parse request, call service, shape response
│   │   │   ├── auth.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── conversations.py
│   │   │   ├── feedback.py
│   │   │   └── health.py
│   │   │
│   │   ├── services/             # business workflows; orchestrate core + db
│   │   │   ├── health_service.py # readiness probes (Postgres, Chroma) so api/ never touches them
│   │   │   ├── auth_service.py
│   │   │   ├── document_service.py
│   │   │   ├── ingestion.py
│   │   │   └── chat_service.py
│   │   │
│   │   ├── core/                 # pure RAG building blocks; no FastAPI imports
│   │   │   ├── parsers.py
│   │   │   ├── chunking.py
│   │   │   ├── metadata.py
│   │   │   ├── llm.py
│   │   │   ├── embeddings.py
│   │   │   ├── vectorstore.py
│   │   │   ├── bm25.py
│   │   │   ├── filters.py
│   │   │   ├── fusion.py
│   │   │   ├── retrieval.py
│   │   │   ├── rerank.py
│   │   │   ├── query_rewrite.py
│   │   │   ├── prompts.py
│   │   │   ├── generation.py
│   │   │   └── guard.py
│   │   │
│   │   └── db/
│   │       ├── database.py       # engine, session factory
│   │       └── models.py         # SQLAlchemy models
│   │
│   └── tests/
│       ├── conftest.py           # forces dummy env (never the real DB); `db`/`database_url` fixtures create a
│       │                         # throwaway migrated Postgres database; mocked LLM/embeddings
│       ├── test_config.py       # settings defaults and validation
│       ├── test_health.py       # /health and /ready
│       ├── test_middleware.py   # request ID, JSON logging, access-log privacy
│       ├── test_pinned_versions.py  # chroma image tag == chromadb-client pin (rule 28)
│       ├── test_models.py       # defaults, CHECK constraints, cascades (real Postgres)
│       ├── test_migrations.py   # upgrade/downgrade, models == migration (incl. CHECKs and FK actions)
│       ├── test_auth_service.py # hashing, JWT attacks, signup race, timing equalization, log privacy
│       ├── test_auth_api.py     # signup/login/me, role changes applied to live tokens
│       ├── test_access_control.py  # role -> access-level map (retrieval-time tests join it with filters.py)
│       ├── test_parsers.py
│       ├── test_chunking.py
│       ├── test_filters.py
│       ├── test_fusion.py
│       ├── test_guard.py
│       ├── test_citations.py
│       ├── test_access_control.py
│       └── test_integration.py
│
├── frontend/
│   ├── Dockerfile                # multi-stage: node build → nginx-unprivileged (Phase 1: nginx only)
│   ├── nginx.conf                # SPA fallback, /api proxy (re-resolving), SSE buffering off
│   ├── static/index.html         # Phase 1 stack-status page; removed when the React app lands (Phase 5)
│   ├── package.json, vite.config.ts, tailwind.config.ts, tsconfig.json
│   └── src/
│       ├── main.tsx              # entry
│       ├── App.tsx               # router + providers
│       ├── api/                  # fetch wrappers: client.ts, auth.ts, documents.ts, chat.ts, sse.ts
│       ├── hooks/                # useAuth, useDocuments, useConversations, useChatStream
│       ├── pages/                # LoginPage, SignupPage, LibraryPage, ChatPage, NotFoundPage
│       ├── components/
│       │   ├── layout/           # AppShell, Sidebar, ProtectedRoute, ThemeToggle
│       │   ├── library/          # UploadDropzone, DocumentTable, StatusBadge, DocumentFilters
│       │   ├── chat/             # ConversationList, MessageList, MessageBubble, ChatInput,
│       │   │                     # FilterBar, CitationChip, FeedbackButtons, RefusalNotice
│       │   ├── source/           # SourceDrawer, ChunkHighlight, DocumentViewer
│       │   └── ui/               # small shared primitives (Button, Modal, Badge, Skeleton, Toast)
│       ├── lib/                  # types.ts, markdown.tsx, format.ts
│       └── styles/index.css
│
├── eval/
│   ├── questions.jsonl           # question, expected_document, expected_page, answerable
│   ├── sample_docs/              # small generated corpus for the sample questions
│   ├── run_eval.py               # hit-rate@5, MRR, refusal accuracy, latency (p50/p95)
│   └── README.md
│
└── data/                         # gitignored; runtime data when not using named volumes
```

---

## 3. What Each Backend File Is For

### 3.1 App skeleton

| File | Purpose | Notes |
| --- | --- | --- |
| `main.py` | Creates the FastAPI app, registers middleware, mounts routers under `/api`, runs startup tasks | Startup: verify embedding model/dimension vs Chroma collection, ensure Chroma collection exists, load BM25 index from Postgres, mark stuck `processing` documents `failed`/re-queue |
| `config.py` | `Settings` (Pydantic) read from env | The **only** place that reads environment variables |
| `constants.py` | `REFUSAL_MESSAGE = "Insufficient information available in the selected documents."`, `ACCESS_LEVELS`, role → allowed-levels map, status enums | Import these; never retype the strings |
| `schemas.py` | Pydantic models for every request and response | API contracts live here |
| `deps.py` | `get_db`, `get_current_user`, `require_admin` (plus `DbSession`, `CurrentUser`, `AdminUser` aliases) | Auth is applied through dependencies, never ad hoc. `get_current_user` re-reads the user from Postgres on every request, so role changes and deletions apply to tokens already issued. Missing/invalid token is 401 (not HTTPBearer's default 403) |
| `middleware.py` | Request ID, JSON logging, request timing (pure ASGI, so SSE streams are not buffered) | Never logs document text, questions, secrets, or query strings. `configure_logging` must leave `uvicorn.access` without handlers (uvicorn logs access lines, with query strings, whenever that logger has handlers) |

### 3.2 `api/` — HTTP layer

Handlers validate input, call one service function, and return a schema. **No business logic, no direct Chroma/LLM calls.**

| File | Purpose |
| --- | --- |
| `auth.py` | Signup, login, current-user. First registered user becomes `admin`. Signup and login both return `{access_token, token_type, expires_in, user}`. Handlers are plain `def` (bcrypt is CPU-bound; FastAPI runs them in a worker thread). 409 on duplicate email; one identical 401 for wrong password and unknown email |
| `documents.py` | Upload (multi-file), list, detail, delete, serve original file |
| `chat.py` | `POST /chat` — streams Server-Sent Events |
| `conversations.py` | List, read, rename, delete conversations |
| `feedback.py` | Thumbs up/down on an assistant message |
| `health.py` | `/health` (liveness, touches nothing) and `/ready` (Postgres, Chroma; the embedding-model probe is added with `core/embeddings.py`). Delegates to `health_service` |

### 3.3 `services/` — workflows

| File | Purpose |
| --- | --- |
| `health_service.py` | Readiness probes run concurrently in worker threads with a timeout; failure details are exception class names only (errors can embed hosts or credentials) |
| `auth_service.py` | bcrypt hashing (work factor `BCRYPT_ROUNDS`), HS256 JWT issue/verify (algorithm pinned, claims `sub`/`iat`/`exp` only), `register_user`, `authenticate`. Signup takes a Postgres advisory lock so concurrent first signups cannot create two admins; unknown-email logins still run a bcrypt check so timing does not reveal which emails exist. Typed errors (`EmailAlreadyRegistered`, `InvalidCredentials`, `InvalidToken`) are translated to HTTP in `api/` and `deps.py` |
| `document_service.py` | Validate uploads, store files under generated names, create DB rows, enqueue ingestion, delete (vectors → DB rows → file → BM25 refresh) |
| `ingestion.py` | Background job: parse → chunk → metadata → write chunks to Postgres → embed → upsert to Chroma → refresh BM25 → set status. Cleans up on failure |
| `chat_service.py` | Full chat turn: rewrite → filter → retrieve → guard → generate → verify citations → persist → stream events |

### 3.4 `core/` — RAG building blocks (no web framework imports)

| File | Purpose | Key rules |
| --- | --- | --- |
| `parsers.py` | PDF (page-aware, pdfplumber), DOCX, MD, HTML, TXT → list of `(page, section, text)` | Detects scanned PDFs (no extractable text) and raises a clear error |
| `chunking.py` | `Chunker` interface + structure-aware implementation (~800 chars, ~100 overlap) | Keeps page range; never exceeds the embedding token ceiling; semantic chunker is a future implementation of the same interface |
| `metadata.py` | Resolve department, type, creation date, access level | Order: explicit upload values → folder/filename → file properties → upload date; produces `creation_date_int` |
| `llm.py` | Builds the `ChatGoogleGenerativeAI` instance from settings | The **only** place that constructs the LLM client |
| `embeddings.py` | Wraps `GoogleGenerativeAIEmbeddings`: document vs query task types, batching, retries/backoff, L2 normalization, query LRU cache | The **only** place that calls the embedding API |
| `vectorstore.py` | Chroma client: create/get collection (cosine), upsert, dense query, delete by `document_id` | The **only** place that imports `chromadb`; passes embeddings explicitly. Create collections with `configuration={"hnsw": {"space": "cosine"}}` and `embedding_function=None` (the client's default is Chroma's built-in embedder; verified against 1.5.9) |
| `bm25.py` | In-memory BM25 index, identifier-aware tokenizer, add/remove/rebuild, filtered top-N | Rebuilt from Postgres at startup; filter parity with Chroma |
| `filters.py` | `build_filter(user, request_filters)` returning one neutral filter object plus Chroma `where` and SQL translations | **The only place filters are built**; always injects the access-level constraint |
| `fusion.py` | `rrf(rankings, k)` | Pure function; RRF scores are for ordering only |
| `retrieval.py` | Orchestrates embed query → dense + BM25 (parallel) → fusion → optional rerank; returns candidates with dense similarity and timings | |
| `rerank.py` | Optional FlashRank cross-encoder | Behind `RERANK_ENABLED` |
| `query_rewrite.py` | Turns a follow-up into a standalone question using prior turns | Skipped on the first turn; thinking disabled, small token limit |
| `prompts.py` | System prompt and context-block formatting | Treats context as untrusted data; numbered sources `[1]`, `[2]` |
| `generation.py` | Calls the LLM (streaming), assembles the answer, maps citations to chunks, verifies them, retries once | Handles `MAX_TOKENS`, safety blocks, empty candidates |
| `guard.py` | Relevance gate (cosine or reranker score, plus lexical override) and citation verification helpers | Never thresholds on the RRF score |

### 3.5 `db/`

| File | Purpose |
| --- | --- |
| `database.py` | SQLAlchemy engine and session factory |
| `models.py` | `users`, `documents`, `chunks`, `conversations`, `messages`, `feedback` (columns defined in `TDD.md` §4.3). Enum-like columns are `VARCHAR` + named `CHECK` (not PG enums); cascades are `ON DELETE` in the database; constraint names come from `NAMING_CONVENTION` in `database.py`. Every change needs a new migration, and `tests/test_migrations.py` fails if models and migrations disagree |

---

## 4. How the System Works

### 4.1 Ingestion flow (upload → searchable)

1. `POST /api/documents` → `api/documents.py` validates type/size, calls `document_service`.
2. `document_service` stores the file under a generated UUID name, creates a `documents` row with status `queued`, schedules `ingestion`.
3. `ingestion` sets `processing`, then: `parsers` → `chunking` → `metadata` → insert `chunks` rows in Postgres.
4. `embeddings.embed_documents` (task `RETRIEVAL_DOCUMENT`, batched, normalized) → `vectorstore.upsert` (flat metadata, explicit vectors).
5. `bm25` index is updated; document becomes `indexed` with `chunk_count`.
6. Any failure: partial chunks removed from Chroma and Postgres, status `failed` with `error_message`. Scanned PDFs fail with "scanned PDF, OCR not supported".

### 4.2 Chat flow (question → cited answer)

1. `POST /api/chat` → `chat_service` (streams SSE).
2. If the conversation has prior turns → `query_rewrite` makes a standalone question.
3. `filters.build_filter` combines user filters with the role's access levels.
4. `embeddings.embed_query` (task `RETRIEVAL_QUERY`) → in parallel: `vectorstore` dense top 30 and `bm25` top 30, both filtered.
5. `fusion.rrf` merges; optional `rerank`; keep top K (default 5).
6. `guard`: if best cosine (or reranker) score is below `RELEVANCE_THRESHOLD` and the lexical override does not apply → emit `REFUSAL_MESSAGE`, **do not call the LLM**.
7. Otherwise `prompts` + `generation` stream tokens; citations are mapped to chunks and verified; one retry on invalid citations, then invalid markers are stripped and the message is flagged.
8. Persist the message (answer, citations, guard score, timings); emit `citations` and `done`.

### 4.3 Delete flow

`DELETE /api/documents/{id}` → permission check (admin or owner) → delete Chroma vectors by `document_id` → delete Postgres rows → delete the file → refresh BM25. If a step fails, the document stays in a retryable state; it must never be left half-searchable.

---

## 5. API Routing Table

All routes live under `/api`. "Auth" means a valid JWT bearer token.

| Method & path | Auth | Handler file | Purpose |
| --- | --- | --- | --- |
| `POST /auth/signup` | none | `api/auth.py` | Create account (first user = admin) |
| `POST /auth/login` | none | `api/auth.py` | Return JWT |
| `GET /auth/me` | user | `api/auth.py` | Current user and role |
| `POST /documents` | user | `api/documents.py` | Upload files (`Restricted` only if admin) |
| `GET /documents` | user | `api/documents.py` | List documents the user may see |
| `GET /documents/{id}` | user | `api/documents.py` | Document detail and status |
| `GET /documents/{id}/file` | user | `api/documents.py` | Original file (access-checked) |
| `DELETE /documents/{id}` | admin or owner | `api/documents.py` | Delete everywhere |
| `POST /chat` | user | `api/chat.py` | SSE stream: `status`, `token`, `citations`, `done`, `error` |
| `GET /conversations` | user | `api/conversations.py` | Own conversations |
| `GET /conversations/{id}` | user (owner) | `api/conversations.py` | Messages |
| `PATCH /conversations/{id}` | user (owner) | `api/conversations.py` | Rename |
| `DELETE /conversations/{id}` | user (owner) | `api/conversations.py` | Delete |
| `POST /messages/{id}/feedback` | user (owner) | `api/feedback.py` | Thumbs up/down |
| `GET /health` | none | `api/health.py` | Liveness |
| `GET /ready` | none | `api/health.py` | Readiness |

**Access levels:** `admin` → Public, Internal, Restricted. `user` → Public, Internal.

### Frontend routes

| Path | Page | Access |
| --- | --- | --- |
| `/login`, `/signup` | `LoginPage`, `SignupPage` | public |
| `/` | redirect to `/chat` | protected |
| `/chat`, `/chat/:conversationId` | `ChatPage` | protected |
| `/library` | `LibraryPage` | protected |
| `*` | `NotFoundPage` | public |

nginx serves the SPA (with fallback to `index.html`), proxies `/api` to `backend:8000`, and disables buffering for `POST /api/chat`.

---

## 6. Frontend File Purposes

| Area | Files | Purpose |
| --- | --- | --- |
| `api/` | `client.ts` | `fetch` wrapper: base URL, JWT header, 401 → logout |
| | `sse.ts` | Parses the `/chat` event stream (`status`, `token`, `citations`, `done`, `error`) |
| | `auth.ts`, `documents.ts`, `chat.ts` | Typed endpoint functions |
| `hooks/` | `useAuth`, `useDocuments` (polls while any document is `queued`/`processing`), `useConversations`, `useChatStream` | State and data fetching via TanStack Query |
| `pages/` | `LibraryPage` | Upload dropzone, document table, status badges, delete confirmation |
| | `ChatPage` | Conversation sidebar, message list, filter bar, input, source drawer |
| `components/source/` | `SourceDrawer` | Opens on citation click: highlighted chunk, document name, page, section, embedded original (`#page=N` for PDFs; other types show chunk + download link) |
| `components/chat/` | `CitationChip`, `RefusalNotice`, `FeedbackButtons` | Clickable citations, "not found in documents" state, thumbs up/down |

---

## 7. Rules (Invariants — do not break)

### 7.1 Architecture

1. **Layering:** `api → services → core / db`. `core/` never imports FastAPI. `api/` never calls Chroma, BM25, or the LLM directly.
2. **Single points of contact:** only `vectorstore.py` imports `chromadb`; only `embeddings.py` calls the embedding API; only `llm.py` builds the LLM client; only `config.py` reads environment variables; only `filters.py` builds filters.
3. **PostgreSQL is the source of truth.** Chroma and BM25 are derived and rebuildable. Write chunks to Postgres before Chroma.
4. **Single backend worker** (the BM25 index is in-process). Do not add `--workers`.
5. **Model IDs and thresholds come from config.** Never hardcode `gemini-*` names, dimensions, or the relevance threshold in code.

### 7.2 Retrieval and grounding

6. **Access control on every query.** The access-level constraint is applied to both the dense and BM25 channels. A change that could bypass it must come with a test in `test_access_control.py`.
7. **Never threshold on the RRF score** (its maximum is about 0.033). Gate on dense cosine similarity (`1 - Chroma distance`) or the reranker score.
8. **Never skip the guard before calling the LLM.** Below-threshold queries return `REFUSAL_MESSAGE` without an LLM call, except when the lexical override applies.
9. **Every claim needs a citation** that maps to a retrieved chunk. Citation verification is mandatory; invalid markers are never shown to the user.
10. **Document text is untrusted data.** Prompts must frame it as data, never instructions.
11. **Use the constant** `REFUSAL_MESSAGE` verbatim.

### 7.3 Embeddings and Chroma

12. Chunks use `task_type=RETRIEVAL_DOCUMENT`; queries use `RETRIEVAL_QUERY`. Always pass explicitly.
13. Vectors are L2-normalized and have the configured dimension. Ingestion and query must use the identical model and dimension; the startup check enforces this.
14. Chroma collections use **cosine** distance and receive **explicit embeddings** (never Chroma's default embedding function).
15. Chroma metadata is **flat** (str/int/float/bool, no `null`, no nesting). Date filters use `creation_date_int`.
16. Chunks must stay under the 2,048-token embedding input limit; never rely on silent truncation.
17. Changing `EMBEDDING_MODEL`, `EMBEDDING_DIM`, or `LLM_MODEL` requires running `eval/run_eval.py` and recalibrating `RELEVANCE_THRESHOLD`.

### 7.4 LLM behavior

18. Keep `temperature` low and the **thinking budget at 0** for answering and rewriting (latency, cost, and truncation safety).
19. Handle finish reasons: `MAX_TOKENS` (flag partial), safety block or empty response (emit an `error` event). Never return a blank answer silently.
20. Stream answers; do not buffer the full response before sending.

### 7.5 Data and security

21. Never log document text, user questions, tokens, or API keys.
22. Secrets only through `.env` (ignored by git). Commit `.env.example` only.
23. Uploaded files are stored under generated names; validate extension, content type, and size; never use user-supplied paths.
24. Deleting a document must remove vectors, rows, and file; its content must never appear in later answers.
25. Passwords: bcrypt, at most 72 bytes (rejected at signup, never silently truncated). Tokens: short-lived HS256 JWT carrying only the user id. Only nginx publishes a port; Postgres and Chroma stay internal. Compose enforces this with an `internal: true` network (Postgres, Chroma, backend) and an `edge` network (nginx, backend; gives the backend outbound access to Gemini). Postgres and Chroma have no route to the internet and nginx cannot resolve them.

### 7.6 Docker

26. Four services: `frontend` (nginx), `backend`, `chroma`, `postgres`. Named volumes `pg_data`, `chroma_data`, `uploads`.
27. Healthchecks plus `depends_on: condition: service_healthy`; the backend runs `alembic upgrade head` on start.
28. Pin the ChromaDB image and client library versions together (`chroma/Dockerfile` tag == `chromadb-client` in `requirements.txt`). This cannot be checked at runtime (the server's `/api/v2/version` returns a fixed `"1.0.0"`), so `tests/test_pinned_versions.py` enforces it; bump both in one change.
29. No embedding model is baked into the image; only the optional FlashRank model may be.
30. Containers run as non-root: backend uid 10001, chroma uid 10001 (via `chroma/Dockerfile`, which pre-creates `/data` so the named volume inherits ownership), nginx via `nginx-unprivileged` (host port 8081, container port 8080), postgres via `user: postgres`.

---

## 8. Coding Standards

* Python: type hints everywhere, docstrings on public functions, small single-purpose functions, `async` for I/O paths that stream. Format with `ruff`/`black`; keep imports sorted.
* TypeScript: strict mode, no `any`, typed API responses in `lib/types.ts`.
* No placeholder or `TODO` stubs in delivered features.
* Errors: raise typed exceptions in `core/`, translate them to HTTP errors in `api/`.
* Add a dependency only when needed, and note why in the PR/commit message.
* Keep prompts, constants, and thresholds in their dedicated files (`prompts.py`, `constants.py`, `config.py`), not inline.

---

## 9. Testing & Evaluation Rules

* **Database tests** run against a throwaway database (`groundedrag_test_<random>`) that the fixtures create on the stack's Postgres, migrate with the real Alembic migrations, and drop afterwards; they skip when no server is reachable and refuse to truncate any other database. New concurrency-sensitive code needs a test that **fails when the protection is removed** (see the signup-lock test, which widens the race window on purpose); mutate a copy of the code to check.
* **Unit tests:** parsers, chunking (page numbers preserved, no mid-sentence cuts, overlap, token ceiling), filter building, RRF, guard (including lexical override), citation verification, access-level filtering.
* **Integration test:** upload a small sample document → wait for `indexed` → query → assert a valid citation. Mock the LLM and embeddings in tests (no network calls, no API key needed).
* **Evaluation (`eval/run_eval.py`):** hit-rate@5 and MRR for dense-only, keyword-only, hybrid, hybrid + rerank; refusal accuracy on unanswerable questions; per-stage latency p50/p95. Include exact-token questions and about 10 unanswerable ones.
* A change is not done until `pytest` passes and, if it touches retrieval, embeddings, the guard, or prompts, the eval script has been run and compared with the previous result.

---

## 10. Working Method

1. Read `PRD.md`, `TDD.md`, and this file first.
2. Build in phases and **verify each phase by running it** before moving on:
   1. Compose skeleton (postgres, chroma, backend health, nginx) — all containers healthy
   2. DB models, migrations, auth
   3. Parsers, chunking, embeddings, vector store, BM25, ingestion, documents API
   4. Retrieval, guard, rewriting, generation, streaming chat, citation verification
   5. Frontend (auth, library, chat, source drawer)
   6. Tests, eval script, dev override, README
3. Commit after each verified phase with a clear message.
4. If a requirement is ambiguous, choose the most reasonable option, document it under "Decisions" in `README.md`, and continue; ask only when truly blocked.
5. If a library API differs from expectations (LangChain Google GenAI, Chroma, qdrant-free code paths), check the **installed version's** documentation rather than guessing parameter names.
6. When behavior changes, update `TDD.md` (and its changelog) and this file in the same change.

### Where to change things

| To change… | Edit… |
| --- | --- |
| Embedding model, dimension, batch size | `.env` / `config.py`; then re-run eval and re-embed |
| LLM model ID, temperature, thinking budget | `.env` / `config.py` (built in `core/llm.py`) |
| Refusal threshold | `.env` (`RELEVANCE_THRESHOLD`) after calibrating with eval |
| Answer style, citation format | `core/prompts.py` (and update citation verification if the format changes) |
| Chunk size or strategy | `core/chunking.py` + `.env` (`CHUNK_*`); re-ingest documents; re-run eval |
| Add a metadata filter | `core/filters.py`, `schemas.py`, `vectorstore.py` metadata, `chunks` model + migration, frontend `FilterBar` |
| Add a file type | `core/parsers.py`, upload validation in `document_service.py`, frontend dropzone accept list |
| Add an API route | `api/<name>.py`, `schemas.py`, `main.py` router, frontend `api/` wrapper, routing table above |
| Add a table or column | `db/models.py` + a new Alembic migration (never edit old migrations) |

---

## 11. Commands

```bash
# first run
cp .env.example .env              # set GOOGLE_API_KEY, JWT_SECRET, DB password
docker compose up --build         # app at http://localhost:8081

# development with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# tests, lint and evaluation
docker compose exec backend pytest
docker compose exec backend ruff check . && docker compose exec backend ruff format --check .
docker compose exec backend python -m eval.run_eval

# host port 8081 already in use? Set HTTP_PORT in .env (e.g. HTTP_PORT=8082)

# migrations
docker compose exec backend alembic revision --autogenerate -m "describe change"
docker compose exec backend alembic upgrade head
```

---

## 12. Definition of Done (acceptance checks)

1. `docker compose up --build` from a clean checkout starts everything; `http://localhost:8081` loads.
2. Register, log in, upload a multi-page PDF and a DOCX; statuses go queued → processing → indexed.
3. Ask a question: streamed answer with working citation chips; clicking one opens the drawer with the exact chunk and the original PDF at the cited page.
4. An exact-token query (error code or ID present in a document) is found even when phrased awkwardly, and is not wrongly refused.
5. An unrelated question returns the refusal sentence **without calling the LLM**.
6. A department filter returns zero chunks from other departments; a `user` cannot retrieve `Restricted` documents.
7. Deleting a document removes it from the library and its content never appears in answers again.
8. After `docker compose down` and `up`, documents, users, and conversations persist.
9. `pytest` passes; the eval script prints the comparison table.

---

## 13. Known Limitations

* Scanned PDFs (no extractable text) are rejected; OCR is not supported.
* Single backend worker (in-memory BM25 index).
* Document text and questions are sent to Google's Gemini API; use a paid-tier key and review current terms before uploading confidential documents.
* `gemini-2.5-flash` is a legacy model with an uncertain shutdown date; keep the model ID in configuration and re-run the eval before migrating.
* ChromaDB has no per-user authorization; it must stay on the internal Docker network.
