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
* **LLM:** Google `gemini-3.8-flash` via `langchain-google-genai` (model ID comes from config)
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
├── docker-compose.yml            # production-like stack (4 services, 2 networks, 3 volumes); port bound to
│                                 # BIND_ADDR (127.0.0.1), memory limits, no-new-privileges, log rotation
├── docker-compose.dev.yml        # dev override: mounted source (backend and eval/), hot reload
├── .env.example                  # every environment variable, documented
├── .gitignore
├── .gitattributes                # forces LF endings so scripts/Dockerfiles work from Windows
│
├── chroma/
│   └── Dockerfile                # pinned chromadb/chroma image + non-root user and /data ownership
│
├── backend/
│   ├── Dockerfile                # python:3.11-slim, non-root, runs docker-entrypoint.sh
│   ├── docker-entrypoint.sh      # refuses WEB_CONCURRENCY/UVICORN_WORKERS > 1; alembic upgrade head; ONE uvicorn worker
│   ├── requirements.txt          # runtime deps (pinned)
│   ├── requirements-dev.txt      # pytest, httpx2, ruff (installed in the image by default)
│   ├── pyproject.toml            # pytest + ruff configuration
│   ├── alembic.ini
│   ├── alembic/                  # env.py, script.py.mako, versions/ (0001 schema, 0002 nullable creation_date,
│   │                             # 0003 document_insights + messages.conflicts, 0004 query_events + messages.suggestions,
│   │                             # 0005 knowledge graph: kg_entities, kg_mentions, kg_relations, the `graph` kinds)
│   ├── app/
│   │   ├── main.py               # app factory, startup/shutdown, router mounting
│   │   ├── config.py             # Settings loaded from env (single source of config)
│   │   ├── constants.py          # REFUSAL_MESSAGE, access-level map, enums
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── deps.py               # FastAPI dependencies: db session, current user, require_admin
│   │   ├── maintenance.py        # `python -m app.maintenance reembed`: rebuild vectors from Postgres (resumable)
│   │   ├── middleware.py         # request-ID, structured logging, timing
│   │   │
│   │   ├── api/                  # HTTP layer only: parse request, call service, shape response
│   │   │   ├── auth.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── conversations.py      # conversations CRUD and POST /messages/{id}/feedback
│   │   │   ├── analysis.py           # summaries, extraction, comparison (SSE) + a document's saved insights
│   │   │   ├── suggestions.py        # GET /suggestions: questions from documents and often asked
│   │   │   ├── graph.py              # GET /graph, GET /graph/entity, POST /documents/{id}/graph: the knowledge graph
│   │   │   ├── metrics.py            # evaluation dashboard (admins only)
│   │   │   ├── users.py              # account administration: list, promote/demote, delete (admins only)
│   │   │   └── health.py
│   │   │
│   │   ├── services/             # business workflows; orchestrate core + db
│   │   │   ├── analysis_service.py  # access, saved results, streaming drivers, background executive summary,
│   │   │   │                        # suggested questions per document
│   │   │   ├── graph_service.py     # build a document's knowledge graph (insights thread), read it back for a reader
│   │   │   ├── cache_service.py     # drop saved answers when a document is indexed, deleted or re-graphed; evict a rated-down one
│   │   │   ├── metrics_service.py   # query_events: record one per request, dashboard figures, often-asked questions
│   │   │   ├── user_service.py      # list/promote/demote/delete accounts; always one admin, under the accounts lock
│   │   │   ├── health_service.py # readiness probes (Postgres, Chroma, embeddings) so api/ never touches them
│   │   │   ├── index_service.py  # startup: verify collection, recover jobs, refuse inconsistent index, load BM25
│   │   │   ├── auth_service.py
│   │   │   ├── document_service.py
│   │   │   ├── ingestion.py
│   │   │   ├── chat_service.py
│   │   │   └── conversation_service.py  # conversation history, rename, delete, message feedback
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
│   │   │   ├── graph_retrieval.py    # GraphRAG: passages linked to the retrieved ones through the knowledge graph
│   │   │   ├── answer_cache.py       # semantic answer cache: scoped, in memory, invalidated by document changes
│   │   │   ├── (rerank.py: planned, does not exist yet; RERANK_ENABLED is rejected while true)
│   │   │   ├── query_rewrite.py
│   │   │   ├── prompts.py
│   │   │   ├── generation.py
│   │   │   ├── guard.py
│   │   │   ├── analysis.py       # summaries, extraction, conflict detection: pure, cited, validated
│   │   │   ├── graph.py          # entity and relation extraction from chunks: pure, every item checked against its chunk
│   │   │   └── gemini_errors.py  # shared Gemini error classification (rate limit, key, transient)
│   │   │
│   │   └── db/
│   │       ├── database.py       # engine, session factory
│   │       └── models.py         # SQLAlchemy models
│   │
│   └── tests/
│       ├── conftest.py           # pins every setting to its default and forces dummy env (never the real DB);
│       │                         # `db`/`database_url` fixtures create a throwaway migrated Postgres database;
│       │                         # mocked LLM/embeddings; REQUIRE_STACK=1 turns "no Postgres/Chroma" into a failure
│       ├── test_config.py       # settings defaults and validation
│       ├── test_health.py       # /health and /ready
│       ├── test_middleware.py   # request ID, JSON logging, access-log privacy
│       ├── test_pinned_versions.py  # chroma image tag == chromadb-client pin (rule 28; skipped inside the image)
│       ├── test_entrypoint.py   # more than one worker is refused (rule 4)
│       ├── test_models.py       # defaults, CHECK constraints, cascades (real Postgres)
│       ├── test_migrations.py   # upgrade/downgrade, models == migration (incl. CHECKs and FK actions)
│       ├── test_auth_service.py # hashing, JWT attacks, signup race, timing equalization, log privacy
│       ├── test_auth_api.py     # signup/login/me, role changes applied to live tokens, SIGNUP_ENABLED
│       ├── test_access_control.py  # role -> access-level map (retrieval-time tests join it with filters.py)
│       ├── pdf_factory.py       # builds PDFs (pages, outlines, columns, scans) for tests; no binary fixtures
│       ├── test_parsers.py      # pages, headings, columns, chrome removal, every format, hostile-input timing
│       ├── test_chunking.py     # sentence integrity, overlap rules, page ranges, size ceilings, timing
│       ├── test_metadata.py     # precedence: upload > file name > file properties > upload date
│       ├── test_embeddings.py   # task types, normalization, batching, retries, per-minute budget, errors
│       ├── test_vectorstore.py  # real Chroma: cosine, flat metadata, mismatch refusal, filters
│       ├── test_index_health.py # unreachable/missing vectors, repair, rebuild recovery, write lock
│       ├── test_bm25.py         # identifier-aware tokenizer, filtering, concurrent updates
│       ├── test_ingestion.py    # end to end with a fake embedder; failure cleanup; headings; size cap; worker pool
│       ├── test_index_service.py  # startup consistency and refusal to start
│       ├── test_maintenance.py  # reembed and verify-index
│       ├── test_documents_api.py  # upload validation, visibility, file serving, delete
│       ├── test_conversations_api.py  # ownership of conversations/feedback, saved citations, every route needs auth, no DB connection held
│       ├── test_filters.py
│       ├── test_fusion.py
│       ├── test_guard.py        # relevance gate and the lexical override with its floor
│       ├── test_citations.py    # citation filter, regeneration, finish reasons, blank answers
│       ├── test_query_rewrite.py  # follow-up rewriting, bounded and treated as untrusted
│       ├── test_llm.py          # model client construction, one client per event loop, background jobs close theirs
│       ├── test_chat.py         # chat end to end (the integration test: upload -> index -> ask -> cited answer)
│       ├── test_analysis_core.py  # JSON parsing, batching, global numbering, grounding rules (summaries, extraction, conflicts)
│       ├── test_analysis_api.py   # the endpoints, saved results, access, deletion, auto summary, conflicts in chat
│       ├── test_users_api.py      # admin-only, role changes on live tokens, last-admin rule (+ race), delete cascades
│       ├── test_metrics.py        # one event per turn/run incl. failures, figure definitions, admin-only, retention
│       ├── test_suggestions.py    # document questions, often-asked privacy rules, follow-ups after an answer
│       ├── test_semantic_cache.py # the cache alone (threshold, scope, TTL, LRU, invalidation, generation), then chat: hits, scopes, invalidation, thumbs-down
│       ├── test_graph_core.py     # extraction: grounding of entities and relations, merging, batches, bad JSON
│       ├── test_graph_api.py      # build, read, filters, shared entities, access (Restricted), delete cascade, failure and retry
│       ├── test_graph_retrieval.py  # GraphRAG: pulls in linked passages, never past the filter, never after a refusal, hops
│       └── test_eval_corpus.py  # eval/questions.jsonl agrees with eval/sample_docs/ (needs eval/ mounted)
│
├── frontend/
│   ├── Dockerfile                # multi-stage: node build → nginx-unprivileged
│   ├── nginx.conf                # SPA fallback and cache headers, /api proxy (re-resolving), SSE buffering off,
│   │                             # per-route body limits, request-rate limits, sign-in checked before upload bodies
│   ├── nginx-snippets/           # security-headers.conf (incl. Content-Security-Policy), proxy-common.conf
│   ├── package.json, vite.config.ts, tsconfig.json   # Tailwind v4 is configured in styles/index.css
│   └── src/
│       ├── main.tsx              # entry: error boundary, query client, router, auth provider
│       ├── App.tsx               # routes, protected layout (header, navigation), 404 page
│       ├── api/client.ts         # fetch wrapper (token, 401 -> logout, readable errors); `streamSse` (the one SSE reader,
│       │                         # finishes only on done/error) and `streamChat` on top of it
│       ├── hooks/                # useAuth.tsx (auth context, clears cached data), useAnalysisRun.ts (one streamed
│       │                         # analysis at a time), useIndexedDocuments.ts, useSuggestions.ts, useKnowledgeGraph.ts
│       │                         # (the graph, one entity, building a graph)
│       ├── pages/                # LoginPage (login and signup, by `mode`), LibraryPage, ChatPage, AnalyzePage,
│       │                         # DashboardPage and UsersPage (admins)
│       ├── components/
│       │   ├── ErrorBoundary.tsx
│       │   ├── ui/               # icons.tsx (inline SVG, LogoMark), primitives.tsx (Spinner, ProgressBar,
│       │   │                     # Skeleton, StatusPill, AccessBadge, FileTypeBadge, Modal), Toast.tsx
│       │   ├── chat/             # Answer (text with citation chips, streaming caret), CitationChip, ConflictPanel,
│       │   │                     # SuggestionChips (follow-ups), FilterBar
│       │   ├── analyze/          # CompareTab, ExtractTab, SummariesTab, GraphTab, DocumentPicker (tabs of AnalyzePage)
│       │   ├── graph/            # ForceGraph.tsx (SVG graph: zoom, pan, drag, keyboard), layout.ts (the force layout, pure),
│       │   │                     # entityTypes.ts (colours and names of entity types)
│       │   ├── dashboard/        # charts.tsx: Sparkline, BarList, Histogram, StackedBars (SVG/CSS)
│       │   └── source/           # SourceDrawer (cited passage + original PDF at the page)
│       ├── lib/                  # types.ts, files.ts (download, save, sizes, page labels), ui.ts (shared classes),
│       │                         # format.ts (dates), analysis.ts (Markdown/JSON/CSV export of results)
│       └── styles/index.css      # Tailwind v4 theme: the warm peach colours (indigo = peach, violet = coral,
│                                 # slate = warm neutral are redefined here; change the theme here, not in
│                                 # components), animations (keyframes) and reduced-motion handling
│   public/favicon.svg
│
├── eval/
│   ├── questions.jsonl           # question, expected_document, expected_page, answerable, kind, requires?
│   ├── sample_docs/              # seven small generated documents (the three SANS PDFs are a documented download)
│   ├── run_eval.py               # hit-rate@5, MRR, guard on the production retrieval, hybrid+graph and multi-hop, latency (p50/p95)
│   └── README.md                 # corpus, how to run it, question kinds, reference result
│
└── data/                         # gitignored; runtime data when not using named volumes
```

---

## 3. What Each Backend File Is For

### 3.1 App skeleton

| File | Purpose | Notes |
| --- | --- | --- |
| `main.py` | Creates the FastAPI app, registers middleware, mounts routers under `/api`, runs startup tasks | Startup: verify embedding model/dimension vs Chroma collection, ensure Chroma collection exists, load BM25 index from Postgres, mark stuck `processing` documents `failed`/re-queue |
| `config.py` | `Settings` (Pydantic) read from env | The **only** place that reads environment variables. Validators reject inconsistent values (chunk size vs the embedding ceiling, `RETRIEVAL_TOP_K` > candidates, `RERANK_ENABLED=true`); `database_url` is excluded from `repr` |
| `constants.py` | `REFUSAL_MESSAGE = "Insufficient information available in the selected documents."`, `ACCESS_LEVELS`, role → allowed-levels map, status enums | Import these; never retype the strings |
| `schemas.py` | Pydantic models for every request and response | API contracts live here |
| `deps.py` | `get_db`, `get_current_user`, `require_admin` (plus `DbSession`, `CurrentUser`, `AdminUser` aliases) | Auth is applied through dependencies, never ad hoc. `get_current_user` re-reads the user from Postgres on every request, so role changes and deletions apply to tokens already issued, then **detaches the user and ends the read transaction** (otherwise a streamed answer holds one pooled connection for its whole length). Missing/invalid token is 401 (not HTTPBearer's default 403) |
| `middleware.py` | Request ID, JSON logging, request timing (pure ASGI, so SSE streams are not buffered) | Never logs document text, questions, secrets, or query strings. `configure_logging` must leave `uvicorn.access` without handlers (uvicorn logs access lines, with query strings, whenever that logger has handlers) |

### 3.2 `api/` — HTTP layer

Handlers validate input, call one service function, and return a schema. **No business logic, no direct Chroma/LLM calls.**

| File | Purpose |
| --- | --- |
| `auth.py` | Signup, login, current-user. First registered user becomes `admin`. Signup and login both return `{access_token, token_type, expires_in, user}`. Handlers are plain `def` (bcrypt is CPU-bound; FastAPI runs them in a worker thread). 409 on duplicate email; one identical 401 for wrong password and unknown email; 403 when `SIGNUP_ENABLED=false` and an account exists (checked before the email lookup, so a closed door reveals no accounts) |
| `documents.py` | Upload (multi-file; 413 above `MAX_FILES_PER_REQUEST`, 422 for a bad or implausible field), list, detail, delete, serve original file |
| `chat.py` | `POST /chat` — streams Server-Sent Events |
| `conversations.py` | List, read (citations the reader can no longer see come back `unavailable`), rename, delete conversations; thumbs up/down on an assistant message (`POST /messages/{id}/feedback`) |
| `suggestions.py` | `GET /suggestions?document_ids=`: questions the chosen documents answer (written once in the background, `pending` meanwhile) and questions often asked by several users |
| `graph.py` | `GET /graph?document_ids=&types=&q=` (nodes and edges, entities merged by name across documents), `GET /graph/entity?key=` (mentions as citation-shaped passages, relations with evidence), `POST /documents/{id}/graph` (queue a build: 202; 404 above the caller's level, 409 not indexed). 404 everywhere when `KNOWLEDGE_GRAPH=false` |
| `metrics.py` | `GET /metrics/summary`, `GET /metrics/events`: the evaluation dashboard, **admins only** (`AdminUser`) |
| `users.py` | `GET /users`, `PATCH /users/{id}` (`{role}`), `DELETE /users/{id}`: account administration, **admins only**; 409 for the last administrator and for deleting yourself |
| `analysis.py` | `GET /documents/{id}/insights` (saved results) and the SSE endpoints `POST /analysis/summary`, `/analysis/extract`, `/analysis/compare`. Everything that can fail before the stream starts (unknown or unreadable document 404, not indexed 409, bad fields or document count 422) is an HTTP error; everything after is an `error` event |
| `health.py` | `/health` (liveness, touches nothing) and `/ready` (Postgres, Chroma, and a cached embedding probe). Delegates to `health_service` |

### 3.3 `services/` — workflows

| File | Purpose |
| --- | --- |
| `health_service.py` | Readiness probes run concurrently in worker threads with a timeout; failure details are exception class names only. The embedding probe is a real API call, so a good result is cached 5 minutes and a failure 30 seconds (a free-tier key allows ~100 requests/minute) |
| `index_service.py` | `run_startup`: first finish or discard an interrupted collection rebuild (`vectorstore.recover_interrupted_rebuild`), then open the Chroma collection (refuse a model/dimension/distance mismatch), fail interrupted `processing` documents and requeue `queued` ones, **refuse to start when Postgres has indexed chunks but Chroma is empty** (changed `EMBEDDING_DIM`, lost volume: serving an empty index would silently return nothing), rebuild BM25 from Postgres, start workers |
| `auth_service.py` | bcrypt hashing (work factor `BCRYPT_ROUNDS`), HS256 JWT issue/verify (algorithm pinned, claims `sub`/`iat`/`exp` only), `register_user`, `authenticate`. Signup takes a Postgres advisory lock so concurrent first signups cannot create two admins; unknown-email logins still run a bcrypt check so timing does not reveal which emails exist, and both release the database connection before hashing. Typed errors (`EmailAlreadyRegistered`, `SignupDisabled`, `InvalidCredentials`, `InvalidToken`) are translated to HTTP in `api/` and `deps.py` |
| `document_service.py` | Validate uploads (extension, declared type, magic bytes, size, creation-date plausibility; per-file accept/reject), store under `<uuid><ext>`, create `queued` rows, enqueue ingestion; a failure after some files were stored removes them. Role-aware listing (a document above your level is a 404, never a 403). Delete: park as `failed` first (immediately unsearchable), then BM25 → Chroma → blank the passages copied into saved messages (`messages.citations`) → rows → file; a busy document answers 409; a failed step leaves it retryable |
| `ingestion.py` | Background job (one worker by default): parse → size check (`MAX_DOCUMENT_CHARS`) → chunk → resolve metadata → **chunk rows to Postgres first** → embed → upsert to Chroma → BM25 (heading + content) → `indexed`. Any failure removes partial data from all three stores and sets a user-safe message. Per-document lock shared with delete. Logs ids, counts, timings and, for unexpected errors, where it failed (never the exception message: it can quote document text) |
| `chat_service.py` | Full chat turn: rewrite → filter → **saved-answer lookup** → retrieve → guard → **graph expansion** → generate → verify citations → (conflict check when the answer cites 2+ documents) → persist → store in the cache → stream events. A cache hit (`_serve_cached`) skips everything between the lookup and persisting and re-validates its cited chunks in Postgres first |
| `cache_service.py` | Reads a document's filterable metadata and tells `core/answer_cache` to drop what a change can make wrong: `on_document_indexed` (an `ingestion.on_indexed` hook), `invalidate_document` (delete flow, finished graph build), `evict_message` (a thumbs-down). Never raises; when the metadata cannot be read it clears everything |
| `graph_service.py` | The knowledge graph. **Building:** `submit_graph` (after indexing, when `AUTO_KNOWLEDGE_GRAPH`) and `request_build` (on demand) queue `_run_graph` on the insights thread (one model job at a time, shared with the executive summary and suggested questions); status is a `document_insights` row of kind `graph` (`pending` before the model call), the graph is written in one short transaction after it (old rows cascade away first), one `query_events` row per run. **Reading:** `build_view` (nodes merged by `key` across documents, ranked by connections, capped at `KG_VIEW_MAX_NODES`) and `entity_detail` (mentions and relations as citation-shaped passages); only readable **and** indexed documents contribute and every passage is filtered by the caller's access levels again |
| `conversation_service.py` | Conversations scoped to their owner; `readable_citations` marks citations whose document is gone or above the reader's level `unavailable` (no passage, no name), and `readable_conflicts` drops conflict claims whose source the reader can no longer see; ratings are one atomic upsert |
| `metrics_service.py` | `query_events`: `new_event`/`add_usage`/`add_retrieval`/`record` (one row per chat turn or analysis run, failures included; `record` never raises), `prune` (retention, at startup), `summary` and `list_events` (dashboard, SQL aggregates), `popular_questions` (a question is suggested to others only when `POPULAR_MIN_USERS`+ users asked it and every cited document is readable by the reader) |
| `analysis_service.py` | Summaries, extraction, comparison, the background executive summary and each document's suggested questions (`document_suggestions`, written once on the insights thread). Every stream is wrapped by `_measured` (one event). Access: only documents the user may read **and** that are `indexed`. Saved results live in `document_insights` (written `pending` before the model call; `ready`/`failed` after; a recent `pending` row is waited for, not paid for twice; `recover_interrupted` fails the rest at startup). No database session is held across a model call. The automatic executive summary runs on its own single-thread executor, registered through `ingestion.on_indexed` in `index_service.run_startup` (a hook, so `ingestion` never imports this module) and never fails or delays indexing. Compare narrows the search filter to one document per column and makes no model call for a column the guard stops |

### 3.4 `core/` — RAG building blocks (no web framework imports)

| File | Purpose | Key rules |
| --- | --- | --- |
| `parsers.py` | PDF (page-aware, pdfplumber), DOCX, MD, HTML, TXT → list of `(page, section, text)` | PDF: physical page numbers; sections from the outline (titles may wrap over lines) or font size/boldness; two-column pages read column by column (tables are not split); running headers/footers/banners and bare page labels removed, never more than 30% of the text; hyphenation repaired. Scanned/blank PDFs raise `ScannedPdfError` ("scanned PDF, OCR not supported"). Error messages are user-safe and never contain text. **Every text parser must run in time linear in the input** (no unbounded regex quantifiers rescanning from each opener, no string growth per line: hostile files stalled the whole server before); `tests/test_parsers.py` asserts wall-time ceilings. DOCX reads text inside content controls (`w:sdt`) |
| `chunking.py` | `Chunker` interface + structure-aware implementation (~800 chars, ~100 overlap) | Packs by sentence; never crosses a section; overlap is whole trailing sentences, same page and section only; a chunk is at most target + 1.5×overlap + 1 separator (enforced against `CHUNK_HARD_MAX_CHARS` = 1500, far under the 2,048-token embedding limit); records `page_number`..`page_end`; sentence splitting and long-text cutting are linear; semantic chunker is a future implementation of the same interface |
| `metadata.py` | Resolve department, type, creation date, access level | Order: explicit upload values → folder/filename → file properties → upload date; produces `creation_date_int`. A month- or year-only file name ("April2025") is refined to the exact day by a *consistent* file date. Implausible dates (before 1990, far future) are ignored when read from a file name or file, and refused (422) when passed at upload; labels read from inside a file are clamped to the 100-character columns |
| `llm.py` | Builds the `ChatGoogleGenerativeAI` instance from settings | The **only** place that constructs the LLM client. The client is cached **per running event loop** (one carried into another loop fails its first call with "Event loop is closed"); background jobs run through `run_background`, which gives the job a loop of its own and closes that loop's clients before it ends |
| `embeddings.py` | Wraps `GoogleGenerativeAIEmbeddings`: document vs query task types, batching, retries/backoff, L2 normalization, query LRU cache | The **only** place that calls the embedding API. Walks the exception chain (LangChain wraps the SDK error) to classify 429/quota/auth/input/transient; honours the server's "retry in N s" hint with a cooldown shared by all threads; never retries a bad key or a daily quota. Rejects inputs over 2,000 characters instead of letting the API truncate them. Document batches are spaced to `EMBEDDING_TEXTS_PER_MINUTE` (questions are never held back); `probe()` never sleeps through a cooldown |
| `vectorstore.py` | Chroma client: create/get collection (cosine), upsert, dense query, delete by `document_id` | The **only** place that imports `chromadb`; passes embeddings explicitly. Create collections with `configuration={"hnsw": {"space": "cosine"}}` and `embedding_function=None` (the client's default is Chroma's built-in embedder; verified against 1.5.9). **Never trust a Chroma write**: Chroma can store a vector its index never returns (seen live: 15 of 155 after delete and re-upload). `ensure_reachable(ids)` queries each new vector with its own embedding and repairs misses (re-insert, then `rebuild_collection`); `find_unreachable`/`repair_unreachable` back startup verification and `python -m app.maintenance verify-index`; an id that was never stored counts as unreachable. Call `ensure_reachable` after any code path that writes vectors (ingestion **and** `reembed`). All writers (upsert, delete, repair, rebuild) hold `_write_lock`; the HTTP session has timeouts (chromadb's default is none); `recover_interrupted_rebuild` runs before the collection is first opened |
| `bm25.py` | In-memory BM25 index (`bm25s`), identifier-aware tokenizer, add/remove/rebuild, filtered top-N | Rebuilt from Postgres at startup; filter parity with Chroma via a predicate over the same metadata fields; immutable snapshots swapped atomically so searches never block on ingestion. Indexes `keyword_text(section_title, content)`: the heading is searchable by keyword but is not embedded |
| `filters.py` | `build_filter(user, request_filters)` returning one neutral filter object plus Chroma `where` and SQL translations | **The only place filters are built**; always injects the access-level constraint |
| `fusion.py` | `rrf(rankings, k)` | Pure function; RRF scores are for ordering only |
| `retrieval.py` | Orchestrates embed query → dense + BM25 (parallel) → fusion → optional rerank; returns candidates with dense similarity and timings | Takes the question's vector when the caller already embedded it (the cache does). `load_chunks` and `candidate_from_row` are how other channels (the graph) turn chunk ids into candidates under the same indexed-and-filter check |
| `graph.py` | Entity and relation extraction: batches of numbered `<source>` blocks → JSON → grounded entities (with mentions) and relations | Pure. **Nothing is kept on trust:** an entity must occur, as whole words, in a chunk read (its mentions are every such chunk, not what the model listed); a relation needs a real source number of its batch, both names in *that* chunk, two different entities; a quote that is not in the chunk is blanked; labels and types are normalised. Merged by `normalize_key` (case, punctuation, a leading "the" ignored). Prompt in `prompts.py` (`GRAPH_TASK`), limits in `config.py` (`KG_*`) |
| `graph_retrieval.py` | `expand(seeds, filter)`: chunks reached from the retrieved ones through shared entities and relations (`GRAPH_HOPS` steps, across documents), evidence chunks first, appended behind the seeds | Runs **after** the guard and on the seeds only; added chunks have no similarity, so the guard and the relevance figures never see them. A relation is followed only when its evidence chunk passes the caller's `SearchFilter` and its document is indexed (in the query and again in `load_chunks`). Entities mentioned in more than `KG_MAX_ENTITY_CHUNKS` chunks are skipped |
| `answer_cache.py` | The semantic answer cache: entries keyed by the question vector and a **scope** (filter, model, collection, threshold, top-k, graph, prompt version), cosine ≥ `SEMANTIC_CACHE_THRESHOLD`, TTL and LRU | Pure and in memory (one worker, rule 4). `invalidate_document` drops entries the document is in the filter of or that cite it; `generation` stops a turn that began before an invalidation from storing its answer. Only clean answered turns are stored |
| `rerank.py` | Optional FlashRank cross-encoder (**does not exist yet**) | Behind `RERANK_ENABLED`, which must stay `false` (a `true` value fails settings validation) |
| `query_rewrite.py` | Turns a follow-up into a standalone question using prior turns | Skipped on the first turn; thinking disabled, small token limit. The conversation is untrusted data (earlier answers quote documents); a rewrite far longer than the follow-up falls back to the original question |
| `prompts.py` | System prompt and context-block formatting | Treats context as untrusted data; numbered sources `[1]`, `[2]` |
| `generation.py` | Calls the LLM (streaming), assembles the answer, maps citations to chunks, verifies them, retries once | Handles `MAX_TOKENS`, safety blocks, empty candidates; an answer with nothing left to show (only removed markers) is an `empty` error, never a saved blank message |
| `analysis.py` | Summaries (executive, chapters, FAQ, glossary, key changes), field extraction, conflict detection, suggested questions (`suggest_questions`, `suggest_followups`) | Pure (no FastAPI, no database). Sources are numbered `<source id="n">` and framed as untrusted data; long documents are batched with **global numbering** so citations survive map-reduce. Every item is validated (Pydantic, one retry on bad JSON) and **dropped unless its `[n]` citation maps to a real source**; an extracted quote that is not in its source is blanked; a conflict needs claims from two different documents. Prompts live in `prompts.py`, limits in `config.py` |
| `guard.py` | Relevance gate (cosine or reranker score, plus lexical override) and citation verification helpers | Never thresholds on the RRF score. The lexical override applies only down to `RELEVANCE_THRESHOLD - LEXICAL_OVERRIDE_MARGIN` and only to a passage the model is given. `CitationFilter` treats `[n]`, `[n, m]` and a dash range between two real sources as citations; other brackets and anything inside code are left alone; a removed marker takes its preceding space and nothing else changes |

### 3.5 `db/`

| File | Purpose |
| --- | --- |
| `database.py` | SQLAlchemy engine and session factory |
| `models.py` | `users`, `documents`, `chunks`, `conversations`, `messages` (incl. `conflicts`), `feedback`, `document_insights`, `query_events`, `kg_entities`, `kg_mentions`, `kg_relations` (the graph; all cascade from the document and its chunks) (columns defined in `TDD.md` §4.3). Enum-like columns are `VARCHAR` + named `CHECK` (not PG enums); cascades are `ON DELETE` in the database; constraint names come from `NAMING_CONVENTION` in `database.py`. Every change needs a new migration, and `tests/test_migrations.py` fails if models and migrations disagree |

---

## 4. How the System Works

### 4.1 Ingestion flow (upload → searchable)

1. `POST /api/documents` → `api/documents.py` validates type/size, calls `document_service`.
2. `document_service` stores the file under a generated UUID name, creates a `documents` row with status `queued`, schedules `ingestion`.
3. `ingestion` sets `processing`, then: `parsers` → `chunking` → `metadata` → insert `chunks` rows in Postgres.
4. `embeddings.embed_documents` (task `RETRIEVAL_DOCUMENT`, batched, normalized) → `vectorstore.upsert` (flat metadata, explicit vectors).
5. `bm25` index is updated; document becomes `indexed` with `chunk_count`.
   Then the after-indexing hooks run (they only queue work): the cache drops saved answers the new document can change, the executive summary and the knowledge graph are queued on the insights thread.
6. Any failure: partial chunks removed from Chroma and Postgres, status `failed` with `error_message`. Scanned PDFs fail with "scanned PDF, OCR not supported".

### 4.2 Chat flow (question → cited answer)

1. `POST /api/chat` → `chat_service` (streams SSE).
2. If the conversation has prior turns → `query_rewrite` makes a standalone question.
3. `filters.build_filter` combines user filters with the role's access levels.
3a. **Saved answer (`SEMANTIC_CACHE`).** `embeddings.embed_query` on the (rewritten) question, then `answer_cache.lookup` in the scope of this filter. On a hit whose cited chunks are all still indexed and readable (checked in Postgres): emit `cache_hit`, the stored answer, citations, conflicts and suggestions, persist it as a normal message, record one `query_events` row, done: no search, no model call. A follow-up whose rewrite changed nothing skips the cache.
4. `embeddings.embed_query` (task `RETRIEVAL_QUERY`; already done in 3a, the vector is passed on) → in parallel: `vectorstore` dense top 30 and `bm25` top 30, both filtered.
5. `fusion.rrf` merges; optional `rerank`; keep top K (default 5).
6. `guard`: if best cosine (or reranker) score is below `RELEVANCE_THRESHOLD` and the lexical override does not apply → emit `REFUSAL_MESSAGE`, **do not call the LLM**.
6a. **Graph expansion (`GRAPH_RETRIEVAL`, after the guard has passed).** `graph_retrieval.expand` follows the entities of the retrieved passages along their relations (up to `GRAPH_HOPS`, across documents) and appends up to `GRAPH_EXPANSION_MAX_CHUNKS` linked passages behind them, under the same access filter. A failure costs only the extra context.
7. Otherwise `prompts` + `generation` stream tokens; citations are mapped to chunks and verified; one retry on invalid citations, then invalid markers are stripped and the message is flagged.
8. If `CONFLICT_DETECTION` is on and the cited chunks come from **two or more different documents**, `analysis.detect_conflicts` checks whether they contradict each other (one extra model call; never for a single-document answer or a refusal; a failure is logged and ignored, never costs the answer).
   In parallel with it (`asyncio.gather`), if `FOLLOWUP_SUGGESTIONS` is on and the turn was answered, `analysis.suggest_followups` proposes follow-up, sub- and related questions, each naming a retrieved passage (one extra call; a failure is ignored).
9. Persist the message (answer, citations, conflicts, suggestions, guard score, timings); emit `citations`, `conflicts` and `suggestions` (only if any) and `done`.
10. Record one `query_events` row (`metrics_service.record`); a failed turn is recorded by `run_chat` with its error code.
11. A clean answered turn (not refused, not flagged, no extra that failed, at least one citation) is stored in the answer cache, unless a document was indexed or deleted since the turn began searching.

### 4.3 Delete flow

`DELETE /api/documents/{id}` → permission check (admin or owner) → park the document as `failed` (immediately unsearchable) → drop the saved answers that cite it or that it could change (`cache_service.invalidate_document`) → remove it from BM25 → delete Chroma vectors by `document_id` → blank its passages in saved `messages.citations` → delete the saved analyses that quote it (`document_insights` citing it or comparing against it) → delete Postgres rows (chunks, its own insights and its part of the knowledge graph cascade) → delete the file. If a step fails, the document stays parked and the delete can be repeated; it must never be left half-searchable. (`documents.py`: 404 for a document above the caller's level, 403 only after it is visible, 409 while it is being indexed.)

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
| `POST /chat` | user | `api/chat.py` | SSE stream: `status`, `token`, `citations`, `conflicts`, `suggestions`, `done`, `error` |
| `GET /suggestions` | user | `api/suggestions.py` | Questions from the chosen documents and often asked ones |
| `GET /metrics/summary`, `GET /metrics/events` | admin | `api/metrics.py` | Evaluation dashboard |
| `GET /users`, `PATCH /users/{id}`, `DELETE /users/{id}` | admin | `api/users.py` | List accounts, set a role, delete an account |
| `GET /documents/{id}/insights` | user | `api/analysis.py` | Saved summaries and extractions of a readable document |
| `POST /analysis/summary` | user | `api/analysis.py` | SSE: executive, chapters, FAQ, glossary or key-changes summary (saved; `refresh` regenerates) |
| `POST /analysis/extract` | user | `api/analysis.py` | SSE: fields (presets or custom) extracted into cited values |
| `POST /analysis/compare` | user | `api/analysis.py` | SSE: one cited answer per document (2 to `COMPARE_MAX_DOCUMENTS`), then conflicts |
| `GET /graph` | user | `api/graph.py` | The knowledge graph of the chosen documents (or all readable ones): nodes, edges, per-document build status |
| `GET /graph/entity` | user | `api/graph.py` | One entity: mentions (citation-shaped passages) and relations with their evidence |
| `POST /documents/{id}/graph` | user | `api/graph.py` | Queue building or rebuilding a document's graph (202) |
| `GET /conversations` | user | `api/conversations.py` | Own conversations |
| `GET /conversations/{id}` | user (owner) | `api/conversations.py` | Messages |
| `PATCH /conversations/{id}` | user (owner) | `api/conversations.py` | Rename |
| `DELETE /conversations/{id}` | user (owner) | `api/conversations.py` | Delete |
| `POST /messages/{id}/feedback` | user (owner) | `api/conversations.py` | Thumbs up/down |
| `GET /health` | none | `api/health.py` | Liveness |
| `GET /ready` | none | `api/health.py` | Readiness |

**Access levels:** `admin` → Public, Internal, Restricted. `user` → Public, Internal.

### Frontend routes

| Path | Page | Access |
| --- | --- | --- |
| `/login`, `/signup` | `LoginPage` (`mode="login"` / `"signup"`) | public |
| `/` | redirect to `/chat` | protected |
| `/chat`, `/chat/:conversationId` | `ChatPage` | protected |
| `/library` | `LibraryPage` | protected |
| `/analyze` (`?tab=compare\|extract\|summaries\|graph&doc=<id>`) | `AnalyzePage` | protected |
| `/dashboard` | `DashboardPage` (evaluation) | admins (others are redirected to `/chat`) |
| `/users` | `UsersPage` | admins (others are redirected to `/chat`) |
| `*` | `NotFoundPage` | public |

nginx serves the SPA (with fallback to `index.html`, which is never cached; hashed `/assets/` are cached for a year and a missing one is a 404), proxies `/api` to `backend:8000`, disables buffering for `POST /api/chat` and `/api/analysis/*` (5 minute timeout), limits request bodies per route and request rates for sign-in, chat, analysis and uploads, and checks the sign-in (`auth_request`) before it accepts an upload body.

---

## 6. Frontend File Purposes

What exists (the implementation is consolidated; there are no separate `sse.ts`, `auth.ts`, `documents.ts`, `chat.ts`, `useDocuments`, `useChatStream`, `CitationChip` or `RefusalNotice` files):

| Area | Files | Purpose |
| --- | --- | --- |
| `api/` | `client.ts` | `fetch` wrapper (base URL, JWT header, 401 → logout, errors turned into readable sentences by `describeDetail`), `streamSse`, the one event-stream reader (it resolves only after a `done` or `error` event and reports a stream that ends without one), `streamChat` on top of it for `/chat`, and `uploadDocuments`, which uses `XMLHttpRequest` because `fetch` cannot report upload progress |
| `pages/LibraryPage` upload flow | `useUploads` | Per-file progress while bytes are sent (one request; the bytes are spread over the files by size), then each accepted document is polled (`GET /documents/{id}`) through queued → indexing → ready or failed; refused files show the server's reason; the upload can be cancelled; toasts announce results |
| `hooks/` | `useAuth.tsx` | Auth context: login, signup, logout. Logging out clears the TanStack cache; only a 401 drops the session (a network error shows "cannot reach the server" instead) |
| | `useAnalysisRun.ts`, `useIndexedDocuments.ts` | One streamed summary or extraction at a time (progress text, result, error; a run that is no longer current never touches the state); the searchable documents, shared with chat |
| `pages/` | `LoginPage` | Sign in and sign up (by `mode`) |
| | `LibraryPage` | Upload dropzone (keyboard-operable), document table (polls while any document is `queued`/`processing`), status badges, delete confirmation dialog |
| | `ChatPage` | Conversation sidebar, message list, filter bar, input, source drawer, the thumbs up/down and refusal notice, the "Sources disagree" panel (live and from saved `message.conflicts`). A per-question id makes a stream that is no longer current inert (switching conversations, Stop), and the input stays busy until the stored conversation is reloaded |
| | `AnalyzePage` | Tabs Compare / Extract / Summaries / Graph (tab and document in the address; the tab row scrolls inside itself on a phone); all four stay mounted so work in progress survives a tab change; one `SourceDrawer` for every citation |
| `components/` | `ErrorBoundary` | A render error shows a way out instead of a blank page |
| `components/chat/` | `Answer`, `CitationChip`, `ConflictPanel`, `FilterBar` | Text with clickable citation chips (also inside bold text; unavailable sources are shown as such), lists; one chip; each document's conflicting claim with its citation; pre-retrieval filters |
| `components/analyze/` | `CompareTab`, `ExtractTab`, `SummariesTab`, `DocumentPicker` | Side-by-side columns that stream per document ("Not covered in this document" when the guard stops one; columns left unfinished by an error say so); field presets, custom fields, table / JSON view, JSON and CSV export (`lib/analysis.ts`: CSV cells that look like formulas are neutralised); five summary cards with saved status, Regenerate, Copy as Markdown |
| `components/analyze/GraphTab` + `components/graph/` | `GraphTab`, `ForceGraph`, `layout.ts`, `entityTypes.ts` | Document chips with a status dot and a Build / Rebuild button per document (the cost is stated; nothing is built unasked in the tab), type chips, name search, the SVG graph (a cooling force layout in plain TypeScript, so no library and no worker under the CSP; wheel zoom, drag to pan, drag a node, `prefers-reduced-motion` settles it at once), a keyboard-operable entity list, and a detail panel: description, relations with quotes and "Open the source", mentions; every passage opens in the source drawer. The hook is `hooks/useKnowledgeGraph.ts` (polls while a build is pending) |
| `components/source/` | `SourceDrawer` | Opens on citation click: highlighted chunk, document name, page, section, embedded original (`#page=N` for PDFs; other types show chunk + download link); Escape closes it |

---

## 7. Rules (Invariants — do not break)

### 7.1 Architecture

1. **Layering:** `api → services → core / db`. `core/` never imports FastAPI. `api/` never calls Chroma, BM25, or the LLM directly.
2. **Single points of contact:** only `vectorstore.py` imports `chromadb`; only `embeddings.py` calls the embedding API; only `llm.py` builds the LLM client; only `config.py` reads environment variables; only `filters.py` builds filters.
3. **PostgreSQL is the source of truth.** Chroma and BM25 are derived and rebuildable. Write chunks to Postgres before Chroma.
4. **Single backend worker** (the BM25 index is in-process). Do not add `--workers`. `docker-entrypoint.sh` pins `--workers 1` and refuses to start when `WEB_CONCURRENCY`/`UVICORN_WORKERS` ask for more (`tests/test_entrypoint.py`).
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
24. Deleting a document must remove vectors, rows, and file; its content must never appear in later answers. That includes the passages copied into saved conversations (blanked on delete) and, for a reader who may no longer see a document, in reopened conversations (`unavailable`). The answer text of an old message is kept as a record.
25. Passwords: bcrypt, at most 72 bytes (rejected at signup, never silently truncated). Tokens: short-lived HS256 JWT carrying only the user id. Only nginx publishes a port; Postgres and Chroma stay internal. Compose enforces this with an `internal: true` network (Postgres, Chroma, backend) and an `edge` network (nginx, backend; gives the backend outbound access to Gemini). Postgres and Chroma have no route to the internet and nginx cannot resolve them. The port is published on `BIND_ADDR` (default `127.0.0.1`).
25a. **Hold no database connection across slow work.** A request must not keep a session open while it streams an answer, hashes a password or waits for the embedding API; end the transaction first (`deps.get_current_user`, `auth_service.authenticate`). `tests/test_conversations_api.py` checks Postgres for `idle in transaction` mid-stream.
25b. **Text parsing is linear in the input** (see `core/parsers.py` in section 3.4) and a document is bounded by `MAX_DOCUMENT_CHARS`.

### 7.6 Docker

26. Four services: `frontend` (nginx), `backend`, `chroma`, `postgres`. Named volumes `pg_data`, `chroma_data`, `uploads`.
27. Healthchecks plus `depends_on: condition: service_healthy`; the backend runs `alembic upgrade head` on start.
28. Pin the ChromaDB image and client library versions together (`chroma/Dockerfile` tag == `chromadb-client` in `requirements.txt`). This cannot be checked at runtime (the server's `/api/v2/version` returns a fixed `"1.0.0"`), so `tests/test_pinned_versions.py` enforces it; bump both in one change. **That test is skipped inside the backend image** (it has no `chroma/` directory): run it with the directory mounted (README, Development) before any change to either pin.
29. No embedding model is baked into the image; only the optional FlashRank model may be.
30. Containers run as non-root: backend uid 10001, chroma uid 10001 (via `chroma/Dockerfile`, which pre-creates `/data` so the named volume inherits ownership), nginx via `nginx-unprivileged` (host port 8081, container port 8080), postgres via `user: postgres`.

### 7.7 Analysis (summaries, extraction, comparison, conflicts)

31. **Analysis output is grounded like an answer.** Every summary item, extracted value, comparison answer and conflict claim must carry a citation that maps to a real source; the model's `[n]` markers are validated (`guard.filter_citations`, source-number checks) and an item without a valid citation is **dropped, never shown**. Do not add an analysis output that skips this, and keep the sources framed as untrusted data (rule 10).
32. **Only readable, indexed documents are analysed.** Every analysis entry point goes through `analysis_service` (`document_service.get_document` plus `indexed`): a document above the caller's level is a 404, and Compare narrows the search filter to one document per column so a column can only cite its own text. A change that could bypass it needs a test in `test_analysis_api.py`.
33. **Saved results are derived data that must follow deletion** (rule 24): `document_insights` rows cascade with their document, and results that quote or compare against it are deleted by `document_service`. Citations in saved results and conflicts are re-checked at read time.
34. **Extras never cost the answer, and nothing holds a connection through a model call.** The conflict check, the follow-up suggestions and the automatic executive summary may fail without failing the answer or the upload (log the exception class only); analysis code opens a short session per step (rule 25a). A new analysis kind needs a prompt in `prompts.py`, a validated output model in `core/analysis.py`, an `InsightKind`, and tests for its grounding rules.

### 7.8 Measurement and suggestions

35. **Every chat turn and analysis run records exactly one `query_events` row, failures included** (`metrics_service.record`, after the answer is stored, in its own short session). Recording never fails the request, never stores document text, and the dashboard and its question text are for admins only. A new request type must be measured the same way (`analysis_service._measured` for streams).
36. **A suggested question must name a real passage that answers it** (document questions and follow-ups alike; anything else is dropped), and a question someone asked is suggested to another user only when `POPULAR_MIN_USERS` (≥ 2) different users asked it and every document its answer cited is readable by that user. Suggestions from a deleted or unreadable document are never shown.

### 7.9 Accounts

37. **There is always at least one administrator, and one sign-in page serves everyone** (no separate admin portal: the role, re-read from Postgres on every request, decides). Anything that creates, promotes, demotes or removes accounts runs under `auth_service.lock_accounts` (the signup advisory lock), so concurrent requests cannot leave zero administrators; `tests/test_users_api.py` widens the race on purpose and fails without the lock. An administrator cannot delete their own account. Admin-only routes use `AdminUser`; admin-only screens must also redirect non-admins.

### 7.10 Saved answers, the knowledge graph and background jobs

38. **A saved answer is served only inside its scope and only while its sources stand.** The scope is the full `SearchFilter` (access levels included) plus the model, collection, threshold, top-k, graph setting and `PROMPT_VERSION`, so two roles never share an entry. Every hit re-checks in Postgres that each cited chunk still exists, is indexed and passes the reader's filter. Indexing a document, deleting one (right after it is parked, before anything is scrubbed), finishing a graph build and a thumbs-down all invalidate (`cache_service`); a turn that began before an invalidation cannot store its answer (`AnswerCache.generation`). Only clean answered turns are stored (never a refusal, a flagged answer or one whose extras failed). Anything that can change what an answer says needs a test in `test_semantic_cache.py` showing the stale answer is not served.
39. **Knowledge-graph facts are grounded like answers, and the graph never widens access.** An entity is kept only if its name occurs as whole words in a chunk read, and its mentions are the chunks that contain it, not what the model claimed; a relation needs both names in the chunk it cites, a real source number and two different entities; a quote that is not in the chunk is blanked (`core/graph.py`, `tests/test_graph_core.py`). Reading the graph (`graph_service`) and walking it (`graph_retrieval`) use only indexed documents the caller may read and filter every passage by their access levels; a relation stated in an unreadable passage is not followed. Expansion runs **after** the guard, on the seeds only, adds chunks without a similarity score, and is never used to rescue a refusal (rules 7 and 8). A change that could bypass any of this needs a test in `test_graph_api.py` / `test_graph_retrieval.py`.
40. **Background model jobs run through `llm.run_background`.** The Gemini async client belongs to the event loop that opened its connections; `llm.py` caches it per running loop and `run_background` closes a job's clients before its loop ends. Never call `asyncio.run` on a model coroutine from a background thread, and never keep a client of your own.

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

* **Database tests** run against a throwaway database (`groundedrag_test_<random>`) that the fixtures create on the stack's Postgres, migrate with the real Alembic migrations, and drop afterwards; they skip when no server is reachable (unless `REQUIRE_STACK=1`, which turns that into a failure: use it wherever a green run has to mean something) and refuse to truncate any other database. `conftest.py` drops every setting from the environment except the infrastructure wiring, so a `.env` override cannot change a test's outcome. New concurrency-sensitive code needs a test that **fails when the protection is removed** (see the signup-lock test, which widens the race window on purpose); mutate a copy of the code to check.
* **Index-health tests simulate the symptom** (a wrapped `query` that never returns chosen vectors) because real Chroma cannot be made to lose vectors on demand; detection, re-insertion and rebuilding run for real. Scripts that drop or rebuild collections must force a scratch `CHROMA_COLLECTION` name first: run against `.env` they would hit the real collection.
* **Real services in tests:** vector store, ingestion, startup and API tests run against the stack's real Chroma (a unique `test_<id>` collection per run) and a throwaway Postgres database, with a deterministic bag-of-words fake embedder (`FakeEmbeddings` in `conftest.py`). Startup is neutralized by an autouse fixture; call `REAL_RUN_STARTUP` to test it.
* **Unit tests:** parsers, chunking (page numbers preserved, no mid-sentence cuts, overlap, token ceiling), filter building, RRF, guard (including lexical override), citation verification, access-level filtering.
* **Integration test:** `tests/test_chat.py` uploads a small sample document → waits for `indexed` → queries → asserts a valid citation. Mock the LLM and embeddings in tests (no network calls, no API key needed).
* **Coverage that must not erode:** every route except the public ones refuses a missing or bad token (`test_conversations_api.py` reads the route list from the OpenAPI schema, so a new route is covered automatically); conversations and feedback are scoped to their owner; text parsers have wall-time ceilings on hostile input.
* **Evaluation (`eval/run_eval.py`):** hit-rate@5 and MRR for dense-only, keyword-only and hybrid (hybrid + rerank once reranking exists), plus `hybrid+graph` and the multi-hop count once a knowledge graph has been built; the guard decided by the production `retrieval.retrieve` + `guard.evaluate`; per-stage latency p50/p95. The set includes exact-token questions, 12 unanswerable ones and 3 multi-hop ones (`eval/README.md`; a question whose documents are not indexed is skipped); `test_eval_corpus.py` keeps questions and corpus in step. A change to graph extraction or expansion is compared with `RETRIEVAL_TOP_K=1` (plain search too narrow to find both documents) as well as the default.
* **The saved-answer cache and the graph are off or neutral in the old tests:** `conftest.py` sets `SEMANTIC_CACHE=false` (a repeated question would skip the scripted model) and `AUTO_KNOWLEDGE_GRAPH=false` (an upload would queue model calls); `test_semantic_cache.py` switches the cache on, `test_graph_api.py` builds graphs on request. Graph builds use the same scripted model as answers, so a test that counts answer calls must look only at prompts that contain a `QUESTION` block (see `answer_prompts` in `test_graph_retrieval.py`). The exit status fails on off-topic leakage, a wrongly stopped answerable question, or a low hybrid hit-rate.
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
| Add or change a summary kind, extraction preset or conflict prompt | `core/prompts.py`, `core/analysis.py` (output model, `PRESETS`), `constants.InsightKind` (+ a migration if the `CHECK` changes), `schemas.py`, the frontend `SummariesTab` / `ExtractTab` and `lib/analysis.ts` |
| Analysis limits (tokens, batch size, fields, documents), automatic summary, conflict check | `.env` / `config.py` (`ANALYSIS_*`, `AUTO_EXECUTIVE_SUMMARY`, `CONFLICT_DETECTION`, `COMPARE_MAX_DOCUMENTS`, `EXTRACTION_MAX_FIELDS`) |
| Saved-answer cache: on/off, similarity threshold, lifetime, size | `.env` / `config.py` (`SEMANTIC_CACHE*`). Change the answer prompt, citation rules or context format: bump `prompts.PROMPT_VERSION` so old saved answers are never served |
| Knowledge graph: on/off, build after upload, how much is read per call, entities kept | `.env` / `config.py` (`KNOWLEDGE_GRAPH`, `AUTO_KNOWLEDGE_GRAPH`, `KG_*`) |
| Graph retrieval: on/off, hops, passages added, entities too common to follow | `.env` / `config.py` (`GRAPH_RETRIEVAL`, `GRAPH_HOPS`, `GRAPH_EXPANSION_MAX_CHUNKS`, `KG_MAX_ENTITY_CHUNKS`); re-run the eval (also with `RETRIEVAL_TOP_K=1`) |
| Add an entity type or change the graph prompt | `constants.EntityType` (+ a migration, the `CHECK` lists the types), `core/prompts.py` (`GRAPH_TASK`), `core/graph.py`, frontend `components/graph/entityTypes.ts` |

---

## 11. Commands

```bash
# first run
cp .env.example .env              # set GOOGLE_API_KEY, JWT_SECRET, DB password
docker compose up --build         # app at http://localhost:8081

# development with hot reload (the override also mounts eval/, which the plain stack does not contain)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# tests, lint and evaluation
docker compose exec backend pytest                    # REQUIRE_STACK=1 makes a missing Postgres/Chroma a failure
docker compose exec backend ruff check . && docker compose exec backend ruff format --check .
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec backend python -m eval.run_eval

# host port 8081 already in use? Set HTTP_PORT in .env (e.g. HTTP_PORT=8082)
# reachable from other machines? Set BIND_ADDR=0.0.0.0 (read README, "Accounts and roles", first)

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
5. An unrelated question returns the refusal sentence **without calling the LLM** (the follow-up rewrite, which runs before the guard on a conversation's later turns, is the one LLM call that can precede it).
6. A department filter returns zero chunks from other departments; a `user` cannot retrieve `Restricted` documents.
7. Deleting a document removes it from the library and its content never appears in answers again.
8. After `docker compose down` and `up`, documents, users, and conversations persist.
9. `pytest` passes; the eval script prints the comparison table.

---

## 13. Known Limitations

* Scanned PDFs (no extractable text) are rejected; OCR is not supported.
* Single backend worker (in-memory BM25 index, and the in-memory saved-answer cache).
* The saved-answer cache is lost on restart and is only as good as its invalidation (a re-check on every hit is the backstop). A question must be near-identical (cosine ≥ 0.95 on the standalone question); a follow-up whose rewrite changed nothing is not cached. Answers that carry a conflict panel or follow-ups are cached with them, so those can be stale for a time after the model would have said something else; a thumbs-down evicts.
* The knowledge graph costs about one model call per 30,000 characters per document (a 1,500-chunk manual is about 25 calls: it is read at evenly spaced parts beyond `KG_MAX_BATCHES`, and `coverage` says how much). Documents indexed before the feature existed have no graph until someone presses Build; on a free-tier key the builds queue behind the other background jobs and a Gemini 503 fails one (it is saved as `failed` and can be retried). Entities are merged **by name only**: two different things called the same are one node, and a synonym ("the Registrar", "Registrar's Office") is another. The model can still miss relations; what it does report is checked against the text, but a stated relation can be a misreading of that text.
* Graph retrieval adds at most `GRAPH_EXPANSION_MAX_CHUNKS` passages and can add noise to an answer whose seeds were already complete; on the small evaluation corpus it changed nothing at the default `RETRIEVAL_TOP_K` and recovered all three multi-hop questions when plain search was narrowed to one passage.
* Document text and questions are sent to Google's Gemini API; use a paid-tier key and review current terms before uploading confidential documents.
* `gemini-2.5-flash` is closed to new API keys (404 "no longer available to new users"), so the default is `gemini-3.8-flash`; keep the model ID in configuration and re-run the eval before migrating.
* Free-tier Gemini keys have per-minute and per-day limits (embeddings and chat). Chat retries a rate limit briefly, then shows a friendly error.
* Reranking (`core/rerank.py`) is not implemented; `RERANK_ENABLED` must stay `false`.
* Follow-up suggestions add one model call per answered question (`FOLLOWUP_SUGGESTIONS=false` turns them off), and a document's suggested questions cost one call the first time it is selected. The dashboard's groundedness figures come from citation checks, not from an LLM judge, and it shows live traffic only; `eval/run_eval.py` remains the labelled, offline measurement. `tests/conftest.py` switches follow-ups off for the existing chat tests (they script every model reply); `tests/test_suggestions.py` switches them on.
* The analysis features cost model calls (TDD §16.6): one per upload (executive summary), one per summary or extraction (more for very long documents), one per covered document plus a conflict check for a comparison, and one extra for a chat answer that cites two or more documents. On a free-tier key, and when Gemini answers 503 for a while, an analysis can fail with the friendly "temporarily unavailable" error; it is saved as `failed` and can be retried. A comparison that fails part-way keeps the columns it already finished. Conflict detection is a model judgment over the cited passages only: it can miss a contradiction in text that was not retrieved.
* The first account to sign up is the administrator and signup is open until `SIGNUP_ENABLED=false`; there is no TLS, no account lockout and no per-user quotas. nginx rate limits are per client address, which behind Docker Desktop can be one address for everybody.
* The sign-in token is kept in `localStorage`; the Content-Security-Policy in `frontend/nginx-snippets/security-headers.conf` is the mitigation. It allows `blob:` for the PDF drawer and has not been exercised in every browser's PDF viewer.
* Headings are searchable by keyword but are not part of the embedded text, so a purely semantic question about a heading's wording can miss (changing that means re-embedding and recalibrating the threshold).
* The text of an old assistant message is kept when the document it cited is deleted (the passages and citation chips are not).
* The conversation sidebar is hidden below the `md` breakpoint; there is no conversation list on a phone-sized screen.
* The UI has no automated tests (no JS test framework is installed); its logic was checked by build-time type checking and by running the SSE reader and answer renderer under Node.
* ChromaDB has no per-user authorization; it must stay on the internal Docker network.
