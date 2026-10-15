"""Application settings.

This is the only module that reads environment variables (CLAUDE.md rule 2).
Model IDs, dimensions and thresholds live here and are never hardcoded
elsewhere (rule 5). Defaults mirror the configuration reference in TDD section 10.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import CHUNK_HARD_MAX_CHARS

_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_MIN_JWT_SECRET_CHARS = 32


class Settings(BaseSettings):
    """Typed settings loaded from environment variables (case-insensitive names)."""

    # Empty values (e.g. `REWRITE_MODEL=` copied from .env.example) mean "unset".
    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    # --- Gemini -----------------------------------------------------------
    google_api_key: SecretStr = SecretStr("")

    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = Field(default=768, ge=128, le=3072)
    embedding_batch_size: int = Field(default=32, ge=1, le=100)
    embedding_max_concurrency: int = Field(default=4, ge=1, le=32)
    # Retries per batch for rate limits (429) and transient (5xx/network) errors.
    embedding_max_retries: int = Field(default=6, ge=0, le=20)
    # Document texts embedded per minute, so a large upload leaves headroom for chat questions
    # (a free-tier key allows about 100 requests per minute). 0 turns the limiter off.
    embedding_texts_per_minute: int = Field(default=90, ge=0)

    llm_model: str = "gemini-3.8-flash"
    rewrite_model: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_thinking_budget: int = Field(default=0, ge=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # A user is waiting on a chat answer: rate limits and transient errors are retried only this
    # often, waiting at most this long once (the server's own retry hint is honoured up to that).
    interactive_max_retries: int = Field(default=2, ge=0, le=6)
    interactive_max_wait_seconds: float = Field(default=30.0, ge=0)

    # --- Analysis: summaries, extraction, comparison, conflicts ---------------------------
    # Output budget of one analysis call (a summary or a table of fields is longer than an answer).
    analysis_max_tokens: int = Field(default=4096, ge=256, le=16384)
    # Summaries, extraction and comparison are long-running requests with a progress display, so
    # they wait out a busy or briefly unavailable model for longer than a chat turn does.
    analysis_max_retries: int = Field(default=4, ge=0, le=10)
    # A document with more text than this is analysed in batches and the notes are combined (map-reduce).
    analysis_max_source_chars: int = Field(default=120_000, ge=5_000)
    # Executive summary of every document right after it is indexed (one model call per upload).
    auto_executive_summary: bool = True
    # After an answer that cites two or more documents, check whether they contradict each other.
    conflict_detection: bool = True
    compare_max_documents: int = Field(default=4, ge=2, le=8)
    extraction_max_fields: int = Field(default=20, ge=1, le=50)

    # --- Suggestions ------------------------------------------------------------------------
    # After every answered chat turn, one extra call proposes follow-up, sub- and related questions.
    followup_suggestions: bool = True
    # Suggested questions written for a document the first time it is selected (one call, saved).
    suggestions_per_document: int = Field(default=6, ge=1, le=12)
    # A past question is suggested to other people only once this many different users asked it.
    popular_min_users: int = Field(default=2, ge=2)

    # --- Knowledge graph ----------------------------------------------------------------------
    # Entities (people, policies, courses, concepts...) and the relationships between them are
    # extracted from each document with the model, linked to the chunks they come from, shown in
    # Analyze > Graph, and (GRAPH_RETRIEVAL) used to pull related passages into an answer.
    knowledge_graph: bool = True
    # Build the graph of every document right after it is indexed: about one model call per
    # KG_BATCH_CHARS of text. False: only when someone asks for it in the Graph tab.
    auto_knowledge_graph: bool = True
    # Text read per model call, at most this many calls per document (a longer document is read
    # at evenly spaced parts), and what one call may return.
    kg_batch_chars: int = Field(default=30_000, ge=5_000, le=200_000)
    kg_max_batches: int = Field(default=25, ge=1, le=200)
    kg_max_entities_per_batch: int = Field(default=40, ge=5, le=200)
    kg_max_relations_per_batch: int = Field(default=60, ge=5, le=300)
    kg_max_output_tokens: int = Field(default=8192, ge=1024, le=32768)
    # Entities kept per document (those mentioned in most chunks), and nodes drawn at once.
    kg_max_entities: int = Field(default=400, ge=10, le=5_000)
    kg_view_max_nodes: int = Field(default=150, ge=10, le=1_000)
    # GraphRAG: after the guard has passed, passages connected to the retrieved ones through shared
    # entities and their relations (up to GRAPH_HOPS steps, across documents) are added to what the
    # model is given, at most GRAPH_EXPANSION_MAX_CHUNKS of them. The guard never sees them.
    graph_retrieval: bool = True
    graph_hops: int = Field(default=2, ge=1, le=3)
    graph_expansion_max_chunks: int = Field(default=3, ge=1, le=10)
    # An entity mentioned in more chunks than this ("the university") connects everything to
    # everything and is not followed.
    kg_max_entity_chunks: int = Field(default=40, ge=1)

    # --- Semantic answer cache --------------------------------------------------------------
    # A clean, cited answer is reused for a near-identical question asked under the same access
    # level, filters and document selection (no retrieval, no model call). Kept in memory only.
    semantic_cache: bool = True
    # Minimum cosine similarity between the two standalone question vectors. Paraphrases of one
    # question score above about 0.95; neighbouring questions ("sick leave" / "annual leave") can
    # reach 0.9, so do not lower it without running the evaluation.
    semantic_cache_threshold: float = Field(default=0.95, ge=0.85, le=1.0)
    semantic_cache_ttl_seconds: int = Field(default=86_400, ge=60)
    semantic_cache_max_entries: int = Field(default=1_000, ge=1, le=100_000)

    # --- Evaluation dashboard ------------------------------------------------------------
    # Measurements (query_events) older than this are deleted at startup.
    metrics_retention_days: int = Field(default=90, ge=1)
    # Price of the LLM in US dollars per million tokens, for the cost estimate (0: tokens only).
    llm_input_usd_per_mtok: float = Field(default=0.0, ge=0.0)
    llm_output_usd_per_mtok: float = Field(default=0.0, ge=0.0)

    # --- Retrieval and grounding -------------------------------------------
    retrieval_candidates: int = Field(default=30, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    # Calibrated with eval/questions.jsonl on gemini-embedding-001 @ 768 dims (TDD 6.3).
    relevance_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    # An exact identifier in the top keyword hit (ERR_404_NET) lets a question through the gate
    # even below the threshold, but never by more than this margin: a question that is nowhere
    # near the documents is refused whatever numbers it contains. Measured on the eval corpus,
    # unrelated questions score about 0.55 and identifier questions 0.67 or more.
    lexical_override_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    rerank_enabled: bool = False

    # --- Ingestion ----------------------------------------------------------
    chunk_target_chars: int = Field(default=800, ge=200)
    chunk_overlap_chars: int = Field(default=100, ge=0)
    max_upload_mb: int = Field(default=50, ge=1)
    # Files accepted in one upload request.
    max_files_per_request: int = Field(default=20, ge=1, le=200)
    # Extracted text beyond this size fails the document instead of tying up the worker.
    max_document_chars: int = Field(default=5_000_000, ge=10_000)
    # Documents indexed at the same time. 1 keeps embedding API usage predictable.
    ingestion_workers: int = Field(default=1, ge=1, le=8)

    # --- Infrastructure -----------------------------------------------------
    # repr=False keeps the password out of any logged or printed settings object.
    database_url: str = Field(repr=False)
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    chroma_host: str = "chroma"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_collection: str = "chunks"
    upload_dir: Path = Path("/data/uploads")

    # --- Authentication and logging -----------------------------------------
    jwt_secret: SecretStr
    jwt_expire_minutes: int = Field(default=60, ge=1, le=10_080)  # at most a week
    # bcrypt work factor. 12 costs roughly 250 ms per hash; tests lower it.
    bcrypt_rounds: int = Field(default=12, ge=4, le=15)
    # False closes self-service signup. While no account exists yet, signup still works so a
    # fresh deployment can create its first (admin) user.
    signup_enabled: bool = True
    log_level: str = "INFO"

    @field_validator("rerank_enabled")
    @classmethod
    def _rerank_not_implemented(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "RERANK_ENABLED=true is not supported: core/rerank.py is not implemented"
            )
        return value

    @field_validator("jwt_secret")
    @classmethod
    def _jwt_secret_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < _MIN_JWT_SECRET_CHARS:
            raise ValueError(f"JWT_SECRET must be at least {_MIN_JWT_SECRET_CHARS} characters")
        return value

    @field_validator("log_level")
    @classmethod
    def _log_level_valid(cls, value: str) -> str:
        level = value.upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")
        return level

    @model_validator(mode="after")
    def _overlap_smaller_than_target(self) -> Self:
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_TARGET_CHARS")
        # A chunk can be the target plus up to 1.5x the overlap plus one separator character
        # (CLAUDE.md rule 16: stay far below the embedding model's 2,048-token input limit
        # instead of relying on truncation).
        longest = self.chunk_target_chars + int(self.chunk_overlap_chars * 1.5) + 1
        if longest > CHUNK_HARD_MAX_CHARS:
            raise ValueError(
                f"CHUNK_TARGET_CHARS plus 1.5 x CHUNK_OVERLAP_CHARS plus 1 must not exceed "
                f"{CHUNK_HARD_MAX_CHARS}"
            )
        if self.retrieval_top_k > self.retrieval_candidates:
            raise ValueError("RETRIEVAL_TOP_K must not exceed RETRIEVAL_CANDIDATES")
        return self

    @property
    def graph_retrieval_active(self) -> bool:
        """Whether answers are widened through the graph (it needs the graph feature itself)."""
        return self.knowledge_graph and self.graph_retrieval

    @property
    def rewrite_model_id(self) -> str:
        """Model used to rewrite follow-ups; falls back to ``llm_model``."""
        return self.rewrite_model or self.llm_model

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def collection_name(self) -> str:
        """Chroma collection name encoding embedding model and dimension.

        Example: ``chunks__gemini-embedding-001__768``. Changing the model or
        dimension therefore selects a different collection instead of silently
        mixing incompatible vectors (TDD 3.3, CLAUDE.md rule 13).
        """
        model = self.embedding_model.removeprefix("models/")
        model = re.sub(r"[^A-Za-z0-9_-]", "-", model)
        return f"{self.chroma_collection}__{model}__{self.embedding_dim}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, validated on first use."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
