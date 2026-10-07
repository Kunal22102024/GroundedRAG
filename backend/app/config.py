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

    llm_model: str = "gemini-2.5-flash"
    rewrite_model: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_thinking_budget: int = Field(default=0, ge=0)

    # --- Retrieval and grounding -------------------------------------------
    retrieval_candidates: int = Field(default=30, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    relevance_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    rerank_enabled: bool = False

    # --- Ingestion ----------------------------------------------------------
    chunk_target_chars: int = Field(default=800, ge=200)
    chunk_overlap_chars: int = Field(default=100, ge=0)
    max_upload_mb: int = Field(default=50, ge=1)
    # Documents indexed at the same time. 1 keeps embedding API usage predictable.
    ingestion_workers: int = Field(default=1, ge=1, le=8)

    # --- Infrastructure -----------------------------------------------------
    database_url: str
    chroma_host: str = "chroma"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_collection: str = "chunks"
    upload_dir: Path = Path("/data/uploads")

    # --- Authentication and logging -----------------------------------------
    jwt_secret: SecretStr
    jwt_expire_minutes: int = Field(default=60, ge=1)
    # bcrypt work factor. 12 costs roughly 250 ms per hash; tests lower it.
    bcrypt_rounds: int = Field(default=12, ge=4, le=15)
    log_level: str = "INFO"

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
        # A chunk can be the target plus up to 1.5x the overlap (CLAUDE.md rule 16: stay far
        # below the embedding model's 2,048-token input limit instead of relying on truncation).
        longest = self.chunk_target_chars + int(self.chunk_overlap_chars * 1.5)
        if longest > CHUNK_HARD_MAX_CHARS:
            raise ValueError(
                f"CHUNK_TARGET_CHARS plus 1.5 x CHUNK_OVERLAP_CHARS must not exceed {CHUNK_HARD_MAX_CHARS}"
            )
        return self

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
