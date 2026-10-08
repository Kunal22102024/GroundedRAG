"""Settings: defaults, validation and derived values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

SECRET = "s" * 40


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every variable Settings reads, then set only the required ones."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    return monkeypatch


def make() -> Settings:
    return Settings()  # type: ignore[call-arg]


def test_defaults_match_tdd_configuration_reference(clean_env: pytest.MonkeyPatch) -> None:
    settings = make()
    assert settings.embedding_model == "models/gemini-embedding-001"
    assert settings.embedding_dim == 768
    assert (settings.embedding_batch_size, settings.embedding_max_concurrency) == (32, 4)
    assert settings.llm_model == "gemini-3.8-flash"
    assert (settings.llm_temperature, settings.llm_max_tokens) == (0.0, 1024)
    assert settings.llm_thinking_budget == 0
    assert (settings.retrieval_candidates, settings.retrieval_top_k) == (30, 5)
    assert settings.rrf_k == 60
    assert settings.relevance_threshold == 0.60
    assert settings.rerank_enabled is False
    assert (settings.chunk_target_chars, settings.chunk_overlap_chars) == (800, 100)
    assert settings.max_upload_mb == 50
    assert (settings.embedding_max_retries, settings.ingestion_workers) == (6, 1)
    assert settings.log_level == "INFO"


def test_missing_required_database_url_fails(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.delenv("DATABASE_URL")
    with pytest.raises(ValidationError, match="database_url"):
        make()


def test_missing_jwt_secret_fails(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.delenv("JWT_SECRET")
    with pytest.raises(ValidationError, match="jwt_secret"):
        make()


def test_short_jwt_secret_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("JWT_SECRET", "too-short")
    with pytest.raises(ValidationError, match="at least 32 characters"):
        make()


def test_empty_values_are_treated_as_unset(clean_env: pytest.MonkeyPatch) -> None:
    """`.env.example` ships blank entries such as REWRITE_MODEL= and must not break parsing."""
    clean_env.setenv("EMBEDDING_DIM", "")
    clean_env.setenv("REWRITE_MODEL", "")
    clean_env.setenv("GOOGLE_API_KEY", "")
    settings = make()
    assert settings.embedding_dim == 768
    assert settings.rewrite_model is None
    assert settings.google_api_key.get_secret_value() == ""


def test_rewrite_model_falls_back_to_llm_model(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_MODEL", "gemini-test-model")
    assert make().rewrite_model_id == "gemini-test-model"
    clean_env.setenv("REWRITE_MODEL", "gemini-small")
    assert make().rewrite_model_id == "gemini-small"


def test_overlap_must_be_smaller_than_target(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("CHUNK_TARGET_CHARS", "300")
    clean_env.setenv("CHUNK_OVERLAP_CHARS", "300")
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_CHARS"):
        make()


@pytest.mark.parametrize(
    ("target", "overlap", "ok"),
    [("1200", "200", True), ("1400", "200", False), ("1500", "0", True), ("1600", "0", False)],
)
def test_chunk_size_plus_overlap_must_stay_under_the_embedding_safe_ceiling(
    clean_env: pytest.MonkeyPatch, target: str, overlap: str, ok: bool
) -> None:
    clean_env.setenv("CHUNK_TARGET_CHARS", target)
    clean_env.setenv("CHUNK_OVERLAP_CHARS", overlap)
    if ok:
        make()
    else:
        with pytest.raises(ValidationError, match="must not exceed 1500"):
            make()


@pytest.mark.parametrize("dim", ["64", "4096"])
def test_embedding_dim_out_of_range_rejected(clean_env: pytest.MonkeyPatch, dim: str) -> None:
    clean_env.setenv("EMBEDDING_DIM", dim)
    with pytest.raises(ValidationError, match="embedding_dim"):
        make()


def test_log_level_normalised_and_validated(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LOG_LEVEL", "debug")
    assert make().log_level == "DEBUG"
    clean_env.setenv("LOG_LEVEL", "chatty")
    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        make()


def test_collection_name_encodes_model_and_dimension(clean_env: pytest.MonkeyPatch) -> None:
    assert make().collection_name == "chunks__gemini-embedding-001__768"
    clean_env.setenv("EMBEDDING_DIM", "1536")
    clean_env.setenv("CHROMA_COLLECTION", "kb")
    assert make().collection_name == "kb__gemini-embedding-001__1536"


def test_secrets_are_masked_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("GOOGLE_API_KEY", "AIza-super-secret-key")
    text = repr(make())
    assert SECRET not in text
    assert "AIza-super-secret-key" not in text


def test_max_upload_bytes(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MAX_UPLOAD_MB", "2")
    assert make().max_upload_bytes == 2 * 1024 * 1024
