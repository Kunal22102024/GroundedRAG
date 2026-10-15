"""saved document analyses and the conflicts found in an answer

Adds ``document_insights`` (summaries and extractions, one row per document, kind and parameters)
and ``messages.conflicts`` (where the documents behind an answer disagreed).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_insights",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("params_key", sa.String(length=64), server_default="", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=100), server_default="", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "kind IN ('executive', 'chapters', 'faq', 'glossary', 'key_changes', 'extraction')",
            name=op.f("ck_document_insights_kind_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name=op.f("ck_document_insights_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_insights_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_insights")),
        sa.UniqueConstraint(
            "document_id", "kind", "params_key", name=op.f("uq_document_insights_document_id")
        ),
    )
    op.add_column("messages", sa.Column("conflicts", postgresql.JSONB(astext_type=sa.Text())))


def downgrade() -> None:
    op.drop_column("messages", "conflicts")
    op.drop_table("document_insights")
