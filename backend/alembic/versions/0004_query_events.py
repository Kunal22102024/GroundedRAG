"""query events for the evaluation dashboard, and question suggestions

Adds ``query_events`` (one row per chat turn or analysis run, failures included),
``messages.suggestions`` (questions offered after an answer) and the ``questions`` kind of
``document_insights`` (suggested questions a document answers).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_KINDS = "'executive', 'chapters', 'faq', 'glossary', 'key_changes', 'extraction'"
NEW_KINDS = OLD_KINDS + ", 'questions'"


def _flag(name: str) -> sa.Column:
    return sa.Column(name, sa.Boolean(), server_default=sa.false(), nullable=False)


def _count(name: str) -> sa.Column:
    return sa.Column(name, sa.Integer(), server_default="0", nullable=False)


def upgrade() -> None:
    op.create_table(
        "query_events",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), server_default="typed", nullable=False),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("guard_reason", sa.String(length=40), nullable=True),
        sa.Column("best_similarity", sa.Float(), nullable=True),
        sa.Column("mean_similarity", sa.Float(), nullable=True),
        _count("retrieved"),
        _count("cited"),
        _count("cited_documents"),
        sa.Column("cited_document_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        _count("citations_removed"),
        _flag("uncited"),
        _flag("retried"),
        _flag("partial"),
        _flag("rewritten"),
        _count("conflicts"),
        _count("suggestions"),
        _count("input_tokens"),
        _count("output_tokens"),
        _count("total_tokens"),
        sa.Column("model", sa.String(length=100), server_default="", nullable=False),
        sa.Column("total_ms", sa.Float(), nullable=True),
        sa.Column("timings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("document_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("message_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('chat', 'summary', 'extract', 'compare', 'suggest')",
            name=op.f("ck_query_events_kind_valid"),
        ),
        sa.CheckConstraint(
            "outcome IN ('answered', 'refused', 'failed')",
            name=op.f("ck_query_events_outcome_valid"),
        ),
        sa.CheckConstraint(
            "source IN ('typed', 'suggestion', 'followup')",
            name=op.f("ck_query_events_source_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_query_events_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_query_events")),
    )
    op.create_index(op.f("ix_query_events_created_at"), "query_events", ["created_at"])
    op.create_index(op.f("ix_query_events_user_id"), "query_events", ["user_id"])

    op.add_column("messages", sa.Column("suggestions", postgresql.JSONB(astext_type=sa.Text())))

    _replace_kind_check(NEW_KINDS)


def _replace_kind_check(kinds: str) -> None:
    # op.f(): the name is final; the naming convention must not prefix it a second time.
    name = op.f("ck_document_insights_kind_valid")
    op.drop_constraint(name, "document_insights", type_="check")
    op.create_check_constraint(name, "document_insights", f"kind IN ({kinds})")


def downgrade() -> None:
    op.execute("DELETE FROM document_insights WHERE kind = 'questions'")
    _replace_kind_check(OLD_KINDS)
    op.drop_column("messages", "suggestions")
    op.drop_index(op.f("ix_query_events_user_id"), table_name="query_events")
    op.drop_index(op.f("ix_query_events_created_at"), table_name="query_events")
    op.drop_table("query_events")
