"""documents.creation_date is unknown until indexing resolves it

The PRD resolves department, type and creation date during indexing (explicit
values supplied at upload win). Until then the date is genuinely unknown, so the
column must allow NULL instead of holding a made-up placeholder.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("documents", "creation_date", existing_type=sa.Date(), nullable=True)


def downgrade() -> None:
    # Rows indexed before the date was resolved fall back to their upload date.
    op.execute("UPDATE documents SET creation_date = created_at::date WHERE creation_date IS NULL")
    op.alter_column("documents", "creation_date", existing_type=sa.Date(), nullable=False)
