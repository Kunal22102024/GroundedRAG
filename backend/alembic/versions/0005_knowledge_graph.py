"""knowledge graph: entities, their mentions in chunks, and relations between them

Adds ``kg_entities`` (a person, policy, course... found in one document), ``kg_mentions`` (the
chunks an entity appears in) and ``kg_relations`` (a stated relationship between two entities,
with its evidence chunk), plus the ``graph`` kind of ``document_insights`` (build status) and of
``query_events`` (the cost of building one). Everything cascades from the document and its chunks.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_INSIGHT_KINDS = (
    "'executive', 'chapters', 'faq', 'glossary', 'key_changes', 'extraction', 'questions'"
)
NEW_INSIGHT_KINDS = OLD_INSIGHT_KINDS + ", 'graph'"
OLD_EVENT_KINDS = "'chat', 'summary', 'extract', 'compare', 'suggest'"
NEW_EVENT_KINDS = OLD_EVENT_KINDS + ", 'graph'"
ENTITY_TYPES = (
    "'person', 'organization', 'role', 'concept', 'course', 'policy', 'process', 'product', "
    "'location', 'event', 'other'"
)


def _replace_kind_check(table: str, kinds: str) -> None:
    # op.f(): the name is final; the naming convention must not prefix it a second time.
    name = op.f(f"ck_{table}_kind_valid")
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, f"kind IN ({kinds})")


def upgrade() -> None:
    op.create_table(
        "kg_entities",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("type", sa.String(length=16), server_default="other", nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(f"type IN ({ENTITY_TYPES})", name=op.f("ck_kg_entities_type_valid")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_kg_entities_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kg_entities")),
        sa.UniqueConstraint("document_id", "key", name=op.f("uq_kg_entities_document_id")),
    )
    op.create_index(op.f("ix_kg_entities_document_id"), "kg_entities", ["document_id"])
    op.create_index(op.f("ix_kg_entities_key"), "kg_entities", ["key"])

    op.create_table(
        "kg_mentions",
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["kg_entities.id"],
            name=op.f("fk_kg_mentions_entity_id_kg_entities"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_kg_mentions_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("entity_id", "chunk_id", name=op.f("pk_kg_mentions")),
    )
    op.create_index(op.f("ix_kg_mentions_chunk_id"), "kg_mentions", ["chunk_id"])

    op.create_table(
        "kg_relations",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("target_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("quote", sa.Text(), server_default="", nullable=False),
        sa.CheckConstraint("source_id <> target_id", name=op.f("ck_kg_relations_not_self")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_kg_relations_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["kg_entities.id"],
            name=op.f("fk_kg_relations_source_id_kg_entities"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["kg_entities.id"],
            name=op.f("fk_kg_relations_target_id_kg_entities"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_kg_relations_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kg_relations")),
        sa.UniqueConstraint(
            "source_id", "target_id", "label", "chunk_id", name=op.f("uq_kg_relations_source_id")
        ),
    )
    op.create_index(op.f("ix_kg_relations_document_id"), "kg_relations", ["document_id"])
    op.create_index(op.f("ix_kg_relations_source_id"), "kg_relations", ["source_id"])
    op.create_index(op.f("ix_kg_relations_target_id"), "kg_relations", ["target_id"])
    op.create_index(op.f("ix_kg_relations_chunk_id"), "kg_relations", ["chunk_id"])

    _replace_kind_check("document_insights", NEW_INSIGHT_KINDS)
    _replace_kind_check("query_events", NEW_EVENT_KINDS)


def downgrade() -> None:
    op.execute("DELETE FROM document_insights WHERE kind = 'graph'")
    op.execute("DELETE FROM query_events WHERE kind = 'graph'")
    _replace_kind_check("query_events", OLD_EVENT_KINDS)
    _replace_kind_check("document_insights", OLD_INSIGHT_KINDS)
    op.drop_table("kg_relations")
    op.drop_table("kg_mentions")
    op.drop_table("kg_entities")
