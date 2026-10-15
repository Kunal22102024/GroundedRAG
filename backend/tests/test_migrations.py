"""The hand-written migration must build exactly the schema the models describe."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.database import Base, get_engine
from tests.conftest import run_migrations, temporary_database, use_database

EXPECTED_TABLES = {
    "users",
    "documents",
    "chunks",
    "conversations",
    "messages",
    "feedback",
    "document_insights",
    "query_events",
}


@pytest.fixture
def empty_database() -> Iterator[str]:
    """An unmigrated throwaway database, separate from the shared session one."""
    with temporary_database() as url, use_database(url):
        yield url


def user_tables() -> set[str]:
    return set(inspect(get_engine()).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_every_table(empty_database: str) -> None:
    assert user_tables() == set()
    run_migrations("head")
    assert user_tables() == EXPECTED_TABLES


def test_downgrade_removes_everything_and_upgrade_works_again(empty_database: str) -> None:
    run_migrations("head")
    run_migrations("base", downgrade=True)
    assert user_tables() == set()
    run_migrations("head")
    assert user_tables() == EXPECTED_TABLES


def test_models_and_migration_agree(empty_database: str) -> None:
    """Tables, columns, types, nullability, indexes, uniques and foreign keys match."""
    run_migrations("head")
    with get_engine().connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        differences = compare_metadata(context, Base.metadata)
    assert differences == []


def test_check_constraints_agree(empty_database: str) -> None:
    """Alembic's autogenerate cannot compare CHECK constraints, so compare them by name."""
    run_migrations("head")
    expected: set[str] = set()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        expected |= set(re.findall(r"CONSTRAINT (\w+) CHECK", ddl))
    assert expected, "the models should define CHECK constraints"

    # Our tables only: PostgreSQL's own information_schema domains carry CHECKs too.
    with get_engine().connect() as connection:
        actual = set(
            connection.execute(
                text(
                    "SELECT conname FROM pg_constraint WHERE contype = 'c' "
                    "AND conrelid <> 0 AND connamespace = 'public'::regnamespace"
                )
            ).scalars()
        )
    assert actual == expected


def test_foreign_keys_cascade_as_designed(empty_database: str) -> None:
    run_migrations("head")
    with get_engine().connect() as connection:
        rows = connection.execute(
            text(
                "SELECT conrelid::regclass::text AS tbl, confdeltype FROM pg_constraint "
                "WHERE contype = 'f'"
            )
        ).all()
    actions = {(table, kind) for table, kind in rows}
    # 'c' = CASCADE, 'n' = SET NULL
    assert ("chunks", "c") in actions
    assert ("messages", "c") in actions
    assert ("feedback", "c") in actions
    assert ("document_insights", "c") in actions
    assert ("conversations", "c") in actions
    assert ("documents", "n") in actions
    assert ("query_events", "n") in actions  # a measurement outlives its user
    assert all(kind in {"c", "n"} for _, kind in actions), "no unexpected RESTRICT/NO ACTION FKs"
