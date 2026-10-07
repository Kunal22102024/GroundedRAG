"""Role -> access-level rules (PRD 4.4, TDD section 8).

Retrieval-time enforcement (the filter applied to both search channels) is added
and tested with ``core/filters.py``; this pins the mapping it will be built from.
"""

from __future__ import annotations

import pytest

from app.constants import ACCESS_LEVELS, ROLE_ACCESS_LEVELS, AccessLevel, Role


def test_every_role_has_a_mapping() -> None:
    assert set(ROLE_ACCESS_LEVELS) == set(Role)


def test_admin_reads_everything() -> None:
    assert set(ROLE_ACCESS_LEVELS[Role.ADMIN]) == {"Public", "Internal", "Restricted"}


def test_user_never_reads_restricted() -> None:
    allowed = ROLE_ACCESS_LEVELS[Role.USER]
    assert set(allowed) == {"Public", "Internal"}
    assert AccessLevel.RESTRICTED not in allowed


def test_levels_are_stored_with_the_exact_case_used_by_filters() -> None:
    assert ACCESS_LEVELS == ("Public", "Internal", "Restricted")


@pytest.mark.parametrize("role", list(Role))
def test_mapped_levels_are_valid_levels(role: Role) -> None:
    assert all(level in ACCESS_LEVELS for level in ROLE_ACCESS_LEVELS[role])
