"""Metadata resolution: upload values > file name/folder > file properties > upload date."""

from __future__ import annotations

from datetime import date

import pytest

from app.core import metadata
from app.core.metadata import ExplicitMetadata
from app.core.parsers import DocumentProperties

TODAY = date(2026, 10, 7)
NOTHING = ExplicitMetadata()
NO_PROPS = DocumentProperties()


def resolve(
    filename: str, explicit: ExplicitMetadata = NOTHING, props: DocumentProperties = NO_PROPS
):
    return metadata.resolve(explicit, filename, props, TODAY)


# --- Precedence ------------------------------------------------------------------------------


def test_explicit_upload_values_beat_everything() -> None:
    result = resolve(
        "HR_Leave_Policy_2026-01-15.pdf",
        ExplicitMetadata("Legal", "Contract", date(2020, 5, 5), "Restricted"),
        DocumentProperties(created=date(2019, 1, 1), department="IT", document_type="Manual"),
    )
    assert (result.department, result.document_type) == ("Legal", "Contract")
    assert result.creation_date == date(2020, 5, 5)
    assert result.access_level == "Restricted"
    assert set(result.sources.values()) >= {"upload"}


def test_filename_beats_file_properties() -> None:
    result = resolve(
        "Engineering_Design_Spec_2025-03-02.pdf",
        props=DocumentProperties(created=date(2012, 1, 1), department="HR", document_type="Policy"),
    )
    assert result.department == "Engineering"
    assert result.document_type == "Technical Spec"
    assert result.creation_date == date(2025, 3, 2)
    assert (result.sources["department"], result.sources["creation_date"]) == (
        "filename",
        "filename",
    )


def test_file_properties_fill_what_the_name_does_not_say() -> None:
    result = resolve(
        "notes.txt",
        props=DocumentProperties(
            created=date(2024, 7, 9), department="Operations", document_type="Policy"
        ),
    )
    assert (result.department, result.document_type, result.creation_date) == (
        "Operations",
        "Policy",
        date(2024, 7, 9),
    )
    assert result.sources["creation_date"] == "file"


def test_title_can_suggest_the_type_when_the_name_cannot() -> None:
    assert (
        resolve("369.pdf", props=DocumentProperties(title="Acceptable Use Policy")).document_type
        == "Policy"
    )


def test_upload_date_is_the_last_resort_and_other_fields_stay_empty() -> None:
    result = resolve("scan0042.pdf")
    assert result.creation_date == TODAY and result.sources["creation_date"] == "upload_date"
    assert (result.department, result.document_type) == ("", "")
    assert result.access_level == "Internal" and result.sources["access_level"] == "default"


def test_creation_date_int_is_yyyymmdd() -> None:
    assert resolve("Policy_2026-01-15.pdf").creation_date_int == 20260115
    assert metadata.date_to_int(date(1999, 12, 31)) == 19991231


# --- File names ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("IT/369.pdf", "IT"),
        ("it/369.pdf", "IT"),
        ("Legal\\Contracts\\nda.docx", "Legal"),
        ("archive/HR/leave.pdf", "HR"),
        ("HR_Leave_Policy.pdf", "HR"),
        ("engineering-handbook.md", "Engineering"),
        ("IT_Infrastructure_Policy_2026.pdf", "IT"),
        ("it works fine.txt", ""),  # the English word "it" is not the IT department
        ("Tesla Car Manuals/Owners_Manual.pdf", ""),
        ("../../etc/passwd.txt", ""),
    ],
)
def test_department_from_name(filename: str, expected: str) -> None:
    assert metadata.department_from_name(filename) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Owners_Manual", "Manual"),
        ("Valve_Handbook_LowRes", "Manual"),
        ("RFC 9110_ HTTP Semantics _ RFC Editor", "Technical Spec"),
        ("API_Spec_v2", "Technical Spec"),
        ("API_Policy_2026", "Policy"),  # a policy about APIs is still a policy
        ("Master Services Agreement", "Contract"),
        ("SANS_Password_Construction_Standard", "Policy"),
        ("Email Management Policy", "Policy"),
        ("notes", ""),
    ],
)
def test_type_from_text(text: str, expected: str) -> None:
    assert metadata.type_from_text(text) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Policy_2026-01-15.pdf", date(2026, 1, 15)),
        ("Policy_20260115.pdf", date(2026, 1, 15)),
        ("Policy 2026.01.15.pdf", date(2026, 1, 15)),
        ("SANS_Password_Construction_Standard_April2025.pdf", date(2025, 4, 1)),
        ("sans_Email_Management_Policy_feb2026.pdf", date(2026, 2, 1)),
        ("report_Sept_2024.pdf", date(2024, 9, 1)),
        ("IT_Infrastructure_Policy_2026.pdf", date(2026, 1, 1)),
        ("RFC 9110_ HTTP Semantics.pdf", None),  # 9110 is not a year
        ("369.pdf", None),
        ("build 3000-01-01.pdf", None),  # far future
        ("scan_1975.pdf", None),  # before 1990
        ("order 2026-13-45.pdf", date(2026, 1, 1)),  # impossible month/day: only the year counts
    ],
)
def test_date_from_name(filename: str, expected: date | None) -> None:
    assert metadata.date_from_name(filename, TODAY) == expected


@pytest.mark.parametrize(
    ("filename", "file_date", "expected", "source"),
    [
        # the name says only the month/year; a consistent file date supplies the day
        (
            "SANS_Password_Standard_April2025.pdf",
            date(2025, 4, 7),
            date(2025, 4, 7),
            "filename+file",
        ),
        ("IT_Infrastructure_Policy_2026.pdf", date(2026, 3, 9), date(2026, 3, 9), "filename+file"),
        # an inconsistent file date (a re-save years later) must not override the name
        ("sans_Email_Policy_feb2026.pdf", date(2025, 5, 5), date(2026, 2, 1), "filename"),
        ("IT_Infrastructure_Policy_2026.pdf", date(2025, 12, 31), date(2026, 1, 1), "filename"),
        ("sans_Email_Policy_feb2026.pdf", date(2026, 3, 1), date(2026, 2, 1), "filename"),
        # a full date in the name is exact: nothing to refine
        ("Policy_2026-01-15.pdf", date(2026, 1, 20), date(2026, 1, 15), "filename"),
        ("sans_Email_Policy_feb2026.pdf", None, date(2026, 2, 1), "filename"),
    ],
)
def test_a_consistent_file_date_refines_a_month_or_year_only_name(
    filename: str, file_date: date | None, expected: date, source: str
) -> None:
    result = resolve(filename, props=DocumentProperties(created=file_date))
    assert (result.creation_date, result.sources["creation_date"]) == (expected, source)


def test_implausible_property_dates_are_ignored() -> None:
    assert (
        resolve("a.pdf", props=DocumentProperties(created=date(1970, 1, 1))).creation_date == TODAY
    )
    assert (
        resolve("a.pdf", props=DocumentProperties(created=date(2099, 1, 1))).creation_date == TODAY
    )
    assert resolve("a.pdf", ExplicitMetadata(creation_date=date(1970, 1, 1))).creation_date == TODAY


# --- Canonical spellings ----------------------------------------------------------------------------


def test_known_values_are_spelled_canonically_unknown_ones_are_kept() -> None:
    assert metadata.canonical_department("  hr ") == "HR"
    assert metadata.canonical_department("Finance") == "Finance"
    assert metadata.canonical_document_type("technical spec") == "Technical Spec"
    assert metadata.canonical_access_level("restricted") == "Restricted"
    assert metadata.canonical_access_level("secret") is None


def test_access_level_defaults_to_internal_and_ignores_invalid_values() -> None:
    assert resolve("a.pdf").access_level == "Internal"
    assert resolve("a.pdf", ExplicitMetadata(access_level="bogus")).access_level == "Internal"
    assert resolve("a.pdf", ExplicitMetadata(access_level="public")).access_level == "Public"
