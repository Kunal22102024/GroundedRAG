"""Resolve a document's department, type, creation date and access level.

Order of precedence, per field (PRD 4.1, TDD 3.2):

1. values supplied explicitly at upload,
2. file name and folder conventions (``IT/Policy_2026-01-15.pdf``),
3. the file's own properties (PDF CreationDate, DOCX core properties, Markdown front matter),
4. the upload date (creation date only).

``creation_date_int`` (YYYYMMDD) is derived here because Chroma range filters only
work on numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.constants import DEPARTMENTS, DOCUMENT_TYPES, AccessLevel
from app.core.parsers import DocumentProperties

_MIN_YEAR = 1990
MAX_LABEL_CHARS = 100  # width of the department and document_type columns
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_FULL_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"
)
_MONTH_YEAR = re.compile(
    r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_.\-]*((?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
_YEAR_ONLY = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# Keyword -> canonical document type, checked against file name and title words.
_TYPE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("contract", "agreement", "nda", "msa", "sla"), "Contract"),
    (("spec", "specs", "specification", "rfc"), "Technical Spec"),
    (("manual", "handbook", "guide", "owners"), "Manual"),
    (("policy", "policies", "guideline", "guidelines", "standard", "procedure"), "Policy"),
)


@dataclass(frozen=True)
class ExplicitMetadata:
    """Values the uploader chose. ``None`` means "work it out"."""

    department: str | None = None
    document_type: str | None = None
    creation_date: date | None = None
    access_level: str | None = None


@dataclass(frozen=True)
class ResolvedMetadata:
    """Final metadata plus where each value came from (for logging and debugging)."""

    department: str
    document_type: str
    creation_date: date
    access_level: str
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def creation_date_int(self) -> int:
        """YYYYMMDD as an integer, for range filters."""
        return date_to_int(self.creation_date)


def date_to_int(value: date) -> int:
    """YYYYMMDD as an integer, for Chroma range filters (which only work on numbers)."""
    return value.year * 10000 + value.month * 100 + value.day


def canonical_department(value: str) -> str:
    """Return the known department spelled canonically, or the trimmed input."""
    value = value.strip()
    lookup = {name.casefold(): name for name in DEPARTMENTS}
    return lookup.get(value.casefold(), value)


def canonical_document_type(value: str) -> str:
    """Return the known document type spelled canonically, or the trimmed input."""
    value = value.strip()
    lookup = {name.casefold(): name for name in DOCUMENT_TYPES}
    return lookup.get(value.casefold(), value)


def canonical_access_level(value: str) -> str | None:
    """Match an access level case-insensitively; ``None`` if it is not one."""
    for level in AccessLevel:
        if level.value.casefold() == value.strip().casefold():
            return level.value
    return None


def _plausible(candidate: date | None, today: date) -> date | None:
    """Reject dates that cannot be a document's creation date (ancient or far in the future)."""
    if candidate is None or candidate.year < _MIN_YEAR or candidate > today + timedelta(days=366):
        return None
    return candidate


def is_plausible_date(candidate: date, today: date) -> bool:
    """True if ``candidate`` can be a creation date (not before 1990, not far in the future)."""
    return _plausible(candidate, today) is not None


def _split_name(filename: str) -> tuple[list[str], str]:
    """Folder components and the file stem of an uploaded name (either slash style)."""
    parts = [p for p in re.split(r"[\\/]+", filename.strip()) if p and p not in {".", ".."}]
    stem = parts[-1].rsplit(".", 1)[0] if parts else ""
    return parts[:-1], stem


def department_from_name(filename: str) -> str:
    """A known department named by a folder or by the first word of the file name."""
    folders, stem = _split_name(filename)
    known = {name.casefold(): name for name in DEPARTMENTS}
    for folder in folders:
        if folder.casefold() in known:
            return known[folder.casefold()]
    first = re.split(r"[\s_.\-]+", stem)[0] if stem else ""
    # Short names such as "IT" must be written in capitals to avoid matching the word "it".
    if first.casefold() in known and (len(first) > 2 or first == first.upper()):
        return known[first.casefold()]
    return ""


def type_from_text(text: str) -> str:
    """A canonical document type suggested by words in a file name or title."""
    words = set(re.findall(r"[a-z0-9]+", text.casefold()))
    for keywords, canonical in _TYPE_KEYWORDS:
        if words.intersection(keywords):
            return canonical
    return ""


def dated_name(filename: str, today: date) -> tuple[date, str] | None:
    """A creation date written in the file name, with how precisely the name states it.

    Recognises 2026-01-15 / 20260115 (precision "day"), "April2025" / "feb_2026" ("month",
    reported as the 1st) and a bare year ("year", reported as January 1st). Returns ``None``
    when nothing plausible is found.
    """
    _, stem = _split_name(filename)
    match = _FULL_DATE.search(stem)
    if match:
        try:
            found = _plausible(
                date(int(match.group(1)), int(match.group(2)), int(match.group(3))), today
            )
        except ValueError:
            found = None
        if found:
            return found, "day"
    match = _MONTH_YEAR.search(stem)
    if match:
        found = _plausible(date(int(match.group(2)), _MONTHS[match.group(1).lower()], 1), today)
        if found:
            return found, "month"
    match = _YEAR_ONLY.search(stem)
    if match:
        found = _plausible(date(int(match.group(1)), 1, 1), today)
        if found:
            return found, "year"
    return None


def date_from_name(filename: str, today: date) -> date | None:
    """Just the date part of ``dated_name``."""
    found = dated_name(filename, today)
    return found[0] if found else None


def resolve(
    explicit: ExplicitMetadata,
    filename: str,
    properties: DocumentProperties,
    today: date,
) -> ResolvedMetadata:
    """Apply the precedence rules and return the final metadata."""
    sources: dict[str, str] = {}

    def pick(name: str, *candidates: tuple[str, str]) -> str:
        for source, value in candidates:
            if value:
                sources[name] = source
                return value
        sources[name] = "none"
        return ""

    department = pick(
        "department",
        ("upload", canonical_department(explicit.department or "")),
        ("filename", department_from_name(filename)),
        ("file", canonical_department(properties.department)),
    )
    document_type = pick(
        "document_type",
        ("upload", canonical_document_type(explicit.document_type or "")),
        ("filename", type_from_text(_split_name(filename)[1])),
        (
            "file",
            canonical_document_type(properties.document_type) or type_from_text(properties.title),
        ),
    )

    # Values read from inside the file are not length-checked by any form; the columns hold 100.
    department = department[:MAX_LABEL_CHARS]
    document_type = document_type[:MAX_LABEL_CHARS]

    from_file = _plausible(properties.created, today)
    named = dated_name(filename, today)
    filename_choice: tuple[str, date | None] = ("filename", named[0] if named else None)
    if named and named[1] != "day" and from_file is not None:
        # "April2025" only says the month. If the file's own date agrees, it adds the day.
        precision, stated = named[1], named[0]
        if from_file.year == stated.year and (
            precision == "year" or from_file.month == stated.month
        ):
            filename_choice = ("filename+file", from_file)
    creation = (
        ("upload", _plausible(explicit.creation_date, today)),
        filename_choice,
        ("file", from_file),
    )
    chosen = next(((source, value) for source, value in creation if value), ("upload_date", today))
    sources["creation_date"] = chosen[0]

    access = canonical_access_level(explicit.access_level or "") or AccessLevel.INTERNAL.value
    sources["access_level"] = "upload" if explicit.access_level else "default"
    return ResolvedMetadata(department, document_type, chosen[1], access, sources)
