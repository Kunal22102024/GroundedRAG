"""Document workflows: validated uploads, role-aware listing, file access and deletion.

Security rules applied here (CLAUDE.md rules 23 and 24):

* Files are stored under generated names (``<uuid>.<validated extension>``); a
  user-supplied name is only ever shown, never used as a path.
* Extension, declared content type, magic bytes and size are all checked.
* Only admins may assign ``Restricted``. Documents above a user's access level do not
  exist as far as that user can tell (404, never 403).
* Deleting a document removes vectors, rows and file, and the document can never be
  left half-searchable.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import (
    ALLOWED_EXTENSIONS,
    AccessLevel,
    DocumentStatus,
    Role,
    allowed_access_levels,
)
from app.core import bm25, metadata, vectorstore
from app.db.models import Document, Message, User
from app.services import ingestion

logger = logging.getLogger(__name__)

_COPY_CHUNK = 1024 * 1024
_MAX_FIELD_CHARS = 100
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DENIED_CONTENT_PREFIXES = ("image/", "video/", "audio/")
_DELETE_FAILED_MESSAGE = "Deletion did not finish. Delete the document again to retry."


class DocumentError(Exception):
    """Base class for document workflow errors."""


class InvalidMetadata(DocumentError):
    """An upload field (department, type, date, access level) is not acceptable."""


class UploadForbidden(DocumentError):
    """The user may not upload with these settings (e.g. ``Restricted`` as a non-admin)."""


class DocumentNotFound(DocumentError):
    """No such document, or one the user may not see."""


class DocumentForbidden(DocumentError):
    """The user can see the document but may not change it."""


class DocumentBusy(DocumentError):
    """The document is being indexed right now."""


class DocumentDeleteFailed(DocumentError):
    """A deletion step failed; the document is parked in a retryable state."""


class TooManyFiles(DocumentError):
    """One upload request carried more files than ``MAX_FILES_PER_REQUEST``."""


class UploadRejected(Exception):
    """One file in an upload was refused (internal; reported per file)."""


@dataclass
class IncomingFile:
    """An uploaded file, independent of the web framework."""

    filename: str | None
    content_type: str | None
    stream: BinaryIO
    size: int | None = None


@dataclass(frozen=True)
class RejectedFile:
    """A file that was not accepted, and why."""

    filename: str
    reason: str


@dataclass
class UploadResult:
    """Outcome of a multi-file upload."""

    accepted: list[Document] = field(default_factory=list)
    rejected: list[RejectedFile] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentFilters:
    """Optional list filters."""

    department: str | None = None
    document_type: str | None = None
    status: str | None = None
    access_level: str | None = None
    query: str | None = None


@dataclass(frozen=True)
class LibrarySummary:
    """Totals across every document the user may see (not affected by list filters)."""

    total_files: int
    total_chunks: int
    by_status: dict[str, int]


@dataclass(frozen=True)
class DocumentPage:
    """A page of documents plus library totals."""

    items: list[Document]
    total: int
    summary: LibrarySummary


# --- Upload ---------------------------------------------------------------------------------


def _clean_field(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    value = _CONTROL.sub("", value).strip()
    if not value:
        return None
    if len(value) > _MAX_FIELD_CHARS:
        raise InvalidMetadata(f"{label} must be at most {_MAX_FIELD_CHARS} characters.")
    return value


def parse_explicit_metadata(
    department: str | None,
    document_type: str | None,
    creation_date: str | None,
    access_level: str | None,
) -> metadata.ExplicitMetadata:
    """Validate the optional upload fields."""
    parsed_date: date | None = None
    if creation_date and creation_date.strip():
        try:
            parsed_date = date.fromisoformat(creation_date.strip())
        except ValueError:
            raise InvalidMetadata("creation_date must look like 2026-01-15.") from None
        # An implausible date used to be dropped silently and replaced by another source's date.
        if not metadata.is_plausible_date(parsed_date, date.today()):
            raise InvalidMetadata(
                "creation_date must be in 1990 or later and at most a year from today."
            )
    level: str | None = None
    if access_level and access_level.strip():
        level = metadata.canonical_access_level(access_level)
        if level is None:
            raise InvalidMetadata("access_level must be Public, Internal or Restricted.")
    dept = _clean_field(department, "department")
    doc_type = _clean_field(document_type, "document_type")
    return metadata.ExplicitMetadata(
        department=metadata.canonical_department(dept) if dept else None,
        document_type=metadata.canonical_document_type(doc_type) if doc_type else None,
        creation_date=parsed_date,
        access_level=level,
    )


def clean_filename(raw: str | None) -> str:
    """The base name of an uploaded file, safe to display: no folders, no control characters."""
    name = unicodedata.normalize("NFC", raw or "").replace("\\", "/").split("/")[-1]
    name = _CONTROL.sub("", name).strip()
    if len(name) > 255:
        stem, ext = os.path.splitext(name)
        name = stem[: 255 - len(ext)] + ext
    return name


def _verify_content(path: Path, extension: str) -> None:
    """Check the bytes really look like the claimed format."""
    with path.open("rb") as handle:
        head = handle.read(8192)
    if extension == ".pdf":
        if b"%PDF-" not in head[:1024]:
            raise UploadRejected("The file is not a valid PDF.")
    elif extension == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                    raise UploadRejected("The file is not a valid DOCX document.")
                if archive.getinfo("word/document.xml").file_size > 64 * 1024 * 1024:
                    raise UploadRejected("The DOCX document is too large when unpacked.")
        except zipfile.BadZipFile:
            raise UploadRejected("The file is not a valid DOCX document.") from None
    elif b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UploadRejected("The file does not look like a text document.")


def _store_file(file: IncomingFile, extension: str, max_bytes: int) -> tuple[str, int]:
    """Copy the upload to a generated name under the upload directory."""
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{extension}"
    final = upload_dir / stored_name
    partial = upload_dir / f"{stored_name}.part"
    written = 0
    try:
        file.stream.seek(0)
        with partial.open("wb") as out:
            while chunk := file.stream.read(_COPY_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadRejected(
                        f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit."
                    )
                out.write(chunk)
        if written == 0:
            raise UploadRejected("The file is empty.")
        _verify_content(partial, extension)
        partial.replace(final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return stored_name, written


def _check_declared(file: IncomingFile, extension: str, max_bytes: int) -> None:
    if file.size is not None and file.size > max_bytes:
        raise UploadRejected(f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    declared = (file.content_type or "").split(";")[0].strip().lower()
    if declared.startswith(_DENIED_CONTENT_PREFIXES):
        raise UploadRejected(
            f"The content type '{declared[:60]}' does not match a {extension} file."
        )


def create_documents(
    db: Session,
    user: User,
    files: list[IncomingFile],
    explicit: metadata.ExplicitMetadata,
    submit: Callable[[uuid.UUID], None] | None = None,
) -> UploadResult:
    """Validate and store uploads, create ``queued`` rows, and schedule ingestion.

    Files are accepted or rejected one by one; a rejected file never blocks the others.
    Raises ``UploadForbidden`` for a ``Restricted`` upload by a non-admin (the whole
    request is refused: it is a permission problem, not a file problem).
    """
    submit = submit or ingestion.submit  # looked up per call so it can be replaced
    if explicit.access_level == AccessLevel.RESTRICTED.value and user.role != Role.ADMIN:
        raise UploadForbidden("Only administrators may upload Restricted documents.")
    limit = get_settings().max_files_per_request
    if len(files) > limit:
        raise TooManyFiles(f"At most {limit} files can be uploaded at once.")

    max_bytes = get_settings().max_upload_bytes
    result = UploadResult()
    stored: list[Path] = []
    documents: list[Document] = []

    try:
        for file in files:
            display = clean_filename(file.filename)
            shown = display[:120] or "(unnamed file)"
            try:
                if not display:
                    raise UploadRejected("The file has no name.")
                extension = os.path.splitext(display)[1].lower()
                if extension not in ALLOWED_EXTENSIONS:
                    allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
                    raise UploadRejected(
                        f"Unsupported file type '{extension[:12]}'. Allowed: {allowed}."
                    )
                _check_declared(file, extension, max_bytes)
                stored_name, size = _store_file(file, extension, max_bytes)
            except UploadRejected as exc:
                result.rejected.append(RejectedFile(shown, str(exc)))
                continue

            stored.append(Path(get_settings().upload_dir) / stored_name)
            folder_department = metadata.department_from_name(file.filename or "")
            documents.append(
                Document(
                    owner_id=user.id,
                    name=display,
                    stored_path=stored_name,
                    mime_type=ALLOWED_EXTENSIONS[extension],
                    size_bytes=size,
                    status=DocumentStatus.QUEUED.value,
                    # Upload values win; a folder or file-name department is a convention-based
                    # fallback.
                    department=explicit.department or folder_department,
                    document_type=explicit.document_type or "",
                    creation_date=explicit.creation_date,
                    access_level=explicit.access_level or AccessLevel.INTERNAL.value,
                )
            )

        if documents:
            db.add_all(documents)
            db.commit()
    except BaseException:
        # A full disk on the third file, or a database error: nothing may be left on disk that
        # no row points at, however the request ended.
        db.rollback()
        for path in stored:
            path.unlink(missing_ok=True)
        raise

    if documents:
        for document in documents:
            db.refresh(document)
            submit(document.id)
        result.accepted = documents
    return result


# --- Read ----------------------------------------------------------------------------------


def _visible(user: User) -> ColumnElement[bool]:
    return Document.access_level.in_([level.value for level in allowed_access_levels(user.role)])


def get_document(db: Session, user: User, document_id: uuid.UUID) -> Document:
    """A document the user may see, else ``DocumentNotFound`` (never reveals Restricted ones)."""
    document = db.get(Document, document_id)
    if document is None or document.access_level not in {
        level.value for level in allowed_access_levels(user.role)
    }:
        raise DocumentNotFound
    return document


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def list_documents(
    db: Session, user: User, filters: DocumentFilters, limit: int, offset: int
) -> DocumentPage:
    """Documents the user may see, newest first, plus library totals."""
    visible = _visible(user)
    conditions = [visible]
    if filters.department:
        conditions.append(func.lower(Document.department) == filters.department.lower())
    if filters.document_type:
        conditions.append(func.lower(Document.document_type) == filters.document_type.lower())
    if filters.status:
        conditions.append(Document.status == filters.status)
    if filters.access_level:
        conditions.append(Document.access_level == filters.access_level)
    if filters.query:
        conditions.append(Document.name.ilike(_like(filters.query), escape="\\"))

    total = db.scalar(select(func.count()).select_from(Document).where(*conditions)) or 0
    items = list(
        db.scalars(
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
    )

    by_status = {
        status: int(count)
        for status, count in db.execute(
            select(Document.status, func.count()).where(visible).group_by(Document.status)
        )
    }
    chunks = db.scalar(select(func.coalesce(func.sum(Document.chunk_count), 0)).where(visible)) or 0
    summary = LibrarySummary(sum(by_status.values()), int(chunks), by_status)
    return DocumentPage(items, int(total), summary)


def file_path(document: Document) -> Path:
    """Where the original file lives, guaranteed to be inside the upload directory."""
    root = Path(get_settings().upload_dir).resolve()
    path = (root / document.stored_path).resolve()
    if not path.is_relative_to(root):
        raise DocumentNotFound
    return path


# --- Delete ----------------------------------------------------------------------------------


def delete_document(db: Session, user: User, document_id: uuid.UUID) -> None:
    """Delete a document everywhere: BM25, vectors, rows, file.

    Allowed for admins and the uploader. The document is first parked as ``failed`` so it is
    excluded from retrieval immediately; if a later step fails it stays that way and the
    delete can simply be repeated. A document being indexed right now cannot be deleted
    (``DocumentBusy``).
    """
    document = get_document(db, user, document_id)
    if user.role != Role.ADMIN and document.owner_id != user.id:
        raise DocumentForbidden

    lock = ingestion.document_lock(document.id)
    if not lock.acquire(blocking=False):
        raise DocumentBusy
    try:
        _delete_locked(db, document)
    finally:
        lock.release()


def _scrub_citations(db: Session, document_id: uuid.UUID) -> None:
    """Blank the passage text copied into saved answers that cite this document (rule 24).

    A chat citation stores the cited chunk's text so a conversation can be reopened later; without
    this the content of a deleted document would stay readable in old conversations. Safe to repeat.
    """
    wanted = str(document_id)
    rows = db.execute(
        select(Message.id, Message.citations).where(
            Message.citations.contains([{"document_id": wanted}])
        )
    ).all()
    for message_id, citations in rows:
        scrubbed = [
            {**citation, "text": ""} if citation.get("document_id") == wanted else citation
            for citation in citations or []
        ]
        db.execute(update(Message).where(Message.id == message_id).values(citations=scrubbed))


def _delete_locked(db: Session, document: Document) -> None:
    document_id, stored = document.id, document.stored_path
    # 1. Park it: from here on it can no longer be returned by any search path.
    db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(status=DocumentStatus.FAILED.value, error_message=_DELETE_FAILED_MESSAGE)
    )
    db.commit()

    try:
        # 2-4. Derived indexes, the passages copied into saved conversations, then the
        # source-of-truth rows (chunks cascade).
        bm25.get_index().remove_document(str(document_id))
        vectorstore.delete_document(str(document_id))
        _scrub_citations(db, document_id)
        db.execute(delete(Document).where(Document.id == document_id))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(
            "document deletion did not finish",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
        raise DocumentDeleteFailed(_DELETE_FAILED_MESSAGE) from exc

    # 5. The original file. A leftover file is wasted space, not a leak: nothing points at it.
    try:
        (Path(get_settings().upload_dir) / stored).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "could not remove file",
            extra={"document_id": str(document_id), "error": type(exc).__name__},
        )
