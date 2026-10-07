"""Document endpoints: upload, list, detail, original file, delete.

Handlers are plain ``def``: file copying and database access are blocking, and FastAPI
runs such handlers in a worker thread.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.constants import AccessLevel, DocumentStatus
from app.deps import CurrentUser, DbSession
from app.schemas import (
    DocumentListResponse,
    DocumentOut,
    LibrarySummaryOut,
    RejectedFileOut,
    UploadResponse,
)
from app.services import document_service
from app.services.document_service import (
    DocumentBusy,
    DocumentDeleteFailed,
    DocumentFilters,
    DocumentForbidden,
    DocumentNotFound,
    IncomingFile,
    InvalidMetadata,
    UploadForbidden,
)

router = APIRouter(prefix="/documents", tags=["documents"])

_PDF = "application/pdf"


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one or more documents",
    responses={400: {"description": "No file was accepted"}, 403: {"description": "Not allowed"}},
)
def upload_documents(
    user: CurrentUser,
    db: DbSession,
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, MD, HTML or TXT files")],
    department: Annotated[
        str | None, Form(description="e.g. HR, Engineering, Legal, Operations, IT")
    ] = None,
    document_type: Annotated[
        str | None, Form(description="e.g. Policy, Manual, Technical Spec, Contract")
    ] = None,
    creation_date: Annotated[str | None, Form(description="YYYY-MM-DD")] = None,
    access_level: Annotated[
        str | None, Form(description="Public, Internal (default) or Restricted (admins only)")
    ] = None,
) -> UploadResponse:
    """Accept files for indexing.

    Accepted files are returned as ``queued`` and move to ``processing`` and ``indexed``
    (or ``failed``) in the background; poll ``GET /documents``. Files that are refused are
    listed in ``rejected`` without blocking the others. Department, type and creation date
    are worked out during indexing when not given.
    """
    try:
        explicit = document_service.parse_explicit_metadata(
            department, document_type, creation_date, access_level
        )
        result = document_service.create_documents(
            db,
            user,
            [IncomingFile(f.filename, f.content_type, f.file, f.size) for f in files],
            explicit,
        )
    except InvalidMetadata as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from None
    except UploadForbidden as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=str(exc)) from None

    rejected = [RejectedFileOut(filename=r.filename, reason=r.reason) for r in result.rejected]
    if not result.accepted:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "No file was accepted.",
                "rejected": [r.model_dump() for r in rejected],
            },
        )
    return UploadResponse(
        documents=[DocumentOut.model_validate(d) for d in result.accepted], rejected=rejected
    )


@router.get("", response_model=DocumentListResponse, summary="List documents you may see")
def list_documents(
    user: CurrentUser,
    db: DbSession,
    department: str | None = None,
    document_type: str | None = None,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    access_level: AccessLevel | None = None,
    q: Annotated[str | None, Query(description="Part of the file name", max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """Newest first. ``summary`` covers every document you may see, ignoring the filters."""
    page = document_service.list_documents(
        db,
        user,
        DocumentFilters(
            department=department,
            document_type=document_type,
            status=status_filter.value if status_filter else None,
            access_level=access_level.value if access_level else None,
            query=q,
        ),
        limit,
        offset,
    )
    return DocumentListResponse(
        items=[DocumentOut.model_validate(d) for d in page.items],
        total=page.total,
        limit=limit,
        offset=offset,
        summary=LibrarySummaryOut(
            total_files=page.summary.total_files,
            total_chunks=page.summary.total_chunks,
            by_status=page.summary.by_status,
        ),
    )


@router.get("/{document_id}", response_model=DocumentOut, summary="Document detail and status")
def get_document(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> DocumentOut:
    """Detail for one document. Documents above your access level answer 404."""
    try:
        return DocumentOut.model_validate(document_service.get_document(db, user, document_id))
    except DocumentNotFound:
        raise _not_found() from None


@router.get("/{document_id}/file", summary="The original uploaded file")
def get_document_file(
    document_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    download: Annotated[
        bool, Query(description="Force a download instead of inline display")
    ] = False,
) -> FileResponse:
    """Serve the original. Only PDFs are shown inline; everything else is a download.

    HTML and Markdown are never rendered from this origin: that would let an uploaded
    page run script next to the session token.
    """
    try:
        document = document_service.get_document(db, user, document_id)
        path = document_service.file_path(document)
    except DocumentNotFound:
        raise _not_found() from None
    if not path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="The original file is no longer available"
        )
    inline = document.mime_type == _PDF and not download
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"}
    if not inline:
        headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
    return FileResponse(
        path,
        media_type=document.mime_type,
        filename=document.name,
        content_disposition_type="inline" if inline else "attachment",
        headers=headers,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document everywhere",
    responses={409: {"description": "The document is being indexed"}},
)
def delete_document(document_id: uuid.UUID, user: CurrentUser, db: DbSession) -> Response:
    """Remove vectors, rows and the file. Allowed for admins and the uploader."""
    try:
        document_service.delete_document(db, user, document_id)
    except DocumentNotFound:
        raise _not_found() from None
    except DocumentForbidden:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Only an administrator or the uploader can delete this document",
        ) from None
    except DocumentBusy:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="The document is being indexed. Try again when it finishes.",
        ) from None
    except DocumentDeleteFailed as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
