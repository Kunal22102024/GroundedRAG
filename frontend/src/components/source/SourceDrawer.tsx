import { useEffect, useRef, useState } from "react";
import { apiBlob } from "../../api/client";
import { downloadOriginal, isPdf, pageLabel } from "../../lib/files";
import type { Citation } from "../../lib/types";
import { btnSecondary, iconBtn } from "../../lib/ui";
import { BookOpenIcon, DownloadIcon, FileTextIcon, XIcon } from "../ui/icons";
import { FileTypeBadge, Skeleton, Spinner } from "../ui/primitives";

/** The original PDF at the cited page (fetched with the token, shown from a blob URL). */
function DocumentViewer({ citation }: { citation: Citation }) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pdf = isPdf(citation.document_name);

  useEffect(() => {
    if (!pdf) return;
    let objectUrl: string | null = null;
    let cancelled = false;
    setUrl(null);
    setError(null);
    apiBlob(`/documents/${citation.document_id}/file`)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not load the document"));
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [citation.document_id, pdf]);

  if (!pdf) {
    return (
      <div className="space-y-2 rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">
        <p>Only PDFs are shown here. Download the original to open it.</p>
        <button
          onClick={() => {
            setError(null);
            downloadOriginal(citation.document_id, citation.document_name).catch((e: unknown) =>
              setError(e instanceof Error ? e.message : "Could not download the file"),
            );
          }}
          className={btnSecondary}
        >
          <DownloadIcon size={15} /> Download the original file
        </button>
        {error && (
          <p role="alert" className="text-sm text-rose-600">
            {error}
          </p>
        )}
      </div>
    );
  }
  if (error)
    return (
      <p role="alert" className="rounded-xl bg-rose-50 p-3 text-sm text-rose-600">
        {error}
      </p>
    );
  if (!url)
    return (
      <div className="relative h-[28rem] overflow-hidden rounded-xl border border-slate-200">
        <Skeleton className="absolute inset-0 rounded-none" />
        <p className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-slate-500">
          <Spinner className="text-indigo-500" /> Loading the document…
        </p>
      </div>
    );
  // `key`: a viewer only follows the #page fragment when the document is first loaded, so a
  // citation to another page of the same file needs a fresh frame.
  return (
    <iframe
      key={citation.chunk_id}
      title="Original document"
      src={`${url}#page=${citation.page_number}`}
      className="h-[32rem] w-full animate-fade-in rounded-xl border border-slate-200 shadow-sm"
    />
  );
}

export default function SourceDrawer({ citation, onClose }: { citation: Citation | null; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  // The parent passes a new function on every render; reading it through a ref keeps the effect
  // below from re-running (and moving focus) each time a streamed token re-renders the page.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!citation) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus(); // back to the citation that opened it
    };
  }, [citation]);

  if (!citation) return null;
  return (
    <>
      <div className="absolute inset-0 z-10 animate-fade-in bg-slate-900/10 lg:hidden" onClick={onClose} />
      <aside
        role="dialog"
        aria-modal="false"
        aria-label={`Source: ${citation.document_name}`}
        className="absolute inset-y-0 right-0 z-20 flex w-full max-w-xl animate-slide-in-right flex-col overflow-hidden border-l border-slate-200 bg-white shadow-2xl"
      >
        <div className="flex items-start gap-3 border-b border-slate-100 p-4">
          <FileTypeBadge name={citation.document_name} />
          <div className="min-w-0 flex-1">
            <h2 className="truncate font-semibold text-slate-900" title={citation.document_name}>
              {citation.document_name}
            </h2>
            <p className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
              <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 font-medium text-indigo-700">Page {pageLabel(citation.page_number, citation.page_end)}</span>
              {citation.section_title && <span className="truncate">{citation.section_title}</span>}
            </p>
          </div>
          <button ref={closeButton} onClick={onClose} aria-label="Close" title="Close (Esc)" className={iconBtn}>
            <XIcon size={18} />
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
          <section>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
              <BookOpenIcon size={14} /> Cited passage
            </h3>
            <p className="whitespace-pre-wrap rounded-xl border border-amber-200/70 bg-amber-50/70 p-3.5 text-sm leading-relaxed text-slate-800 shadow-inner shadow-amber-100">
              {citation.text}
            </p>
          </section>
          <section>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
              <FileTextIcon size={14} /> Original document
            </h3>
            <DocumentViewer citation={citation} />
          </section>
        </div>
      </aside>
    </>
  );
}
