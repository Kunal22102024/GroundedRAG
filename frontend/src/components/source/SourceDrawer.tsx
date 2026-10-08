import { useEffect, useState } from "react";
import { apiBlob } from "../../api/client";
import { downloadOriginal, isPdf, pageLabel } from "../../lib/files";
import type { Citation } from "../../lib/types";

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
      <button onClick={() => void downloadOriginal(citation.document_id, citation.document_name)} className="rounded border px-3 py-1.5 text-sm text-indigo-700 hover:bg-indigo-50">
        Download the original file
      </button>
    );
  }
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!url) return <p className="text-sm text-slate-500">Loading document…</p>;
  return <iframe title="Original document" src={`${url}#page=${citation.page_number}`} className="h-[28rem] w-full rounded border" />;
}

export default function SourceDrawer({ citation, onClose }: { citation: Citation | null; onClose: () => void }) {
  if (!citation) return null;
  return (
    <aside className="absolute inset-y-0 right-0 z-10 flex w-full max-w-xl flex-col gap-3 overflow-y-auto border-l bg-white p-4 shadow-xl">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold">{citation.document_name}</h2>
          <p className="text-sm text-slate-500">
            Page {pageLabel(citation.page_number, citation.page_end)}
            {citation.section_title && ` · ${citation.section_title}`}
          </p>
        </div>
        <button onClick={onClose} aria-label="Close" className="rounded border px-2 py-1 text-sm hover:bg-slate-100">
          ✕
        </button>
      </div>
      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase text-slate-500">Cited passage</h3>
        <p className="whitespace-pre-wrap rounded border-l-4 border-amber-400 bg-amber-50 p-3 text-sm">{citation.text}</p>
      </section>
      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase text-slate-500">Original document</h3>
        <DocumentViewer citation={citation} />
      </section>
    </aside>
  );
}
