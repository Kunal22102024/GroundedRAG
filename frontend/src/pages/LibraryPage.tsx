import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type DragEvent, type FormEvent } from "react";
import { api } from "../api/client";
import { useAuth } from "../hooks/useAuth";
import { downloadOriginal, formatBytes } from "../lib/files";
import type { DocStatus, DocumentItem, DocumentList, UploadResult } from "../lib/types";

const DEPARTMENTS = ["HR", "Engineering", "Legal", "Operations", "IT"];
const TYPES = ["Policy", "Manual", "Technical Spec", "Contract"];
const ACCEPT = ".pdf,.docx,.md,.markdown,.html,.htm,.txt";

const BADGE: Record<DocStatus, string> = {
  queued: "bg-slate-100 text-slate-700",
  processing: "bg-amber-100 text-amber-800",
  indexed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function StatusBadge({ doc }: { doc: DocumentItem }) {
  return (
    <span title={doc.error_message ?? undefined} className={`rounded-full px-2 py-0.5 text-xs font-medium ${BADGE[doc.status]}`}>
      {doc.status}
    </span>
  );
}

function UploadPanel({ isAdmin }: { isAdmin: boolean }) {
  const queryClient = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [department, setDepartment] = useState("");
  const [documentType, setDocumentType] = useState("");
  const [date, setDate] = useState("");
  const [access, setAccess] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const upload = useMutation({
    mutationFn: () => {
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      if (department) form.append("department", department);
      if (documentType) form.append("document_type", documentType);
      if (date) form.append("creation_date", date);
      if (access) form.append("access_level", access);
      return api<UploadResult>("/documents", { method: "POST", body: form });
    },
    onSuccess: (result) => {
      setFiles([]);
      if (input.current) input.current.value = "";
      setMessage(result.rejected.length ? `Not accepted: ${result.rejected.map((r) => `${r.filename} (${r.reason})`).join("; ")}` : null);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (error) => setMessage(error instanceof Error ? error.message : "Upload failed"),
  });

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    setFiles(Array.from(event.dataTransfer.files));
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (files.length) upload.mutate();
  };
  const field = "rounded border px-2 py-1.5 text-sm";

  return (
    <form onSubmit={submit} className="space-y-3 rounded-xl border bg-white p-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm ${dragging ? "border-indigo-500 bg-indigo-50" : "border-slate-300 text-slate-500"}`}
      >
        {files.length ? files.map((f) => f.name).join(", ") : "Drop PDF, DOCX, MD, HTML or TXT files here, or click to choose"}
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-slate-500">
          Department
          <select value={department} onChange={(e) => setDepartment(e.target.value)} className={`${field} block`}>
            <option value="">Auto</option>
            {DEPARTMENTS.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Type
          <select value={documentType} onChange={(e) => setDocumentType(e.target.value)} className={`${field} block`}>
            <option value="">Auto</option>
            {TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Creation date
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className={`${field} block`} />
        </label>
        <label className="text-xs text-slate-500">
          Access level
          <select value={access} onChange={(e) => setAccess(e.target.value)} className={`${field} block`}>
            <option value="">Internal</option>
            <option>Public</option>
            {isAdmin && <option>Restricted</option>}
          </select>
        </label>
        <button disabled={!files.length || upload.isPending} className="ml-auto rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
          {upload.isPending ? "Uploading…" : `Upload${files.length ? ` ${files.length} file${files.length > 1 ? "s" : ""}` : ""}`}
        </button>
      </div>
      {message && <p className="rounded bg-amber-50 p-2 text-sm text-amber-800">{message}</p>}
    </form>
  );
}

export default function LibraryPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [toDelete, setToDelete] = useState<DocumentItem | null>(null);
  const [error, setError] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["documents", { q, status }],
    queryFn: () => {
      const params = new URLSearchParams({ limit: "200" });
      if (q) params.set("q", q);
      if (status) params.set("status", status);
      return api<DocumentList>(`/documents?${params}`);
    },
    refetchInterval: (query) => (query.state.data?.items.some((d) => d.status === "queued" || d.status === "processing") ? 2000 : false),
  });

  const remove = useMutation({
    mutationFn: (doc: DocumentItem) => api<void>(`/documents/${doc.id}`, { method: "DELETE" }),
    onSuccess: () => {
      setToDelete(null);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (e) => {
      setToDelete(null);
      setError(e instanceof Error ? e.message : "Delete failed");
    },
  });

  const summary = list.data?.summary;
  return (
    <div className="mx-auto h-full max-w-6xl space-y-4 overflow-y-auto p-4">
      <UploadPanel isAdmin={user?.role === "admin"} />
      <div className="flex flex-wrap items-center gap-3">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by file name" className="rounded border bg-white px-3 py-1.5 text-sm" />
        <select value={status} onChange={(e) => setStatus(e.target.value)} className="rounded border bg-white px-2 py-1.5 text-sm">
          <option value="">All statuses</option>
          {(["queued", "processing", "indexed", "failed"] as DocStatus[]).map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        {summary && (
          <span className="ml-auto text-sm text-slate-500">
            {summary.total_files} files · {summary.total_chunks} chunks
          </span>
        )}
      </div>
      {error && <p className="rounded bg-red-50 p-2 text-sm text-red-700">{error}</p>}
      <div className="overflow-x-auto rounded-xl border bg-white">
        <table className="w-full text-left text-sm">
          <thead className="border-b bg-slate-50 text-xs uppercase text-slate-500">
            <tr>
              {["Name", "Status", "Chunks", "Department", "Type", "Date", "Access", "Size", ""].map((h) => (
                <th key={h} className="px-3 py-2">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {list.data?.items.map((doc) => (
              <tr key={doc.id} className="border-b last:border-0">
                <td className="max-w-xs truncate px-3 py-2 font-medium" title={doc.name}>
                  {doc.name}
                  {doc.status === "failed" && doc.error_message && <div className="truncate text-xs font-normal text-red-600">{doc.error_message}</div>}
                </td>
                <td className="px-3 py-2">
                  <StatusBadge doc={doc} />
                </td>
                <td className="px-3 py-2">{doc.chunk_count}</td>
                <td className="px-3 py-2">{doc.department || "-"}</td>
                <td className="px-3 py-2">{doc.document_type || "-"}</td>
                <td className="px-3 py-2">{doc.creation_date ?? "-"}</td>
                <td className="px-3 py-2">{doc.access_level}</td>
                <td className="px-3 py-2">{formatBytes(doc.size_bytes)}</td>
                <td className="space-x-2 whitespace-nowrap px-3 py-2 text-right">
                  <button onClick={() => void downloadOriginal(doc.id, doc.name)} className="text-indigo-600 hover:underline">
                    Download
                  </button>
                  <button onClick={() => setToDelete(doc)} className="text-red-600 hover:underline">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {list.data && list.data.items.length === 0 && (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-slate-500">
                  No documents yet. Upload some above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {toDelete && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-sm space-y-4 rounded-xl bg-white p-5">
            <p>
              Delete <b>{toDelete.name}</b>? Its file, its text and its search index are removed for good.
            </p>
            <div className="flex justify-end gap-2">
              <button onClick={() => setToDelete(null)} className="rounded border px-3 py-1.5 text-sm">
                Cancel
              </button>
              <button onClick={() => remove.mutate(toDelete)} disabled={remove.isPending} className="rounded bg-red-600 px-3 py-1.5 text-sm text-white">
                {remove.isPending ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
