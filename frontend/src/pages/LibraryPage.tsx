import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ApiError, api, uploadDocuments } from "../api/client";
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  ChevronDownIcon,
  ClockIcon,
  DatabaseIcon,
  DownloadIcon,
  FileTextIcon,
  SearchIcon,
  SparklesIcon,
  TrashIcon,
  UploadIcon,
  XIcon,
} from "../components/ui/icons";
import { AccessBadge, FileTypeBadge, Modal, ProgressBar, Skeleton, Spinner, StatusPill } from "../components/ui/primitives";
import { useToast } from "../components/ui/Toast";
import { useAuth } from "../hooks/useAuth";
import { downloadOriginal, formatBytes } from "../lib/files";
import { formatDate, timeAgo } from "../lib/format";
import type { DocStatus, DocumentItem, DocumentList } from "../lib/types";
import { btnDanger, btnGhost, btnPrimary, btnSecondary, card, fieldLabel, iconBtn, inputBase, selectBase } from "../lib/ui";

const DEPARTMENTS = ["HR", "Engineering", "Legal", "Operations", "IT"];
const TYPES = ["Policy", "Manual", "Technical Spec", "Contract"];
const ACCEPT = ".pdf,.docx,.md,.markdown,.html,.htm,.txt";

// --- Uploads: bytes sent, then indexing, per file ----------------------------------------------------

type JobPhase = "uploading" | "indexing" | "rejected" | "error" | "cancelled";

interface UploadJob {
  key: string;
  batch: number;
  index: number; // position in its batch, for the progress maths
  name: string;
  size: number;
  phase: JobPhase;
  progress: number; // 0..1 of this file's bytes sent
  documentId?: string;
  message?: string;
}

const clamp = (n: number) => Math.min(1, Math.max(0, n));

function isRejection(value: unknown): value is { filename: string; reason: string } {
  return typeof value === "object" && value !== null && "filename" in value && "reason" in value;
}

/** Uploads with progress, then follows each accepted document until it is indexed or fails. */
function useUploads() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const controllers = useRef(new Map<number, AbortController>());
  const batchCounter = useRef(0);
  const announced = useRef(new Set<string>());

  const patchBatch = (batch: number, patch: (job: UploadJob) => UploadJob) =>
    setJobs((all) => all.map((job) => (job.batch === batch ? patch(job) : job)));

  const start = (files: File[], fields: Record<string, string>) => {
    const batch = ++batchCounter.current;
    const fresh: UploadJob[] = files.map((file, index) => ({
      key: `${batch}-${index}`,
      batch,
      index,
      name: file.name,
      size: file.size,
      phase: "uploading",
      progress: 0,
    }));
    setJobs((all) => [...fresh, ...all]);

    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    Object.entries(fields).forEach(([key, value]) => value && form.append(key, value));

    // One request carries every file, in order: spread the bytes sent over the files by their sizes.
    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    const offsets = files.map((_, i) => files.slice(0, i).reduce((sum, f) => sum + f.size, 0));
    const controller = new AbortController();
    controllers.current.set(batch, controller);

    uploadDocuments(
      form,
      ({ loaded, total }) => {
        const sent = (loaded / Math.max(total, 1)) * totalSize;
        patchBatch(batch, (job) =>
          job.phase !== "uploading" ? job : { ...job, progress: job.size ? clamp((sent - offsets[job.index]) / job.size) : sent >= offsets[job.index] ? 1 : 0 },
        );
      },
      controller.signal,
    )
      .then((result) => {
        // The server answers with the accepted documents in upload order, and the refused files by name.
        const rejected = [...result.rejected];
        const accepted = [...result.documents];
        const outcome = new Map<string, Partial<UploadJob>>();
        for (const job of fresh) {
          const r = rejected.findIndex((x) => x.filename === job.name.slice(0, 120));
          if (r >= 0) {
            outcome.set(job.key, { phase: "rejected", progress: 1, message: rejected[r].reason });
            rejected.splice(r, 1);
          } else {
            const doc = accepted.shift();
            outcome.set(job.key, doc ? { phase: "indexing", progress: 1, documentId: doc.id } : { phase: "error", message: "Not in the server's reply." });
          }
        }
        setJobs((all) => all.map((job) => (outcome.has(job.key) ? { ...job, ...outcome.get(job.key) } : job)));
        void queryClient.invalidateQueries({ queryKey: ["documents"] });
        const n = result.documents.length;
        toast({
          tone: result.rejected.length ? "info" : "success",
          title: `${n} file${n === 1 ? "" : "s"} uploaded`,
          description: result.rejected.length ? `${result.rejected.length} not accepted. Indexing has started for the rest.` : "Indexing has started. You can keep working.",
        });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          patchBatch(batch, (job) => ({ ...job, phase: "cancelled" }));
          return;
        }
        const message = error instanceof Error ? error.message : "The upload failed.";
        // "No file was accepted" carries a reason for each file.
        const reasons =
          error instanceof ApiError && typeof error.detail === "object" && error.detail !== null && Array.isArray((error.detail as { rejected?: unknown }).rejected)
            ? ((error.detail as { rejected: unknown[] }).rejected.filter(isRejection))
            : [];
        patchBatch(batch, (job) => {
          const reason = reasons.find((r) => r.filename === job.name.slice(0, 120));
          return reason ? { ...job, phase: "rejected", message: reason.reason } : { ...job, phase: "error", message };
        });
        toast({ tone: "error", title: "Upload failed", description: message });
      })
      .finally(() => controllers.current.delete(batch));
  };

  const cancel = (batch: number) => controllers.current.get(batch)?.abort();
  const clearFinished = (finished: (job: UploadJob) => boolean) => setJobs((all) => all.filter((job) => !finished(job)));

  // Follow every accepted document until it is indexed or failed.
  const tracked = jobs.filter((job) => job.phase === "indexing" && job.documentId);
  const results = useQueries({
    queries: tracked.map((job) => ({
      queryKey: ["document", job.documentId],
      queryFn: () => api<DocumentItem>(`/documents/${job.documentId}`),
      refetchInterval: (query: { state: { data?: DocumentItem } }) => {
        const status = query.state.data?.status;
        return status === "indexed" || status === "failed" ? false : 1500;
      },
    })),
  });
  const docs = new Map<string, DocumentItem | undefined>(tracked.map((job, i) => [job.documentId as string, results[i]?.data]));

  // Announce each document once when it finishes, and refresh the library list.
  const signature = tracked.map((job) => `${job.documentId}:${docs.get(job.documentId as string)?.status ?? ""}`).join(",");
  useEffect(() => {
    let changed = false;
    for (const doc of docs.values()) {
      if (!doc || announced.current.has(doc.id) || (doc.status !== "indexed" && doc.status !== "failed")) continue;
      announced.current.add(doc.id);
      changed = true;
      if (doc.status === "indexed") toast({ tone: "success", title: "Ready to search", description: `${doc.name} · ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"} indexed` });
      else toast({ tone: "error", title: `${doc.name} could not be indexed`, description: doc.error_message ?? undefined });
    }
    if (changed) void queryClient.invalidateQueries({ queryKey: ["documents"] });
  }, [signature]); // `signature` changes exactly when a followed document's status does

  return { jobs, docs, start, cancel, clearFinished };
}

function jobFinished(job: UploadJob, doc: DocumentItem | undefined): boolean {
  if (job.phase === "uploading") return false;
  if (job.phase === "indexing") return doc?.status === "indexed" || doc?.status === "failed";
  return true;
}

function JobRow({ job, doc }: { job: UploadJob; doc: DocumentItem | undefined }) {
  let status: ReactNode;
  let bar: ReactNode = null;
  if (job.phase === "uploading") {
    status = (
      <span className="text-indigo-600">
        Uploading · {Math.round(job.progress * 100)}% <span className="text-slate-400">of {formatBytes(job.size)}</span>
      </span>
    );
    bar = <ProgressBar value={job.progress} label={`Uploading ${job.name}`} />;
  } else if (job.phase === "indexing") {
    const s = doc?.status ?? "queued";
    if (s === "indexed") {
      status = (
        <span className="inline-flex items-center gap-1 text-emerald-600">
          <CheckCircleIcon size={14} /> Ready · {doc?.chunk_count} chunk{doc?.chunk_count === 1 ? "" : "s"}
        </span>
      );
      bar = <ProgressBar value={1} tone="emerald" label={`${job.name} indexed`} />;
    } else if (s === "failed") {
      status = <span className="text-rose-600">{doc?.error_message ?? "Indexing failed"}</span>;
      bar = <ProgressBar value={1} tone="rose" label={`${job.name} failed`} />;
    } else {
      status = (
        <span className="inline-flex items-center gap-1.5 text-slate-600">
          <Spinner size={12} className="text-indigo-500" />
          {s === "processing" ? "Reading, chunking and embedding…" : "Waiting in the indexing queue…"}
        </span>
      );
      bar = <ProgressBar tone={s === "processing" ? "indigo" : "slate"} label={`Indexing ${job.name}`} />;
    }
  } else if (job.phase === "rejected") {
    status = <span className="text-amber-700">Not accepted: {job.message}</span>;
  } else if (job.phase === "cancelled") {
    status = <span className="text-slate-500">Cancelled</span>;
  } else {
    status = <span className="text-rose-600">{job.message}</span>;
  }
  return (
    <li className="flex animate-fade-up items-center gap-3 py-2.5">
      <FileTypeBadge name={job.name} size="sm" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex items-baseline justify-between gap-3">
          <p className="truncate text-sm font-medium text-slate-800" title={job.name}>
            {job.name}
          </p>
        </div>
        <p className="truncate text-xs">{status}</p>
        {bar}
      </div>
    </li>
  );
}

function UploadTray({ uploads }: { uploads: ReturnType<typeof useUploads> }) {
  const { jobs, docs, cancel, clearFinished } = uploads;
  if (!jobs.length) return null;
  const docOf = (job: UploadJob) => (job.documentId ? docs.get(job.documentId) : undefined);
  const uploading = jobs.filter((j) => j.phase === "uploading");
  const batches = [...new Set(uploading.map((j) => j.batch))];
  const sent = uploading.reduce((s, j) => s + j.size * j.progress, 0);
  const total = uploading.reduce((s, j) => s + j.size, 0);
  const inFlight = jobs.filter((j) => !jobFinished(j, docOf(j))).length;
  return (
    <section className={`${card} animate-fade-up p-5`} aria-label="Uploads">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <h2 className="text-sm font-semibold text-slate-900">Uploads</h2>
        {uploading.length > 0 ? (
          <span className="text-xs text-slate-500">
            Sending {uploading.length} file{uploading.length > 1 ? "s" : ""} · {Math.round((sent / Math.max(total, 1)) * 100)}%
          </span>
        ) : inFlight > 0 ? (
          <span className="text-xs text-slate-500">{inFlight} still indexing</span>
        ) : (
          <span className="text-xs text-emerald-600">All done</span>
        )}
        <div className="ml-auto flex gap-1">
          {batches.map((b) => (
            <button key={b} onClick={() => cancel(b)} className={btnGhost}>
              <XIcon size={14} /> Cancel upload
            </button>
          ))}
          {jobs.length > inFlight && (
            <button onClick={() => clearFinished((j) => jobFinished(j, docOf(j)))} className={btnGhost}>
              Clear finished
            </button>
          )}
        </div>
      </div>
      <ul className="max-h-80 divide-y divide-slate-100 overflow-y-auto pr-1">
        {jobs.map((job) => (
          <JobRow key={job.key} job={job} doc={docOf(job)} />
        ))}
      </ul>
    </section>
  );
}

// --- Choosing files ----------------------------------------------------------------------------------

function UploadPanel({ isAdmin, onUpload }: { isAdmin: boolean; onUpload: (files: File[], fields: Record<string, string>) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [department, setDepartment] = useState("");
  const [documentType, setDocumentType] = useState("");
  const [date, setDate] = useState("");
  const [access, setAccess] = useState("");
  const [showOptions, setShowOptions] = useState(false);
  const [dragging, setDragging] = useState(false);

  const add = (list: FileList | null) => {
    if (!list) return;
    setFiles((current) => {
      const seen = new Set(current.map((f) => `${f.name}:${f.size}`));
      return [...current, ...Array.from(list).filter((f) => !seen.has(`${f.name}:${f.size}`))];
    });
  };
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    add(event.dataTransfer.files);
  };
  const onDropzoneKey = (event: KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.current?.click();
    }
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!files.length) return;
    onUpload(files, { department, document_type: documentType, creation_date: date, access_level: access });
    setFiles([]);
    if (input.current) input.current.value = "";
  };
  const totalSize = files.reduce((s, f) => s + f.size, 0);

  return (
    <form onSubmit={submit} className={`${card} p-5`}>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        onKeyDown={onDropzoneKey}
        role="button"
        tabIndex={0}
        aria-label="Choose files to upload, or drop them here"
        className={`group relative flex cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border-2 border-dashed px-6 py-9 text-center transition-all duration-200 ${
          dragging ? "scale-[1.01] border-indigo-500 bg-indigo-50/70" : "border-slate-200 bg-slate-50/60 hover:border-indigo-300 hover:bg-indigo-50/40"
        }`}
      >
        <span
          className={`grid h-14 w-14 place-items-center rounded-2xl bg-linear-to-br from-indigo-500 to-violet-500 text-white shadow-lg shadow-indigo-500/30 transition-transform duration-300 ${
            dragging ? "-translate-y-1 scale-110" : "group-hover:-translate-y-0.5"
          }`}
        >
          <UploadIcon size={26} />
        </span>
        <div>
          <p className="text-sm font-semibold text-slate-800">{dragging ? "Drop to add them" : "Drag files here, or click to browse"}</p>
          <p className="mt-1 text-xs text-slate-500">PDF, Word (.docx), Markdown, HTML or plain text · up to 50 MB each</p>
        </div>
        <input ref={input} type="file" multiple accept={ACCEPT} tabIndex={-1} className="hidden" onChange={(e) => add(e.target.files)} />
      </div>

      {files.length > 0 && (
        <ul className="mt-4 grid gap-2 sm:grid-cols-2">
          {files.map((file) => (
            <li key={`${file.name}:${file.size}`} className="flex animate-scale-in items-center gap-3 rounded-xl border border-slate-200 bg-white p-2.5">
              <FileTypeBadge name={file.name} size="sm" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800" title={file.name}>
                  {file.name}
                </p>
                <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
              </div>
              <button
                type="button"
                onClick={() => setFiles((all) => all.filter((f) => f !== file))}
                aria-label={`Remove ${file.name}`}
                className={iconBtn}
              >
                <XIcon size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => setShowOptions((v) => !v)}
          aria-expanded={showOptions}
          className={btnGhost}
        >
          <ChevronDownIcon size={16} className={`transition-transform duration-200 ${showOptions ? "rotate-180" : ""}`} />
          Details (optional)
        </button>
        <span className="text-xs text-slate-400">Department, type and date are detected automatically.</span>
        <div className="ml-auto flex items-center gap-3">
          {files.length > 0 && (
            <button type="button" onClick={() => setFiles([])} className={btnGhost}>
              Clear
            </button>
          )}
          <button disabled={!files.length} className={btnPrimary}>
            <UploadIcon size={16} />
            {files.length ? `Upload ${files.length} file${files.length > 1 ? "s" : ""} · ${formatBytes(totalSize)}` : "Upload"}
          </button>
        </div>
      </div>

      {showOptions && (
        <div className="mt-4 grid animate-fade-up gap-3 rounded-xl bg-slate-50 p-4 sm:grid-cols-2 lg:grid-cols-4">
          <label>
            <span className={fieldLabel}>Department</span>
            <select value={department} onChange={(e) => setDepartment(e.target.value)} className={selectBase}>
              <option value="">Detect automatically</option>
              {DEPARTMENTS.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
          <label>
            <span className={fieldLabel}>Document type</span>
            <select value={documentType} onChange={(e) => setDocumentType(e.target.value)} className={selectBase}>
              <option value="">Detect automatically</option>
              {TYPES.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            <span className={fieldLabel}>Creation date</span>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className={inputBase} />
          </label>
          <label>
            <span className={fieldLabel}>Access level</span>
            <select value={access} onChange={(e) => setAccess(e.target.value)} className={selectBase}>
              <option value="">Internal (default)</option>
              <option>Public</option>
              {isAdmin && <option>Restricted</option>}
            </select>
          </label>
        </div>
      )}
    </form>
  );
}

// --- The library -----------------------------------------------------------------------------------

function StatCard({ icon, label, value, accent, live }: { icon: ReactNode; label: string; value: ReactNode; accent: string; live?: boolean }) {
  return (
    <div className={`${card} flex items-center gap-4 p-4`}>
      <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-xl ${accent}`}>{icon}</span>
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-500">{label}</p>
        <p className="flex items-center gap-2 text-xl font-semibold tabular-nums text-slate-900">
          {value}
          {live && <Spinner size={14} className="text-amber-500" />}
        </p>
      </div>
    </div>
  );
}

const STATUS_TABS: { value: "" | DocStatus; label: string }[] = [
  { value: "", label: "All" },
  { value: "indexed", label: "Ready" },
  { value: "processing", label: "Indexing" },
  { value: "queued", label: "Queued" },
  { value: "failed", label: "Failed" },
];

export default function LibraryPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const uploads = useUploads();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<"" | DocStatus>("");
  const [toDelete, setToDelete] = useState<DocumentItem | null>(null);
  const cancelButton = useRef<HTMLButtonElement>(null);

  const list = useQuery({
    queryKey: ["documents", { q, status }],
    queryFn: () => {
      const params = new URLSearchParams({ limit: "200" });
      if (q) params.set("q", q);
      if (status) params.set("status", status);
      return api<DocumentList>(`/documents?${params}`);
    },
    // While something is still indexing, or an indexed document's summary is still being written.
    refetchInterval: (query) =>
      query.state.data?.items.some((d) => d.status === "queued" || d.status === "processing" || (d.status === "indexed" && d.executive_summary === "pending")) ? 2000 : false,
  });

  const download = (doc: DocumentItem) => {
    downloadOriginal(doc.id, doc.name).catch((e: unknown) =>
      toast({ tone: "error", title: "Download failed", description: e instanceof Error ? e.message : undefined }),
    );
  };

  const remove = useMutation({
    mutationFn: (doc: DocumentItem) => api<void>(`/documents/${doc.id}`, { method: "DELETE" }),
    onSuccess: (_, doc) => {
      setToDelete(null);
      toast({ tone: "success", title: "Document deleted", description: `${doc.name} was removed from the library and the search index.` });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (e) => {
      setToDelete(null);
      toast({ tone: "error", title: "Delete failed", description: e instanceof Error ? e.message : undefined });
    },
  });

  const summary = list.data?.summary;
  const byStatus = summary?.by_status ?? {};
  const inProgress = (byStatus.queued ?? 0) + (byStatus.processing ?? 0);
  const items = list.data?.items ?? [];
  const filtering = Boolean(q || status);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <header className="animate-fade-up">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Document library</h1>
          <p className="mt-1 text-sm text-slate-500">Upload files to make them searchable. Answers in Chat only ever come from documents listed here.</p>
        </header>

        <div className="grid animate-fade-up gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard icon={<FileTextIcon size={20} />} label="Documents" value={summary?.total_files ?? "—"} accent="bg-indigo-50 text-indigo-600" />
          <StatCard icon={<CheckCircleIcon size={20} />} label="Ready to search" value={byStatus.indexed ?? 0} accent="bg-emerald-50 text-emerald-600" />
          <StatCard icon={<ClockIcon size={20} />} label="In progress" value={inProgress} accent="bg-amber-50 text-amber-600" live={inProgress > 0} />
          <StatCard icon={<DatabaseIcon size={20} />} label="Searchable chunks" value={summary?.total_chunks ?? "—"} accent="bg-violet-50 text-violet-600" />
        </div>

        <UploadPanel isAdmin={user?.role === "admin"} onUpload={uploads.start} />
        <UploadTray uploads={uploads} />

        <section className={`${card} overflow-hidden`} aria-label="Documents">
          <div className="flex flex-wrap items-center gap-3 border-b border-slate-100 p-4">
            <div className="relative min-w-56 flex-1 sm:max-w-xs">
              <SearchIcon size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by file name" aria-label="Search by file name" className={`${inputBase} pl-9`} />
            </div>
            <div role="tablist" aria-label="Filter by status" className="flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1">
              {STATUS_TABS.map((tab) => (
                <button
                  key={tab.label}
                  role="tab"
                  aria-selected={status === tab.value}
                  onClick={() => setStatus(tab.value)}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-all ${
                    status === tab.value ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  {tab.label}
                  {tab.value && byStatus[tab.value] ? <span className="ml-1 text-slate-400">{byStatus[tab.value]}</span> : null}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50/70 text-xs font-medium text-slate-500">
                <tr>
                  <th className="px-4 py-2.5">Document</th>
                  <th className="px-4 py-2.5">Status</th>
                  <th className="hidden px-4 py-2.5 sm:table-cell">Chunks</th>
                  <th className="hidden px-4 py-2.5 lg:table-cell">Department · Type</th>
                  <th className="hidden px-4 py-2.5 md:table-cell">Date</th>
                  <th className="hidden px-4 py-2.5 md:table-cell">Access</th>
                  <th className="px-4 py-2.5">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {list.isLoading &&
                  [0, 1, 2, 3].map((i) => (
                    <tr key={i}>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <Skeleton className="h-10 w-10 rounded-lg" />
                          <div className="space-y-2">
                            <Skeleton className="h-3.5 w-48" />
                            <Skeleton className="h-3 w-28" />
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <Skeleton className="h-5 w-16 rounded-full" />
                      </td>
                      <td className="hidden px-4 py-3 sm:table-cell">
                        <Skeleton className="h-3.5 w-8" />
                      </td>
                      <td className="hidden px-4 py-3 lg:table-cell">
                        <Skeleton className="h-3.5 w-28" />
                      </td>
                      <td className="hidden px-4 py-3 md:table-cell">
                        <Skeleton className="h-3.5 w-20" />
                      </td>
                      <td className="hidden px-4 py-3 md:table-cell">
                        <Skeleton className="h-5 w-16" />
                      </td>
                      <td />
                    </tr>
                  ))}
                {items.map((doc, i) => (
                  <tr key={doc.id} className="group animate-fade-up transition-colors hover:bg-slate-50/70" style={{ animationDelay: `${Math.min(i, 12) * 25}ms` }}>
                    <td className="max-w-md px-4 py-3">
                      <div className="flex items-center gap-3">
                        <FileTypeBadge name={doc.name} />
                        <div className="min-w-0">
                          <p className="truncate font-medium text-slate-900" title={doc.name}>
                            {doc.name}
                          </p>
                          <p className="text-xs text-slate-500">
                            {formatBytes(doc.size_bytes)} · added {timeAgo(doc.created_at)}
                          </p>
                          {doc.status === "failed" && doc.error_message && (
                            <p className="mt-0.5 flex items-center gap-1 truncate text-xs text-rose-600" title={doc.error_message}>
                              <AlertTriangleIcon size={12} /> {doc.error_message}
                            </p>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusPill status={doc.status} title={doc.error_message ?? undefined} />
                      {doc.status === "indexed" && doc.executive_summary === "pending" && (
                        <p className="mt-1 flex items-center gap-1 text-[11px] text-slate-500">
                          <Spinner size={11} className="text-indigo-500" /> Writing summary…
                        </p>
                      )}
                      {doc.status === "indexed" && doc.executive_summary === "failed" && (
                        <Link to={`/analyze?tab=summaries&doc=${doc.id}`} className="mt-1 inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 hover:underline">
                          <AlertTriangleIcon size={11} /> Retry summary
                        </Link>
                      )}
                      {doc.status === "indexed" && doc.executive_summary === "ready" && (
                        <Link to={`/analyze?tab=summaries&doc=${doc.id}`} className="mt-1 inline-flex items-center gap-1 text-[11px] font-medium text-indigo-600 hover:underline">
                          <SparklesIcon size={11} /> Summary ready
                        </Link>
                      )}
                    </td>
                    <td className="hidden px-4 py-3 tabular-nums text-slate-600 sm:table-cell">{doc.status === "indexed" ? doc.chunk_count : "—"}</td>
                    <td className="hidden px-4 py-3 lg:table-cell">
                      <div className="flex flex-wrap gap-1">
                        {doc.department ? <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600">{doc.department}</span> : null}
                        {doc.document_type ? <span className="rounded-md bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700">{doc.document_type}</span> : null}
                        {!doc.department && !doc.document_type && <span className="text-slate-400">—</span>}
                      </div>
                    </td>
                    <td className="hidden whitespace-nowrap px-4 py-3 text-slate-600 md:table-cell">{formatDate(doc.creation_date)}</td>
                    <td className="hidden px-4 py-3 md:table-cell">
                      <AccessBadge level={doc.access_level} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex justify-end gap-1 opacity-70 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                        {doc.status === "indexed" && (
                          <Link
                            to={`/analyze?tab=summaries&doc=${doc.id}`}
                            aria-label={`Analyze ${doc.name}`}
                            title="Analyze: summaries, extraction, comparison"
                            className={`${iconBtn} hover:bg-indigo-50 hover:text-indigo-600`}
                          >
                            <SparklesIcon size={16} />
                          </Link>
                        )}
                        <button onClick={() => download(doc)} aria-label={`Download ${doc.name}`} title="Download" className={iconBtn}>
                          <DownloadIcon size={16} />
                        </button>
                        <button
                          onClick={() => setToDelete(doc)}
                          aria-label={`Delete ${doc.name}`}
                          title="Delete"
                          className={`${iconBtn} hover:bg-rose-50 hover:text-rose-600`}
                        >
                          <TrashIcon size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {list.data && items.length === 0 && (
            <div className="flex animate-fade-in flex-col items-center px-6 py-14 text-center">
              <span className="grid h-14 w-14 place-items-center rounded-2xl bg-slate-100 text-slate-400">
                <FileTextIcon size={26} />
              </span>
              <p className="mt-4 font-medium text-slate-800">{filtering ? "No documents match" : "No documents yet"}</p>
              <p className="mt-1 max-w-sm text-sm text-slate-500">
                {filtering ? "Try another name or status." : "Upload a PDF, Word file or text document above to start asking questions about it."}
              </p>
              {filtering && (
                <button
                  onClick={() => {
                    setQ("");
                    setStatus("");
                  }}
                  className={`${btnSecondary} mt-4`}
                >
                  Clear search
                </button>
              )}
            </div>
          )}
          {list.isError && (
            <p role="alert" className="p-6 text-center text-sm text-rose-600">
              The documents could not be loaded. {list.error instanceof Error ? list.error.message : ""}
            </p>
          )}
        </section>
      </div>

      <Modal open={toDelete !== null} onClose={() => setToDelete(null)} label={toDelete ? `Delete ${toDelete.name}` : "Delete"} initialFocus={cancelButton}>
        {toDelete && (
          <div className="space-y-5">
            <div className="flex items-start gap-4">
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-rose-50 text-rose-600">
                <TrashIcon size={20} />
              </span>
              <div className="min-w-0 pr-6">
                <h2 className="font-semibold text-slate-900">Delete this document?</h2>
                <p className="mt-1 break-words text-sm text-slate-500">
                  <b className="text-slate-700">{toDelete.name}</b> will be removed with its text and search index. Answers will never cite it again. This
                  cannot be undone.
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <button ref={cancelButton} onClick={() => setToDelete(null)} className={btnSecondary}>
                Cancel
              </button>
              <button onClick={() => remove.mutate(toDelete)} disabled={remove.isPending} className={btnDanger}>
                {remove.isPending ? <Spinner size={14} /> : <TrashIcon size={15} />}
                {remove.isPending ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
