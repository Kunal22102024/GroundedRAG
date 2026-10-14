import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { api } from "../../api/client";
import type { ChatFilters, DocumentList } from "../../lib/types";
import { btnGhost, fieldLabel, inputBase, selectBase } from "../../lib/ui";
import { ChevronDownIcon, XIcon } from "../ui/icons";
import { FileTypeBadge } from "../ui/primitives";

const DEPARTMENTS = ["HR", "Engineering", "Legal", "Operations", "IT"];
const TYPES = ["Policy", "Manual", "Technical Spec", "Contract"];

/** Pre-retrieval filters (PRD 4.2). The server always adds the caller's access levels on top. */
export default function FilterBar({ filters, onChange }: { filters: ChatFilters; onChange: (f: ChatFilters) => void }) {
  const docs = useQuery({
    queryKey: ["documents", "indexed"],
    queryFn: () => api<DocumentList>("/documents?status=indexed&limit=200"),
  });
  const items = docs.data?.items ?? [];
  const departments = [...new Set([...DEPARTMENTS, ...items.map((d) => d.department).filter(Boolean)])];
  const types = [...new Set([...TYPES, ...items.map((d) => d.document_type).filter(Boolean)])];
  const selected = filters.document_ids ?? [];
  const set = (patch: Partial<ChatFilters>) => onChange({ ...filters, ...patch });
  const active = Object.values(filters).some((v) => (Array.isArray(v) ? v.length : Boolean(v)));
  const badRange = Boolean(filters.date_from && filters.date_to && filters.date_from > filters.date_to);

  // A document that was deleted (or never existed for this user) cannot be unticked in the list, and
  // leaving it selected makes every question come back "not found". Drop such ids once the list is known.
  useEffect(() => {
    if (!docs.data || selected.length === 0) return;
    const known = new Set(docs.data.items.map((d) => d.id));
    const kept = selected.filter((id) => known.has(id));
    if (kept.length !== selected.length) onChange({ ...filters, document_ids: kept.length ? kept : undefined });
  }, [docs.data]); // only when the list of documents changes, not on every filter edit

  return (
    <div className="animate-fade-up border-b border-slate-200/80 bg-white px-3 py-3 sm:px-4">
      <div className="mx-auto grid max-w-3xl gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <label>
          <span className={fieldLabel}>Department</span>
          <select className={selectBase} value={filters.department ?? ""} onChange={(e) => set({ department: e.target.value || undefined })}>
            <option value="">All</option>
            {departments.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          <span className={fieldLabel}>Type</span>
          <select className={selectBase} value={filters.document_type ?? ""} onChange={(e) => set({ document_type: e.target.value || undefined })}>
            <option value="">All</option>
            {types.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </label>
        <label>
          <span className={fieldLabel}>From</span>
          <input type="date" className={`${inputBase} ${badRange ? "border-rose-300" : ""}`} value={filters.date_from ?? ""} onChange={(e) => set({ date_from: e.target.value || undefined })} />
        </label>
        <label>
          <span className={fieldLabel}>To</span>
          <input type="date" className={`${inputBase} ${badRange ? "border-rose-300" : ""}`} value={filters.date_to ?? ""} onChange={(e) => set({ date_to: e.target.value || undefined })} />
        </label>
        <div>
          <span className={fieldLabel}>Documents</span>
          <details className="group relative">
            <summary className={`${selectBase} flex list-none items-center justify-between`}>
              <span className="truncate">{selected.length ? `${selected.length} selected` : "All documents"}</span>
              <ChevronDownIcon size={15} className="shrink-0 text-slate-400 transition-transform group-open:rotate-180" />
            </summary>
            <div className="absolute right-0 z-20 mt-1 max-h-72 w-80 animate-scale-in overflow-y-auto rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl">
              {items.length === 0 && <p className="p-3 text-sm text-slate-500">No indexed documents.</p>}
              {items.map((doc) => (
                <label key={doc.id} className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-indigo-600"
                    checked={selected.includes(doc.id)}
                    onChange={(e) => set({ document_ids: e.target.checked ? [...selected, doc.id] : selected.filter((id) => id !== doc.id) })}
                  />
                  <FileTypeBadge name={doc.name} size="sm" />
                  <span className="truncate" title={doc.name}>
                    {doc.name}
                  </span>
                </label>
              ))}
            </div>
          </details>
        </div>
      </div>
      {(badRange || active) && (
        <div className="mx-auto mt-2 flex max-w-3xl items-center gap-3">
          {badRange && (
            <span role="alert" className="text-sm text-rose-600">
              The From date is after the To date.
            </span>
          )}
          {active && (
            <button onClick={() => onChange({})} className={`${btnGhost} ml-auto`}>
              <XIcon size={14} /> Clear filters
            </button>
          )}
        </div>
      )}
    </div>
  );
}
