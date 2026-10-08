import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { ChatFilters, DocumentList } from "../../lib/types";

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
  const field = "rounded border bg-white px-2 py-1 text-sm";

  return (
    <div className="flex flex-wrap items-end gap-3 border-b bg-white px-4 py-2 text-xs text-slate-500">
      <label>
        Department
        <select className={`${field} block`} value={filters.department ?? ""} onChange={(e) => set({ department: e.target.value || undefined })}>
          <option value="">All</option>
          {departments.map((d) => (
            <option key={d}>{d}</option>
          ))}
        </select>
      </label>
      <label>
        Type
        <select className={`${field} block`} value={filters.document_type ?? ""} onChange={(e) => set({ document_type: e.target.value || undefined })}>
          <option value="">All</option>
          {types.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
      </label>
      <label>
        From
        <input type="date" className={`${field} block`} value={filters.date_from ?? ""} onChange={(e) => set({ date_from: e.target.value || undefined })} />
      </label>
      <label>
        To
        <input type="date" className={`${field} block`} value={filters.date_to ?? ""} onChange={(e) => set({ date_to: e.target.value || undefined })} />
      </label>
      <details className="relative">
        <summary className={`${field} cursor-pointer list-none`}>Documents{selected.length ? ` (${selected.length})` : ": all"}</summary>
        <div className="absolute z-20 mt-1 max-h-64 w-72 overflow-y-auto rounded border bg-white p-2 text-sm text-slate-700 shadow-lg">
          {items.length === 0 && <p className="text-slate-500">No indexed documents.</p>}
          {items.map((doc) => (
            <label key={doc.id} className="flex items-center gap-2 py-0.5">
              <input
                type="checkbox"
                checked={selected.includes(doc.id)}
                onChange={(e) => set({ document_ids: e.target.checked ? [...selected, doc.id] : selected.filter((id) => id !== doc.id) })}
              />
              <span className="truncate" title={doc.name}>
                {doc.name}
              </span>
            </label>
          ))}
        </div>
      </details>
      {active && (
        <button onClick={() => onChange({})} className="rounded border px-2 py-1 text-sm text-slate-700 hover:bg-slate-100">
          Clear filters
        </button>
      )}
    </div>
  );
}
