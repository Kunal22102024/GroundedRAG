import { useState } from "react";
import type { DocumentItem } from "../../lib/types";
import { fieldLabel, inputBase, selectBase } from "../../lib/ui";
import { CheckIcon, SearchIcon } from "../ui/icons";
import { FileTypeBadge } from "../ui/primitives";

/** One document, chosen from a list (Extract and Summaries). */
export function DocumentSelect({
  documents,
  value,
  onChange,
  label = "Document",
  exclude,
  placeholder = "Choose a document…",
  disabled = false,
  id,
}: {
  documents: DocumentItem[];
  value: string;
  onChange: (id: string) => void;
  label?: string;
  exclude?: string;
  placeholder?: string;
  disabled?: boolean;
  id: string;
}) {
  return (
    <div>
      <label htmlFor={id} className={fieldLabel}>
        {label}
      </label>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} className={selectBase}>
        <option value="">{placeholder}</option>
        {documents
          .filter((d) => d.id !== exclude)
          .map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
              {d.document_type ? ` · ${d.document_type}` : ""}
            </option>
          ))}
      </select>
    </div>
  );
}

/** Several documents, toggled on and off (Compare), up to `max`. */
export function DocumentMultiPicker({
  documents,
  selected,
  onChange,
  max,
  disabled = false,
}: {
  documents: DocumentItem[];
  selected: string[];
  onChange: (ids: string[]) => void;
  max: number;
  disabled?: boolean;
}) {
  const [search, setSearch] = useState("");
  const shown = documents.filter((d) => d.name.toLowerCase().includes(search.trim().toLowerCase()));
  const full = selected.length >= max;
  const toggle = (id: string) => onChange(selected.includes(id) ? selected.filter((s) => s !== id) : full ? selected : [...selected, id]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className={`${fieldLabel} mb-0`}>
          Documents to compare <span className="text-slate-400">(choose 2 to {max})</span>
        </span>
        <span className={`text-xs tabular-nums ${selected.length >= 2 ? "text-indigo-600" : "text-slate-400"}`}>
          {selected.length} of {max} chosen
        </span>
      </div>
      {documents.length > 8 && (
        <div className="relative">
          <SearchIcon size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search documents" aria-label="Search documents" className={`${inputBase} pl-9`} />
        </div>
      )}
      <ul className="grid max-h-60 gap-2 overflow-y-auto p-0.5 sm:grid-cols-2 lg:grid-cols-3" aria-label="Documents">
        {shown.map((d) => {
          const on = selected.includes(d.id);
          const blocked = !on && full;
          return (
            <li key={d.id} className="min-w-0">
              <button
                type="button"
                onClick={() => toggle(d.id)}
                disabled={disabled || blocked}
                aria-pressed={on}
                title={blocked ? `At most ${max} documents` : d.name}
                className={`flex w-full items-center gap-2.5 rounded-xl border p-2.5 text-left transition-all disabled:cursor-not-allowed ${
                  on ? "border-indigo-300 bg-indigo-50/70 shadow-sm ring-2 ring-indigo-500/10" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                } ${blocked ? "opacity-50" : ""}`}
              >
                <FileTypeBadge name={d.name} size="sm" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-slate-800">{d.name}</span>
                  <span className="block truncate text-[11px] text-slate-500">{[d.department, d.document_type].filter(Boolean).join(" · ") || "—"}</span>
                </span>
                <span
                  aria-hidden="true"
                  className={`grid h-5 w-5 shrink-0 place-items-center rounded-full border transition-all ${on ? "scale-100 border-indigo-600 bg-indigo-600 text-white" : "border-slate-300 text-transparent"}`}
                >
                  <CheckIcon size={12} strokeWidth={3} />
                </span>
              </button>
            </li>
          );
        })}
        {shown.length === 0 && <li className="col-span-full px-1 py-3 text-sm text-slate-500">Nothing matches "{search}".</li>}
      </ul>
    </div>
  );
}
