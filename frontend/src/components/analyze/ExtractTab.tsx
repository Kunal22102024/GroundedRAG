import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState, type FormEvent, type KeyboardEvent } from "react";
import { api } from "../../api/client";
import { useAnalysisRun } from "../../hooks/useAnalysisRun";
import { useIndexedDocuments } from "../../hooks/useIndexedDocuments";
import { extractionCsv, extractionJson, extractionRows } from "../../lib/analysis";
import { baseName, saveText } from "../../lib/files";
import { timeAgo } from "../../lib/format";
import type { Citation, ExtractionContent, Insight } from "../../lib/types";
import { btnGhost, btnPrimary, btnSecondary, card, fieldLabel, inputBase } from "../../lib/ui";
import CitationChip from "../chat/CitationChip";
import { AlertCircleIcon, CheckIcon, CopyIcon, DownloadIcon, InfoIcon, PlusIcon, RefreshIcon, TableIcon, XIcon } from "../ui/icons";
import { ProgressBar, Skeleton } from "../ui/primitives";
import { useToast } from "../ui/Toast";
import { DocumentSelect } from "./DocumentPicker";

const MAX_FIELDS = 20;
const MAX_FIELD_CHARS = 60;

const PRESETS = [
  { key: "dates", label: "Dates", hint: "Effective, signing and review dates" },
  { key: "parties", label: "Parties", hint: "People and organisations, with their role" },
  { key: "amounts", label: "Amounts", hint: "Money, quantities, percentages, limits" },
  { key: "requirements", label: "Requirements", hint: "What must or should be done, and by whom" },
  { key: "deadlines", label: "Deadlines", hint: "Due dates, notice and retention periods" },
  { key: "contacts", label: "Contacts", hint: "Names, emails and phone numbers" },
  { key: "definitions", label: "Definitions", hint: "Terms the document defines" },
];

function Results({ documentName, content, cached, onRefresh, refreshing, onCite }: {
  documentName: string;
  content: ExtractionContent;
  cached: boolean;
  onRefresh: () => void;
  refreshing: boolean;
  onCite: (c: Citation) => void;
}) {
  const { toast } = useToast();
  const [view, setView] = useState<"table" | "json">("table");
  const [copied, setCopied] = useState(false);
  const rows = useMemo(() => extractionRows(content), [content]);
  const json = useMemo(() => extractionJson(documentName, content), [documentName, content]);
  const found = rows.filter((r) => r.value !== null).length;
  const stem = baseName(documentName);

  const copy = () => {
    navigator.clipboard
      .writeText(json)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => toast({ tone: "error", title: "Could not copy" }));
  };

  return (
    <section className={`${card} animate-fade-up overflow-hidden`} aria-label="Extraction result">
      <header className="flex flex-wrap items-center gap-2 border-b border-slate-100 p-3 sm:px-4">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-slate-900">
            {found} value{found === 1 ? "" : "s"} from {content.fields.length} field{content.fields.length === 1 ? "" : "s"}
          </h3>
          {cached && <p className="text-xs text-slate-500">Saved result, no new AI call was made.</p>}
        </div>
        <div className="flex rounded-lg bg-slate-100 p-0.5 text-xs font-medium" role="group" aria-label="View">
          {(["table", "json"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)} aria-pressed={view === v} className={`rounded-md px-2.5 py-1 transition ${view === v ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}>
              {v === "table" ? "Table" : "JSON"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={copy} className={btnGhost} title="Copy as JSON">
            {copied ? <CheckIcon size={15} className="animate-pop text-emerald-600" /> : <CopyIcon size={15} />} Copy
          </button>
          <button onClick={() => saveText(`${stem}-extraction.json`, "application/json", json)} className={btnGhost} title="Download JSON">
            <DownloadIcon size={15} /> JSON
          </button>
          <button onClick={() => saveText(`${stem}-extraction.csv`, "text/csv", `﻿${extractionCsv(documentName, content)}`)} className={btnGhost} title="Download CSV">
            <DownloadIcon size={15} /> CSV
          </button>
          <button onClick={onRefresh} disabled={refreshing} className={btnGhost} title="Extract again (asks the AI again)">
            <RefreshIcon size={15} className={refreshing ? "animate-spin" : ""} /> Redo
          </button>
        </div>
      </header>

      {view === "json" ? (
        <pre className="max-h-[32rem] overflow-auto bg-slate-50 p-4 text-xs leading-relaxed text-slate-800">{json}</pre>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[40rem] text-left text-sm">
            <thead className="bg-slate-50/70 text-xs font-medium text-slate-500">
              <tr>
                <th className="px-4 py-2.5">Field</th>
                <th className="px-4 py-2.5">Value</th>
                <th className="px-4 py-2.5">Source</th>
                <th className="px-4 py-2.5">Quote</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((row, i) => {
                const first = i === 0 || rows[i - 1].field !== row.field;
                return (
                  <tr key={i} className={first && i > 0 ? "border-t-2 border-slate-100" : ""}>
                    <td className="whitespace-nowrap px-4 py-2.5 align-top font-medium text-slate-800">{first ? row.field : ""}</td>
                    {row.value === null ? (
                      <td colSpan={3} className="px-4 py-2.5 text-slate-400">
                        Not found in the document
                      </td>
                    ) : (
                      <>
                        <td className="px-4 py-2.5 align-top text-slate-900">{row.value}</td>
                        <td className="px-4 py-2.5 align-top">{row.citation && <CitationChip citation={row.citation} onCite={onCite} />}</td>
                        <td className="max-w-sm px-4 py-2.5 align-top text-xs italic text-slate-500">{row.quote ? `“${row.quote}”` : ""}</td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/** Pull fields (dates, parties, amounts, requirements, or your own) out of one document into a table or JSON. */
export default function ExtractTab({ documentId, onDocumentChange, onCite }: { documentId: string; onDocumentChange: (id: string) => void; onCite: (c: Citation) => void }) {
  const queryClient = useQueryClient();
  const documents = useIndexedDocuments();
  const items = documents.data?.items ?? [];
  const document = items.find((d) => d.id === documentId);
  const [presets, setPresets] = useState<string[]>(["dates", "parties", "amounts"]);
  const [custom, setCustom] = useState<string[]>([]);
  const [customInput, setCustomInput] = useState("");
  const [customError, setCustomError] = useState<string | null>(null);
  const [opened, setOpened] = useState<ExtractionContent | null>(null);
  const run = useAnalysisRun<ExtractionContent>("/analysis/extract");
  const { reset } = run;

  const saved = useQuery({
    queryKey: ["insights", documentId],
    queryFn: () => api<Insight[]>(`/documents/${documentId}/insights`),
    enabled: Boolean(documentId),
  });
  const previous = (saved.data ?? []).filter((i) => i.kind === "extraction" && i.status === "ready" && i.content);

  // Another document: whatever was shown belongs to the old one.
  useEffect(() => {
    reset();
    setOpened(null);
  }, [documentId, reset]);

  const total = presets.length + custom.length;
  const fields = [...presets, ...custom];
  const shown = run.result ?? opened;

  const addCustom = () => {
    const name = customInput.replace(/\s+/g, " ").trim();
    if (!name) return;
    if (name.length > MAX_FIELD_CHARS) return setCustomError(`A field name can have at most ${MAX_FIELD_CHARS} characters.`);
    const taken = [...custom, ...PRESETS.map((p) => p.label)].some((f) => f.toLowerCase() === name.toLowerCase());
    if (taken) return setCustomError("That field is already in the list.");
    if (total >= MAX_FIELDS) return setCustomError(`Extract at most ${MAX_FIELDS} fields at once.`);
    setCustom([...custom, name]);
    setCustomInput("");
    setCustomError(null);
  };
  const onCustomKey = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !event.nativeEvent.isComposing) {
      event.preventDefault();
      addCustom();
    }
  };

  /** `names` overrides the form (Redo repeats what is on screen, which can be a saved result's fields). */
  const extract = async (refresh: boolean, names: string[] = fields) => {
    if (!documentId || names.length === 0) return;
    setOpened(null);
    const result = await run.run({ document_id: documentId, fields: names, refresh });
    if (result) void queryClient.invalidateQueries({ queryKey: ["insights", documentId] });
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    void extract(false);
  };

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className={`${card} space-y-5 p-4 sm:p-5`}>
        {documents.isLoading ? (
          <Skeleton className="h-16" />
        ) : (
          <DocumentSelect id="extract-document" documents={items} value={documentId} onChange={onDocumentChange} disabled={run.running} />
        )}

        <fieldset disabled={run.running} className="space-y-2">
          <legend className={fieldLabel}>What to extract</legend>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => {
              const on = presets.includes(p.key);
              const blocked = !on && total >= MAX_FIELDS;
              return (
                <button
                  key={p.key}
                  type="button"
                  title={p.hint}
                  aria-pressed={on}
                  disabled={blocked}
                  onClick={() => setPresets(on ? presets.filter((k) => k !== p.key) : [...presets, p.key])}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium transition-all disabled:opacity-50 ${
                    on ? "border-indigo-300 bg-indigo-50 text-indigo-700" : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  {on && <CheckIcon size={13} strokeWidth={3} className="animate-pop" />}
                  {p.label}
                </button>
              );
            })}
            {custom.map((name) => (
              <span key={name} className="inline-flex animate-pop items-center gap-1 rounded-full border border-violet-300 bg-violet-50 py-1 pl-3 pr-1.5 text-sm font-medium text-violet-700">
                {name}
                <button type="button" onClick={() => setCustom(custom.filter((c) => c !== name))} aria-label={`Remove ${name}`} className="grid h-5 w-5 place-items-center rounded-full hover:bg-violet-200/70">
                  <XIcon size={12} />
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              value={customInput}
              onChange={(e) => {
                setCustomInput(e.target.value);
                setCustomError(null);
              }}
              onKeyDown={onCustomKey}
              maxLength={MAX_FIELD_CHARS}
              placeholder="Add your own field, e.g. Governing law"
              aria-label="Custom field name"
              className={inputBase}
            />
            <button type="button" onClick={addCustom} disabled={!customInput.trim()} className={btnSecondary}>
              <PlusIcon size={15} /> Add
            </button>
          </div>
          {customError && (
            <p role="alert" className="text-xs text-rose-600">
              {customError}
            </p>
          )}
          <p className="text-xs text-slate-400">
            {total} of {MAX_FIELDS} fields chosen
          </p>
        </fieldset>

        <div className="flex flex-wrap items-center gap-3">
          <button disabled={!documentId || fields.length === 0 || run.running} className={btnPrimary}>
            <TableIcon size={16} /> {run.running ? "Extracting…" : "Extract"}
          </button>
          {run.running && (
            <button type="button" onClick={run.cancel} className={btnSecondary}>
              Cancel
            </button>
          )}
          {!documentId && <p className="text-xs text-slate-500">Choose a document first.</p>}
        </div>
      </form>

      {run.running && (
        <div role="status" className="animate-fade-in space-y-2">
          <p className="text-sm text-slate-600">{run.message}</p>
          <ProgressBar label="Extracting" />
        </div>
      )}
      {run.error && (
        <p role="alert" className="flex animate-shake items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
          <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {run.error}
        </p>
      )}

      {shown && document && (
        <Results documentName={document.name} content={shown} cached={run.result ? run.cached : true} onRefresh={() => void extract(true, shown.fields.map((f) => f.field))} refreshing={run.running} onCite={onCite} />
      )}
      {!shown && !run.running && !run.error && documentId && (
        <p className="flex items-start gap-2 text-sm text-slate-500">
          <InfoIcon size={16} className="mt-0.5 shrink-0 text-slate-400" /> Every value comes with the page it was found on. Fields the document does not mention are shown as not found.
        </p>
      )}

      {previous.length > 0 && (
        <section aria-label="Saved extractions" className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Saved for this document</h3>
          <ul className="flex flex-wrap gap-2">
            {previous.map((insight) => {
              const content = insight.content as ExtractionContent;
              return (
                <li key={insight.id}>
                  <button
                    onClick={() => {
                      run.reset();
                      setOpened(content);
                    }}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-left text-sm shadow-xs transition hover:border-indigo-200 hover:bg-indigo-50/40"
                  >
                    <span className="block max-w-xs truncate font-medium text-slate-700">{content.fields.map((f) => f.field).join(", ")}</span>
                    <span className="block text-[11px] text-slate-400">{timeAgo(insight.updated_at)}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
