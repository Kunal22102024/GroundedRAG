import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { api } from "../../api/client";
import { useAnalysisRun } from "../../hooks/useAnalysisRun";
import { useIndexedDocuments } from "../../hooks/useIndexedDocuments";
import { SUMMARY_TITLES, summaryMarkdown } from "../../lib/analysis";
import { timeAgo } from "../../lib/format";
import type { Citation, DocumentItem, Insight, SummaryContent, SummaryKind } from "../../lib/types";
import { btnGhost, btnPrimary, card, inputBase } from "../../lib/ui";
import Answer from "../chat/Answer";
import { AlertCircleIcon, BookOpenIcon, CheckIcon, ChevronRightIcon, CopyIcon, FileTextIcon, LayersIcon, ListIcon, RefreshIcon, SearchIcon, SparklesIcon } from "../ui/icons";
import { ProgressBar, Skeleton, Spinner } from "../ui/primitives";
import { useToast } from "../ui/Toast";
import { DocumentSelect } from "./DocumentPicker";

/** A result nobody is writing any more (the server treats a pending row older than this as dead). */
const PENDING_FRESH_MS = 10 * 60 * 1000;
const freshPending = (insight: Insight | undefined) => insight?.status === "pending" && Date.now() - Date.parse(insight.updated_at) < PENDING_FRESH_MS;

const CARDS: { kind: SummaryKind; description: string; icon: ReactNode }[] = [
  { kind: "executive", description: "The point of the document in a few paragraphs.", icon: <SparklesIcon size={18} /> },
  { kind: "chapters", description: "What each section says, in document order.", icon: <LayersIcon size={18} /> },
  { kind: "faq", description: "Questions a reader is likely to ask, answered from the text.", icon: <FileTextIcon size={18} /> },
  { kind: "glossary", description: "Terms the document defines or relies on.", icon: <BookOpenIcon size={18} /> },
  { kind: "key_changes", description: "What is new or changed, or how it differs from an earlier version.", icon: <ListIcon size={18} /> },
];

function Item({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <details className="group rounded-xl border border-slate-200 bg-white open:shadow-sm">
      <summary className="flex cursor-pointer list-none items-start gap-2 rounded-xl px-3.5 py-3 text-sm font-medium text-slate-800 marker:hidden hover:bg-slate-50 [&::-webkit-details-marker]:hidden">
        <ChevronRightIcon size={16} className="mt-0.5 shrink-0 text-slate-400 transition-transform group-open:rotate-90" />
        <span className="min-w-0 flex-1">{title}</span>
      </summary>
      <div className="border-t border-slate-100 px-3.5 py-3 pl-9 text-sm text-slate-700">{children}</div>
    </details>
  );
}

function Glossary({ items, citations, onCite }: { items: { term: string; definition: string }[]; citations: Citation[]; onCite: (c: Citation) => void }) {
  const [search, setSearch] = useState("");
  const q = search.trim().toLowerCase();
  const shown = items.filter((i) => !q || i.term.toLowerCase().includes(q) || i.definition.toLowerCase().includes(q));
  return (
    <div className="space-y-3">
      <div className="relative">
        <SearchIcon size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder={`Search ${items.length} terms`} aria-label="Search the glossary" className={`${inputBase} pl-9`} />
      </div>
      <div className="overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50/70 text-xs font-medium text-slate-500">
            <tr>
              <th className="w-1/3 px-4 py-2.5">Term</th>
              <th className="px-4 py-2.5">Definition</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {shown.map((item) => (
              <tr key={item.term}>
                <td className="px-4 py-2.5 align-top font-medium text-slate-900">{item.term}</td>
                <td className="px-4 py-2.5 align-top text-slate-700">
                  <Answer text={item.definition} citations={citations} onCite={onCite} />
                </td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={2} className="px-4 py-4 text-slate-500">
                  Nothing matches "{search}".
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Body({ content, onCite }: { content: SummaryContent; onCite: (c: Citation) => void }) {
  const { citations } = content;
  switch (content.kind) {
    case "executive":
      return (
        <div className="text-[15px] text-slate-800">
          <Answer text={content.text} citations={citations} onCite={onCite} />
        </div>
      );
    case "chapters":
      return (
        <div className="space-y-2">
          {content.items.map((item, i) => (
            <Item key={i} title={item.title}>
              <Answer text={item.summary} citations={citations} onCite={onCite} />
            </Item>
          ))}
        </div>
      );
    case "faq":
      return (
        <div className="space-y-2">
          {content.items.map((item, i) => (
            <Item key={i} title={item.question}>
              <Answer text={item.answer} citations={citations} onCite={onCite} />
            </Item>
          ))}
        </div>
      );
    case "glossary":
      return <Glossary items={content.items} citations={citations} onCite={onCite} />;
    case "key_changes":
      return (
        <ul className="space-y-3">
          {content.items.map((item, i) => (
            <li key={i} className="rounded-xl border border-slate-200 bg-white p-3.5 text-sm shadow-xs">
              <div className="font-medium text-slate-900">
                <Answer text={item.change} citations={citations} onCite={onCite} />
              </div>
              <div className="mt-2 border-t border-slate-100 pt-2 text-slate-600">
                <span className="mr-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Impact</span>
                <Answer text={item.impact} citations={citations} onCite={onCite} />
              </div>
            </li>
          ))}
        </ul>
      );
  }
}

function SummaryCard({
  kind,
  description,
  icon,
  document,
  others,
  insights,
  onChanged,
  onCite,
}: {
  kind: SummaryKind;
  description: string;
  icon: ReactNode;
  document: DocumentItem;
  others: DocumentItem[];
  insights: Insight[];
  onChanged: () => void;
  onCite: (c: Citation) => void;
}) {
  const { toast } = useToast();
  const run = useAnalysisRun<SummaryContent>("/analysis/summary");
  const [compareTo, setCompareTo] = useState("");
  const [copied, setCopied] = useState(false);

  // Key changes are saved per earlier version; every other kind has one saved result.
  const key = kind === "key_changes" ? compareTo : "";
  const saved = insights.find((i) => i.kind === kind && i.params_key === key);
  const content = run.result ?? (saved?.status === "ready" ? (saved.content as SummaryContent | null) : null);
  const writing = run.running || freshPending(saved);
  const failedMessage = run.error ?? (saved?.status === "failed" && !content ? (saved.error_message ?? "This could not be generated.") : null);

  const generate = async (refresh: boolean) => {
    const body: Record<string, unknown> = { document_id: document.id, kind, refresh };
    if (kind === "key_changes" && compareTo) body.compare_to = compareTo;
    await run.run(body);
    onChanged(); // a failure is saved too, so the saved results change either way
  };

  const copy = () => {
    if (!content) return;
    navigator.clipboard
      .writeText(summaryMarkdown(document.name, content))
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => toast({ tone: "error", title: "Could not copy" }));
  };

  return (
    <section className={`${card} animate-fade-up overflow-hidden`} aria-label={SUMMARY_TITLES[kind]} aria-busy={writing}>
      <header className="flex flex-wrap items-start gap-3 p-4 sm:px-5">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-indigo-50 text-indigo-600">{icon}</span>
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-slate-900">{SUMMARY_TITLES[kind]}</h3>
          <p className="text-sm text-slate-500">{description}</p>
          {content && saved && !run.result && <p className="mt-0.5 text-xs text-slate-400">Saved {timeAgo(saved.updated_at)}</p>}
        </div>
        <div className="flex items-center gap-1">
          {content && (
            <button onClick={copy} className={btnGhost} title="Copy as Markdown">
              {copied ? <CheckIcon size={15} className="animate-pop text-emerald-600" /> : <CopyIcon size={15} />} Copy
            </button>
          )}
          {content ? (
            <button onClick={() => void generate(true)} disabled={writing} className={btnGhost} title="Write it again (asks the AI again)">
              <RefreshIcon size={15} className={run.running ? "animate-spin" : ""} /> Regenerate
            </button>
          ) : (
            <button onClick={() => void generate(false)} disabled={writing} className={btnPrimary}>
              {writing ? <Spinner size={14} /> : <SparklesIcon size={15} />} {writing ? "Writing…" : failedMessage ? "Try again" : "Generate"}
            </button>
          )}
        </div>
      </header>

      {kind === "key_changes" && (
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 bg-slate-50/50 px-4 py-2.5 sm:px-5">
          <label htmlFor="compare-to" className="text-xs font-medium text-slate-500">
            Compare with an earlier version
          </label>
          <select id="compare-to" value={compareTo} onChange={(e) => setCompareTo(e.target.value)} disabled={writing} className={`${inputBase} max-w-xs py-1.5`}>
            <option value="">None: what this document announces</option>
            {others.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {(writing || content || failedMessage) && (
        <div className="space-y-3 border-t border-slate-100 p-4 sm:px-5">
          {writing && (
            <div role="status" className="space-y-2">
              <p className="text-sm text-slate-600">{run.running ? run.message : "Being written…"}</p>
              <ProgressBar label={`Writing the ${SUMMARY_TITLES[kind].toLowerCase()}`} />
              {!content && <Skeleton className="h-16" />}
            </div>
          )}
          {failedMessage && !writing && (
            <p role="alert" className="flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">
              <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {failedMessage}
            </p>
          )}
          {content && !run.running && <Body content={content} onCite={onCite} />}
        </div>
      )}
    </section>
  );
}

/** Executive summary, chapters, FAQ, glossary and key changes of one document, each saved once written. */
export default function SummariesTab({ documentId, onDocumentChange, onCite }: { documentId: string; onDocumentChange: (id: string) => void; onCite: (c: Citation) => void }) {
  const queryClient = useQueryClient();
  const documents = useIndexedDocuments();
  const items = useMemo(() => documents.data?.items ?? [], [documents.data]);
  const document = items.find((d) => d.id === documentId);

  const insights = useQuery({
    queryKey: ["insights", documentId],
    queryFn: () => api<Insight[]>(`/documents/${documentId}/insights`),
    enabled: Boolean(documentId),
    // The executive summary of a new document is written in the background: follow it until it lands.
    refetchInterval: (query) => (query.state.data?.some((i) => freshPending(i)) ? 3000 : false),
  });
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["insights", documentId] });
    void queryClient.invalidateQueries({ queryKey: ["documents"] }); // the Library's "Summary ready"
  };

  return (
    <div className="space-y-6">
      <div className={`${card} p-4 sm:p-5`}>
        {documents.isLoading ? (
          <Skeleton className="h-14" />
        ) : (
          <DocumentSelect id="summary-document" documents={items} value={documentId} onChange={onDocumentChange} />
        )}
      </div>

      {!documentId && <p className="text-sm text-slate-500">Choose a document to see or write its summaries. The executive summary is written automatically after a document is indexed; the others are written when you ask.</p>}

      {document && insights.isLoading && <Skeleton className="h-40" />}
      {document && insights.isError && (
        <p role="alert" className="flex items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
          <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> Could not load the saved summaries. You can still generate them.
        </p>
      )}
      {document && !insights.isLoading && (
        <div className="space-y-4">
          {CARDS.map((c) => (
            <SummaryCard
              key={`${document.id}-${c.kind}`}
              {...c}
              document={document}
              others={items.filter((d) => d.id !== document.id)}
              insights={insights.data ?? []}
              onChanged={refresh}
              onCite={onCite}
            />
          ))}
        </div>
      )}
    </div>
  );
}
