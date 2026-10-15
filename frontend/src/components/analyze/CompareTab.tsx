import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { streamSse } from "../../api/client";
import { useIndexedDocuments } from "../../hooks/useIndexedDocuments";
import type { Citation, CompareColumn, Conflict, DocumentItem } from "../../lib/types";
import { btnPrimary, btnSecondary, card, fieldLabel, inputBase } from "../../lib/ui";
import Answer from "../chat/Answer";
import ConflictPanel from "../chat/ConflictPanel";
import { AlertCircleIcon, CheckCircleIcon, ColumnsIcon, InfoIcon, StopIcon } from "../ui/icons";
import { FileTypeBadge, Skeleton, Spinner } from "../ui/primitives";
import { DocumentMultiPicker } from "./DocumentPicker";

const MAX_DOCUMENTS = 4;
const MAX_QUESTION = 2000;

type Stage = "waiting" | "searching" | "generating" | "rate_limited" | "retrying" | "done" | "stopped";

interface ColumnState {
  documentId: string;
  name: string;
  stage: Stage;
  text: string;
  covered: boolean;
  citations: Citation[];
  flags: CompareColumn["flags"];
}

const STAGE_TEXT: Record<Stage, string> = {
  waiting: "Waiting for its turn",
  searching: "Searching this document",
  generating: "Writing the answer",
  rate_limited: "The AI service is busy, retrying shortly",
  retrying: "Double-checking the citations",
  done: "",
  stopped: "",
};

function Column({ column, onCite }: { column: ColumnState; onCite: (c: Citation) => void }) {
  const working = column.stage !== "done" && column.stage !== "stopped";
  return (
    <article className={`${card} flex min-w-0 animate-fade-up flex-col`} aria-busy={working} aria-label={`Answer from ${column.name}`}>
      <header className="flex items-center gap-2.5 border-b border-slate-100 p-3">
        <FileTypeBadge name={column.name} size="sm" />
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-slate-900" title={column.name}>
          {column.name}
        </h3>
        {column.stage === "done" && column.covered && <CheckCircleIcon size={16} className="shrink-0 text-emerald-500" />}
      </header>
      <div className="flex-1 space-y-2 p-3.5 text-[15px] text-slate-800">
        {column.stage === "done" && !column.covered ? (
          <div className="flex gap-2.5 rounded-lg bg-slate-50 p-3 text-sm text-slate-600">
            <InfoIcon size={17} className="mt-0.5 shrink-0 text-slate-400" />
            <div>
              <p className="font-medium text-slate-700">Not covered in this document</p>
              <p className="mt-0.5 text-slate-500">It does not contain enough information to answer this question.</p>
            </div>
          </div>
        ) : column.stage === "stopped" ? (
          <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-500">This answer was not finished.</p>
        ) : column.text ? (
          <Answer text={column.text} citations={column.citations} onCite={onCite} streaming={column.stage === "generating"} />
        ) : (
          <div className="space-y-2" aria-hidden="true">
            <Skeleton className="h-3.5 w-11/12" />
            <Skeleton className="h-3.5 w-4/5" />
            <Skeleton className="h-3.5 w-2/3" />
          </div>
        )}
        {working && (
          <p role="status" className="flex items-center gap-2 text-xs text-slate-500">
            <Spinner size={12} className="text-indigo-500" /> {STAGE_TEXT[column.stage]}…
          </p>
        )}
        {column.stage === "done" && column.covered && (column.flags.partial || column.flags.citations_removed || column.flags.uncited) && (
          <p className="text-xs text-amber-700">Some citations could not be verified, or the answer was cut short.</p>
        )}
      </div>
    </article>
  );
}

/** One question asked of each chosen document on its own: an answer per document, then where they disagree. */
export default function CompareTab({ onCite }: { onCite: (c: Citation) => void }) {
  const documents = useIndexedDocuments();
  const items: DocumentItem[] = documents.data?.items ?? [];
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [columns, setColumns] = useState<ColumnState[]>([]);
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [checking, setChecking] = useState(false);
  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const current = useRef(0);
  const results = useRef<HTMLElement>(null);
  const [runs, setRuns] = useState(0);

  useEffect(
    () => () => {
      current.current += 1;
      controller.current?.abort();
    },
    [],
  );
  // The form fills most of the screen, so bring the answers into view when a comparison starts.
  useEffect(() => {
    if (runs > 0) results.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [runs]);
  // A document that was deleted or is no longer searchable cannot stay chosen.
  useEffect(() => {
    if (!documents.data) return;
    const known = new Set(documents.data.items.map((d) => d.id));
    setSelected((ids) => (ids.every((id) => known.has(id)) ? ids : ids.filter((id) => known.has(id))));
  }, [documents.data]);

  const canRun = selected.length >= 2 && question.trim().length > 0 && !running;

  const start = async () => {
    if (!canRun) return;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const id = ++current.current;
    const mine = () => current.current === id;
    const patchColumn = (documentId: string, change: Partial<ColumnState> | ((c: ColumnState) => Partial<ColumnState>)) =>
      setColumns((all) => all.map((c) => (c.documentId === documentId ? { ...c, ...(typeof change === "function" ? change(c) : change) } : c)));

    // A column still waiting or writing when the run ends early is incomplete: say so rather than
    // leave a spinner or a half answer that looks final.
    const settle = () => setColumns((all) => all.map((c) => (c.stage === "done" || c.stage === "stopped" ? c : { ...c, stage: "stopped", text: "", citations: [] })));

    const names = new Map(items.map((d) => [d.id, d.name]));
    setColumns(selected.map((documentId) => ({ documentId, name: names.get(documentId) ?? "Document", stage: "waiting", text: "", covered: false, citations: [], flags: {} })));
    setConflicts([]);
    setChecking(false);
    setError(null);
    setFinished(false);
    setRunning(true);
    setRuns((n) => n + 1);

    try {
      await streamSse(
        "/analysis/compare",
        { question: question.trim(), document_ids: selected },
        (name, data) => {
          if (!mine()) return;
          if (name === "status") {
            if (typeof data.document_id === "string") {
              patchColumn(data.document_id, (c) => ({ stage: data.stage as Stage, text: data.reset ? "" : c.text }));
            } else if (data.stage === "checking_sources") setChecking(true);
          } else if (name === "token") {
            patchColumn(String(data.document_id), (c) => ({ text: c.text + String(data.text) }));
          } else if (name === "column") {
            const column = data as unknown as CompareColumn;
            patchColumn(column.document_id, { stage: "done", text: column.answer, covered: column.covered, citations: column.citations, flags: column.flags });
          } else if (name === "conflicts") {
            setConflicts(data.conflicts as Conflict[]);
            setChecking(false);
          } else if (name === "done") {
            setRunning(false);
            setChecking(false);
            setFinished(true);
          } else if (name === "error") {
            setRunning(false);
            setChecking(false);
            settle();
            setError(String(data.message));
          }
        },
        abort.signal,
      );
    } catch (e) {
      if (!mine()) return;
      setRunning(false);
      setChecking(false);
      settle();
      if (!abort.signal.aborted) setError(e instanceof Error ? e.message : "Request failed");
    }
  };

  const stop = () => {
    current.current += 1;
    controller.current?.abort();
    setRunning(false);
    setChecking(false);
    setColumns((all) => all.map((c) => (c.stage === "done" || c.stage === "stopped" ? c : { ...c, stage: "stopped", text: "", citations: [] })));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void start();
  };
  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void start();
    }
  };

  const answered = columns.filter((c) => c.stage === "done" && c.covered).length;
  const grid = columns.length >= 4 ? "md:grid-cols-2 xl:grid-cols-4" : columns.length === 3 ? "md:grid-cols-2 lg:grid-cols-3" : "md:grid-cols-2";

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className={`${card} space-y-4 p-4 sm:p-5`}>
        {documents.isLoading ? (
          <Skeleton className="h-24" />
        ) : items.length < 2 ? (
          <p className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-sm text-slate-600">
            <InfoIcon size={17} className="mt-0.5 shrink-0 text-slate-400" />
            Comparing needs at least two searchable documents. Upload more in the Library.
          </p>
        ) : (
          <DocumentMultiPicker documents={items} selected={selected} onChange={setSelected} max={MAX_DOCUMENTS} disabled={running} />
        )}
        <div>
          <label htmlFor="compare-question" className={fieldLabel}>
            Question
          </label>
          <textarea
            id="compare-question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKey}
            maxLength={MAX_QUESTION}
            rows={2}
            disabled={running}
            placeholder="e.g. How many days of paid leave do employees get?"
            className={`${inputBase} resize-y`}
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {running ? (
            <button type="button" onClick={stop} className={btnSecondary}>
              <StopIcon size={14} /> Stop
            </button>
          ) : (
            <button disabled={!canRun} className={btnPrimary}>
              <ColumnsIcon size={16} /> Compare
            </button>
          )}
          <p className="text-xs text-slate-500">
            Each document answers on its own, from its own text only.
            {selected.length < 2 && items.length >= 2 ? " Choose at least two." : ""}
          </p>
        </div>
      </form>

      {error && (
        <p role="alert" className="flex animate-shake items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
          <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {error}
        </p>
      )}

      {columns.length > 0 && (
        <section ref={results} aria-label="Answers" aria-live="polite" className={`scroll-mt-4 grid gap-4 ${grid}`}>
          {columns.map((column) => (
            <Column key={column.documentId} column={column} onCite={onCite} />
          ))}
        </section>
      )}

      {checking && (
        <p role="status" className="flex items-center gap-2 text-sm text-slate-500">
          <Spinner size={14} className="text-indigo-500" /> Checking the sources against each other…
        </p>
      )}
      {conflicts.length > 0 && <ConflictPanel conflicts={conflicts} citations={columns.flatMap((c) => c.citations)} onCite={onCite} />}
      {finished && conflicts.length === 0 && (
        <p className="flex animate-fade-up items-start gap-2 rounded-xl border border-emerald-200 bg-emerald-50/70 p-3.5 text-sm text-emerald-800">
          {answered >= 2 ? (
            <>
              <CheckCircleIcon size={17} className="mt-0.5 shrink-0 text-emerald-500" /> No conflicts found: the documents do not contradict each other on this question.
            </>
          ) : (
            <>
              <InfoIcon size={17} className="mt-0.5 shrink-0 text-emerald-600" /> Conflicts can only be checked when at least two documents answer the question.
            </>
          )}
        </p>
      )}
    </div>
  );
}
