import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { ApiError } from "../../api/client";
import { useIndexedDocuments } from "../../hooks/useIndexedDocuments";
import { useBuildGraph, useEntityDetail, useKnowledgeGraph } from "../../hooks/useKnowledgeGraph";
import { pageLabel } from "../../lib/files";
import type { Citation, EntityRelation, GraphDocument, GraphNode, GraphStatus } from "../../lib/types";
import { btnGhost, btnPrimary, btnSecondary, card, fieldLabel, inputBase } from "../../lib/ui";
import { ENTITY_STYLES, ENTITY_TYPE_ORDER, entityStyle, relationText } from "../graph/entityTypes";
import ForceGraph, { type SelectedEdge } from "../graph/ForceGraph";
import { AlertCircleIcon, CheckIcon, InfoIcon, RefreshIcon, SearchIcon, XIcon } from "../ui/icons";
import { ProgressBar, Skeleton, Spinner } from "../ui/primitives";
import { useToast } from "../ui/Toast";

const STATUS_STYLE: Record<GraphStatus, { label: string; pill: string }> = {
  none: { label: "Not built", pill: "bg-slate-100 text-slate-600 ring-slate-200" },
  pending: { label: "Reading…", pill: "bg-amber-50 text-amber-700 ring-amber-200" },
  ready: { label: "Ready", pill: "bg-emerald-50 text-emerald-700 ring-emerald-200" },
  failed: { label: "Failed", pill: "bg-rose-50 text-rose-700 ring-rose-200" },
};

function GraphStatusPill({ status }: { status: GraphStatus }) {
  const s = STATUS_STYLE[status];
  return <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${s.pill}`}>{status === "pending" && <Spinner size={11} />}{s.label}</span>;
}

function passageLabel(c: Citation): string {
  return `${c.document_name} · p. ${pageLabel(c.page_number, c.page_end)}${c.section_title ? ` · ${c.section_title}` : ""}`;
}

function snippet(text: string, around: string): string {
  const flat = text.replace(/\s+/g, " ");
  const at = flat.toLowerCase().indexOf(around.toLowerCase());
  const start = Math.max(0, at - 60);
  return `${start > 0 ? "…" : ""}${flat.slice(start, start + 170)}${start + 170 < flat.length ? "…" : ""}`;
}

/** A passage the graph links to: opens in the source drawer. */
function PassageButton({ citation, around, onCite }: { citation: Citation; around: string; onCite: (c: Citation) => void }) {
  return (
    <button type="button" onClick={() => onCite(citation)} className="block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-left transition hover:border-indigo-300 hover:bg-indigo-50/40">
      <span className="block truncate text-xs font-medium text-indigo-700">{passageLabel(citation)}</span>
      <span className="mt-0.5 block text-xs text-slate-600">{snippet(citation.text, around)}</span>
    </button>
  );
}

function RelationRow({ relation, subject, chosen, onOpen, onCite }: { relation: EntityRelation; subject: string; chosen: boolean; onOpen: (id: string) => void; onCite: (c: Citation) => void }) {
  const out = relation.direction === "out";
  const other = (
    <button type="button" onClick={() => onOpen(relation.other.id)} className="inline-flex items-center gap-1.5 rounded-md bg-slate-100 px-1.5 py-0.5 text-sm font-medium text-slate-800 hover:bg-indigo-50 hover:text-indigo-700">
      <span className={`h-2 w-2 rounded-full ${entityStyle(relation.other.type).dot}`} aria-hidden="true" />
      {relation.other.name}
    </button>
  );
  return (
    <li className={`space-y-1.5 rounded-lg border p-2.5 ${chosen ? "border-indigo-300 bg-indigo-50/60" : "border-slate-200"}`}>
      <p className="flex flex-wrap items-center gap-1.5 text-sm text-slate-700">
        {out ? <span className="font-medium">{subject}</span> : other}
        <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-xs font-medium text-indigo-700">{relationText(relation.label)} →</span>
        {out ? other : <span className="font-medium">{subject}</span>}
      </p>
      {relation.quote && <p className="border-l-2 border-slate-200 pl-2 text-xs italic text-slate-600">“{relation.quote}”</p>}
      <button type="button" onClick={() => onCite(relation.evidence)} className="text-xs font-medium text-indigo-700 hover:underline">
        Open the source · {passageLabel(relation.evidence)}
      </button>
    </li>
  );
}

function EntityPanel({ id, documentIds, edge, onCite, onOpen, onClose }: { id: string; documentIds: string[]; edge: SelectedEdge | null; onCite: (c: Citation) => void; onOpen: (id: string) => void; onClose: () => void }) {
  const detail = useEntityDetail(id, documentIds);
  const data = detail.data;
  return (
    <div className="space-y-4" aria-live="polite">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          {data && (
            <>
              <h3 className="break-words text-base font-semibold text-slate-900">{data.name}</h3>
              <p className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-500">
                <span className={`h-2 w-2 rounded-full ${entityStyle(data.type).dot}`} aria-hidden="true" /> {entityStyle(data.type).label}
                <span aria-hidden="true">·</span> in {data.documents.length} document{data.documents.length === 1 ? "" : "s"}
              </p>
            </>
          )}
        </div>
        <button type="button" onClick={onClose} aria-label="Close the details" className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
          <XIcon size={16} />
        </button>
      </div>
      {detail.isLoading && <Skeleton className="h-32" />}
      {detail.isError && (
        <p role="alert" className="flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">
          <AlertCircleIcon size={16} className="mt-0.5 shrink-0" /> {detail.error instanceof ApiError && detail.error.status === 404 ? "This entity is no longer in the graph." : "Could not load the details."}
        </p>
      )}
      {data && (
        <>
          {data.description && <p className="text-sm text-slate-700">{data.description}</p>}
          {data.documents.length > 1 && <p className="text-xs text-slate-500">Found in {data.documents.map((d) => d.name).join(", ")}: this is what connects them.</p>}
          {data.relations.length > 0 && (
            <section aria-label="Relations" className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Relations</h4>
              <ul className="space-y-2">
                {data.relations.map((r, i) => (
                  <RelationRow
                    key={`${r.direction}-${r.label}-${r.other.id}-${r.evidence.chunk_id}-${i}`}
                    relation={r}
                    subject={data.name}
                    chosen={Boolean(edge) && ((r.direction === "out" && edge?.source === id && edge.target === r.other.id) || (r.direction === "in" && edge?.target === id && edge.source === r.other.id)) && edge?.label === r.label}
                    onOpen={onOpen}
                    onCite={onCite}
                  />
                ))}
              </ul>
            </section>
          )}
          <section aria-label="Mentions" className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Mentioned in</h4>
            {data.mentions.length === 0 ? <p className="text-sm text-slate-500">No passages are available.</p> : data.mentions.map((m) => <PassageButton key={m.chunk_id} citation={m} around={data.name} onCite={onCite} />)}
          </section>
        </>
      )}
    </div>
  );
}

function EntityList({ nodes, onOpen }: { nodes: GraphNode[]; onOpen: (id: string) => void }) {
  const ranked = useMemo(() => [...nodes].sort((a, b) => b.degree - a.degree || b.mentions - a.mentions || a.name.localeCompare(b.name)), [nodes]);
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-900">Entities</h3>
      <p className="text-xs text-slate-500">Select a circle or a line, or an entity here, to see where it comes from. Drag to move things around, scroll to zoom.</p>
      <ul className="max-h-[26rem] space-y-1 overflow-y-auto pr-1" aria-label="Entities">
        {ranked.map((n) => (
          <li key={n.id}>
            <button type="button" onClick={() => onOpen(n.id)} className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-slate-700 hover:bg-slate-100">
              <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${entityStyle(n.type).dot}`} aria-hidden="true" />
              <span className="min-w-0 flex-1 truncate">{n.name}</span>
              <span className="shrink-0 text-xs tabular-nums text-slate-400" title="Connections">{n.degree}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DocumentRow({ doc, onBuild, busy }: { doc: GraphDocument; onBuild: (id: string) => void; busy: boolean }) {
  const partial = doc.status === "ready" && doc.coverage !== null && doc.coverage < 1;
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2">
      <span className="min-w-0 flex-1 truncate text-sm text-slate-800" title={doc.name}>{doc.name}</span>
      <GraphStatusPill status={doc.status} />
      {doc.status === "ready" && (
        <span className="text-xs text-slate-500">
          {doc.entities} entities · {doc.relations} relations{partial ? ` · read ${Math.round((doc.coverage ?? 0) * 100)}% of it` : ""}
        </span>
      )}
      {doc.status === "failed" && doc.error && <span className="max-w-xs truncate text-xs text-rose-600" title={doc.error}>{doc.error}</span>}
      {doc.status !== "pending" && (
        <button type="button" disabled={busy} onClick={() => onBuild(doc.id)} className={`${btnSecondary} !px-2.5 !py-1 text-xs`}>
          {doc.status === "none" ? "Build" : doc.status === "failed" ? "Try again" : (<><RefreshIcon size={12} /> Rebuild</>)}
        </button>
      )}
    </li>
  );
}

/** The knowledge graph of the chosen documents: entities and how they relate, each linked back to the text it was read from. */
export default function GraphTab({ documentId, onCite }: { documentId: string; onCite: (c: Citation) => void }) {
  const { toast } = useToast();
  const documents = useIndexedDocuments();
  const [selected, setSelected] = useState<string[]>(documentId ? [documentId] : []);
  const [types, setTypes] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [focus, setFocus] = useState<string | null>(null);
  const [edge, setEdge] = useState<SelectedEdge | null>(null);
  const [showAll, setShowAll] = useState(false);

  // A document chosen on another tab (?doc=) is the starting point here, once.
  useEffect(() => {
    if (documentId) setSelected((current) => (current.length ? current : [documentId]));
  }, [documentId]);

  const items = useMemo(() => documents.data?.items ?? [], [documents.data]);
  const graph = useKnowledgeGraph(selected, types, deferredSearch);
  const build = useBuildGraph();
  const view = graph.data;
  const statuses = useMemo(() => new Map((view?.documents ?? []).map((d) => [d.id, d.status])), [view]);
  const needBuild = (view?.documents ?? []).filter((d) => d.status === "none" || d.status === "failed");
  const pending = (view?.documents ?? []).filter((d) => d.status === "pending");
  const hasGraph = (view?.documents ?? []).some((d) => d.status === "ready");

  // A chosen entity that the filters or a deleted document removed is let go.
  useEffect(() => {
    if (focus && view && !view.nodes.some((n) => n.id === focus) && !graph.isPlaceholderData) {
      setFocus(null);
      setEdge(null);
    }
  }, [focus, view, graph.isPlaceholderData]);

  const toggleDocument = (id: string) => {
    setFocus(null);
    setEdge(null);
    setSelected((current) => (current.includes(id) ? current.filter((d) => d !== id) : [...current, id]));
  };
  const toggleType = (type: string) => setTypes((current) => (current.includes(type) ? current.filter((t) => t !== type) : [...current, type]));
  const open = (id: string) => {
    setFocus(id);
    setEdge(null);
  };

  const startBuild = async (ids: string[]) => {
    let failed = 0;
    for (const id of ids) {
      try {
        await build.mutateAsync(id);
      } catch {
        failed += 1;
      }
    }
    if (failed) toast({ tone: "error", title: failed === ids.length ? "Could not start the build" : `${failed} of ${ids.length} builds could not be started` });
    else toast({ tone: "success", title: ids.length === 1 ? "Reading the document…" : `Reading ${ids.length} documents…` });
  };

  const rows = showAll ? view?.documents ?? [] : (view?.documents ?? []).filter((d) => d.status !== "ready");
  const nothingToShow = Boolean(view) && view!.nodes.length === 0;

  return (
    <div className="space-y-6">
      <div className={`${card} space-y-4 p-4 sm:p-5`}>
        <div className="flex items-center justify-between gap-3">
          <span className={`${fieldLabel} mb-0`}>
            Documents <span className="text-slate-400">({selected.length ? `${selected.length} chosen` : "all you can read"})</span>
          </span>
          {selected.length > 0 && (
            <button type="button" onClick={() => setSelected([])} className={`${btnGhost} !py-1 text-xs`}>
              Use all documents
            </button>
          )}
        </div>
        {documents.isLoading ? (
          <Skeleton className="h-10" />
        ) : items.length === 0 ? (
          <p className="text-sm text-slate-500">No document is searchable yet. Upload one in the Library and wait until it is ready.</p>
        ) : (
          <ul className="flex max-h-40 flex-wrap gap-2 overflow-y-auto p-0.5 pr-2" aria-label="Documents">
            {items.map((d) => {
              const on = selected.includes(d.id);
              const status = statuses.get(d.id);
              return (
                <li key={d.id}>
                  <button type="button" onClick={() => toggleDocument(d.id)} aria-pressed={on} title={d.name} className={`flex max-w-[16rem] items-center gap-2 rounded-full border px-3 py-1 text-sm transition ${on ? "border-indigo-300 bg-indigo-50 text-indigo-800 ring-2 ring-indigo-500/10" : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"}`}>
                    <span className={`h-2 w-2 shrink-0 rounded-full ${status === "ready" ? "bg-emerald-500" : status === "pending" ? "animate-pulse bg-amber-400" : status === "failed" ? "bg-rose-500" : "bg-slate-300"}`} aria-hidden="true" />
                    <span className="truncate">{d.name}</span>
                    {on && <CheckIcon size={13} className="shrink-0" />}
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        {view && view.documents.length > 0 && (
          <div className="space-y-2 border-t border-slate-100 pt-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm text-slate-600">
                {hasGraph ? `${view.documents.filter((d) => d.status === "ready").length} of ${view.documents.length} documents have a graph.` : "No graph has been built for these documents yet."}
                {pending.length > 0 && ` ${pending.length} being read.`}
              </p>
              <div className="flex items-center gap-2">
                {view.documents.some((d) => d.status === "ready") && (
                  <button type="button" onClick={() => setShowAll((v) => !v)} className={`${btnGhost} !py-1 text-xs`}>
                    {showAll ? "Hide finished" : "Show all"}
                  </button>
                )}
                {needBuild.length > 0 && (
                  <button type="button" disabled={build.isPending} onClick={() => void startBuild(needBuild.map((d) => d.id))} className={btnPrimary}>
                    {build.isPending && <Spinner size={14} />} Build {needBuild.length === 1 ? "the graph" : `${needBuild.length} graphs`}
                  </button>
                )}
              </div>
            </div>
            {pending.length > 0 && <ProgressBar label="Reading documents to build their graphs" />}
            {rows.length > 0 && (
              <ul className="max-h-56 divide-y divide-slate-100 overflow-y-auto pr-2" aria-label="Graph status of each document">
                {rows.map((d) => (
                  <DocumentRow key={d.id} doc={d} busy={build.isPending} onBuild={(id) => void startBuild([id])} />
                ))}
              </ul>
            )}
            <p className="flex items-start gap-1.5 text-xs text-slate-500">
              <InfoIcon size={13} className="mt-0.5 shrink-0" />
              A graph is built by reading the document with the AI model (about one call for every 30,000 characters), once. Every entity and relation is checked against the text it was read from, and the graph is deleted with the document.
            </p>
          </div>
        )}
      </div>

      {graph.isError && (
        <p role="alert" className="flex items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
          <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {graph.error instanceof ApiError && graph.error.status === 404 ? "The knowledge graph is switched off, or a chosen document is no longer available." : "Could not load the graph."}
        </p>
      )}
      {graph.isLoading && <Skeleton className="h-96" />}

      {view && (hasGraph || types.length > 0 || search) && (
        <div className={`${card} p-4 sm:p-5`}>
          <div className="mb-3 flex flex-wrap items-end gap-3">
            <div className="relative min-w-[12rem] flex-1 sm:max-w-xs">
              <label htmlFor="graph-search" className={fieldLabel}>Find an entity</label>
              <SearchIcon size={15} className="pointer-events-none absolute bottom-2.5 left-3 text-slate-400" />
              <input id="graph-search" value={search} onChange={(e) => setSearch(e.target.value)} maxLength={100} placeholder="Name contains…" className={`${inputBase} pl-9`} />
            </div>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Show only these kinds of entity">
              {ENTITY_TYPE_ORDER.map((type) => (
                <button key={type} type="button" onClick={() => toggleType(type)} aria-pressed={types.includes(type)} className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition ${types.includes(type) ? "border-indigo-300 bg-indigo-50 text-indigo-800" : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"}`}>
                  <span className={`h-2 w-2 rounded-full ${ENTITY_STYLES[type].dot}`} aria-hidden="true" /> {ENTITY_STYLES[type].label}
                </button>
              ))}
            </div>
          </div>

          {nothingToShow ? (
            <p className="py-10 text-center text-sm text-slate-500">{types.length || search ? "No entity matches these filters." : "These documents have no entities yet."}</p>
          ) : (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_21rem]">
              <div className="min-w-0 space-y-2">
                <ForceGraph nodes={view.nodes} edges={view.edges} selectedId={focus} selectedEdge={edge} onSelectNode={(id) => (setFocus(id), setEdge(null))} onSelectEdge={(e) => (setFocus(e.source), setEdge(e))} />
                <p className="text-xs text-slate-500">
                  {view.nodes.length} of {view.total_nodes} entities{view.truncated ? " (the best connected are shown: narrow it with the filters)" : ""} · {view.edges.length} connection{view.edges.length === 1 ? "" : "s"}
                  {graph.isFetching && " · updating…"}
                </p>
              </div>
              <aside className="min-w-0 rounded-xl border border-slate-200 bg-slate-50/50 p-3 lg:max-h-[40rem] lg:overflow-y-auto">
                {focus ? <EntityPanel id={focus} documentIds={selected} edge={edge} onCite={onCite} onOpen={open} onClose={() => (setFocus(null), setEdge(null))} /> : <EntityList nodes={view.nodes} onOpen={open} />}
              </aside>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
