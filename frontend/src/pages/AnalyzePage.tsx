import { useCallback, useEffect, useState, type ComponentType } from "react";
import { useSearchParams } from "react-router-dom";
import CompareTab from "../components/analyze/CompareTab";
import ExtractTab from "../components/analyze/ExtractTab";
import GraphTab from "../components/analyze/GraphTab";
import SummariesTab from "../components/analyze/SummariesTab";
import SourceDrawer from "../components/source/SourceDrawer";
import { ColumnsIcon, GraphIcon, ListIcon, TableIcon } from "../components/ui/icons";
import { useIndexedDocuments } from "../hooks/useIndexedDocuments";
import type { Citation } from "../lib/types";

type TabId = "compare" | "extract" | "summaries" | "graph";

const TABS: { id: TabId; label: string; blurb: string; Icon: ComponentType<{ size?: number }> }[] = [
  { id: "compare", label: "Compare", blurb: "Ask one question across 2 to 4 documents and read each document's answer side by side, with the points where they disagree.", Icon: ColumnsIcon },
  { id: "extract", label: "Extract", blurb: "Pull dates, parties, amounts, requirements or your own fields out of a document into a table you can download.", Icon: TableIcon },
  { id: "summaries", label: "Summaries", blurb: "An executive summary, chapter summaries, FAQ, glossary and key changes of a document, each with citations.", Icon: ListIcon },
  { id: "graph", label: "Graph", blurb: "The people, policies, courses and concepts your documents talk about, and how they relate, including across documents. Every entity links back to the passages it comes from.", Icon: GraphIcon },
];

const isTab = (value: string | null): value is TabId => TABS.some((t) => t.id === value);

/** Compare, Extract and Summaries. The tab and the chosen document live in the address, so a link opens the same view. */
export default function AnalyzePage() {
  const [params, setParams] = useSearchParams();
  const tab: TabId = isTab(params.get("tab")) ? (params.get("tab") as TabId) : "compare";
  const documentId = params.get("doc") ?? "";
  const [drawer, setDrawer] = useState<Citation | null>(null);
  const documents = useIndexedDocuments();

  const update = useCallback(
    (change: Record<string, string>) =>
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          Object.entries(change).forEach(([key, value]) => (value ? next.set(key, value) : next.delete(key)));
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );

  // A link to a document that was deleted (or is not searchable) must not leave a dead choice behind.
  useEffect(() => {
    if (documentId && documents.data && !documents.data.items.some((d) => d.id === documentId)) update({ doc: "" });
  }, [documentId, documents.data, update]);

  const active = TABS.find((t) => t.id === tab) ?? TABS[0];
  return (
    <div className="relative h-full">
      <div className="h-full overflow-y-auto bg-linear-to-b from-slate-50 to-white">
        <div className="mx-auto max-w-7xl space-y-6 px-4 py-6">
          <div className="space-y-3">
            <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Analyze</h1>
            <div role="tablist" aria-label="Analysis" className="flex w-full gap-1 overflow-x-auto rounded-xl bg-slate-100 p-1 sm:w-fit">
              {TABS.map(({ id, label, Icon }) => (
                <button
                  key={id}
                  role="tab"
                  id={`tab-${id}`}
                  aria-selected={tab === id}
                  aria-controls={`panel-${id}`}
                  onClick={() => update({ tab: id })}
                  className={`flex min-w-fit flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium transition-all sm:flex-none sm:px-4 ${
                    tab === id ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Icon size={16} /> {label}
                </button>
              ))}
            </div>
            <p className="max-w-3xl text-sm text-slate-500">{active.blurb}</p>
          </div>

          {/* All four stay mounted, so a comparison or an extraction in progress survives a look at another tab. */}
          <div role="tabpanel" id="panel-compare" aria-labelledby="tab-compare" hidden={tab !== "compare"}>
            <CompareTab onCite={setDrawer} />
          </div>
          <div role="tabpanel" id="panel-extract" aria-labelledby="tab-extract" hidden={tab !== "extract"}>
            <ExtractTab documentId={documentId} onDocumentChange={(id) => update({ doc: id })} onCite={setDrawer} />
          </div>
          <div role="tabpanel" id="panel-summaries" aria-labelledby="tab-summaries" hidden={tab !== "summaries"}>
            <SummariesTab documentId={documentId} onDocumentChange={(id) => update({ doc: id })} onCite={setDrawer} />
          </div>
          <div role="tabpanel" id="panel-graph" aria-labelledby="tab-graph" hidden={tab !== "graph"}>
            <GraphTab documentId={documentId} onCite={setDrawer} />
          </div>
        </div>
      </div>
      <SourceDrawer citation={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}
