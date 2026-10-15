import type { Citation, Conflict } from "../../lib/types";
import { AlertTriangleIcon } from "../ui/icons";
import CitationChip from "./CitationChip";

/**
 * "Sources disagree": for each point on which documents contradict each other, what every document
 * says, side by side, each with the citation it comes from. `citations` is every citation of the
 * answer; a claim is matched by its number and its document (in a comparison each column numbers
 * its own citations from 1, so the number alone is not enough).
 */
export default function ConflictPanel({
  conflicts,
  citations,
  onCite,
}: {
  conflicts: Conflict[];
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  if (conflicts.length === 0) return null;
  return (
    <section aria-label="Sources disagree" className="animate-fade-up space-y-3 rounded-xl border border-amber-300/80 bg-amber-50/70 p-3.5 text-sm">
      <h3 className="flex items-center gap-2 font-semibold text-amber-900">
        <AlertTriangleIcon size={16} className="shrink-0 text-amber-500" />
        Sources disagree
        <span className="rounded-full bg-amber-200/70 px-2 py-0.5 text-[11px] font-medium text-amber-900">{conflicts.length}</span>
      </h3>
      <ul className="space-y-3">
        {conflicts.map((conflict, i) => (
          <li key={i} className="space-y-2">
            <p className="font-medium text-slate-800">{conflict.topic}</p>
            <div className={`grid gap-2 ${conflict.claims.length > 2 ? "sm:grid-cols-2 lg:grid-cols-3" : "sm:grid-cols-2"}`}>
              {conflict.claims.map((claim, j) => {
                const citation = citations.find((c) => c.n === claim.n && c.document_id === claim.document_id);
                return (
                  <div key={j} className="flex flex-col gap-2 rounded-lg border border-amber-200 bg-white p-3 shadow-xs">
                    <p className="truncate text-[11px] font-semibold uppercase tracking-wide text-slate-500" title={claim.document_name}>
                      {claim.document_name}
                    </p>
                    <p className="flex-1 leading-relaxed text-slate-800">{claim.claim}</p>
                    <div>{citation && <CitationChip citation={citation} onCite={onCite} />}</div>
                  </div>
                );
              })}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
