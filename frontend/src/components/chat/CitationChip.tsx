import { pageLabel } from "../../lib/files";
import type { Citation } from "../../lib/types";

/** A clickable "document • page" chip for one citation; a source that is gone says so instead. */
export default function CitationChip({ citation, onCite }: { citation: Citation; onCite: (c: Citation) => void }) {
  if (citation.unavailable) {
    return (
      <span className="cite-chip cite-chip-unavailable" title="This source is no longer available">
        [{citation.n}] source no longer available
      </span>
    );
  }
  const name = citation.document_name.length > 26 ? `${citation.document_name.slice(0, 24)}…` : citation.document_name;
  return (
    <button type="button" className="cite-chip" title={`${citation.document_name} • page ${pageLabel(citation.page_number, citation.page_end)}`} onClick={() => onCite(citation)}>
      {name} • p{pageLabel(citation.page_number, citation.page_end)}
    </button>
  );
}
