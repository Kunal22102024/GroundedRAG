import type { ReactNode } from "react";
import { pageLabel } from "../../lib/files";
import type { Citation } from "../../lib/types";

const INLINE = /(\[\d+\]|\*\*[^*]+\*\*|`[^`]+`)/g;

function inline(text: string, citations: Citation[], onCite: (c: Citation) => void, keyBase: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyBase}-${i}`;
    const cite = /^\[(\d+)\]$/.exec(part);
    if (cite) {
      const citation = citations.find((c) => c.n === Number(cite[1]));
      if (!citation) return null; // never show a marker that maps to no source
      const name = citation.document_name.length > 26 ? `${citation.document_name.slice(0, 24)}…` : citation.document_name;
      return (
        <button key={key} type="button" className="cite-chip" title={`${citation.document_name} • page ${pageLabel(citation.page_number, citation.page_end)}`} onClick={() => onCite(citation)}>
          {name} • p{pageLabel(citation.page_number, citation.page_end)}
        </button>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={key} className="rounded bg-slate-100 px-1 text-sm">{part.slice(1, -1)}</code>;
    return part;
  });
}

/** Plain-text answer with paragraphs, bullet lists, bold, code and clickable citation chips. */
export default function Answer({ text, citations, onCite }: { text: string; citations: Citation[]; onCite: (c: Citation) => void }) {
  const blocks: ReactNode[] = [];
  let items: string[] = [];
  const flush = (key: number) => {
    if (!items.length) return;
    const list = items;
    items = [];
    blocks.push(
      <ul key={`ul-${key}`} className="ml-5 list-disc space-y-1">
        {list.map((item, i) => (
          <li key={i}>{inline(item, citations, onCite, `li-${key}-${i}`)}</li>
        ))}
      </ul>,
    );
  };
  text.split("\n").forEach((line, index) => {
    const bullet = /^\s*(?:[-*]|\d+\.)\s+(.*)$/.exec(line);
    if (bullet) {
      items.push(bullet[1]);
      return;
    }
    flush(index);
    if (line.trim()) blocks.push(<p key={`p-${index}`}>{inline(line, citations, onCite, `p-${index}`)}</p>);
  });
  flush(text.length);
  return <div className="space-y-2 leading-relaxed">{blocks}</div>;
}
