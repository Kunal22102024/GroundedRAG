import type { ReactNode } from "react";
import type { Citation } from "../../lib/types";
import CitationChip from "./CitationChip";

// Citation markers, bold and inline code. Bold and code can contain markers, which `inline` resolves
// by calling itself on their text.
const INLINE = /(\[\d+\]|\*\*[^*]+\*\*|`[^`]+`)/g;
const BULLET = /^\s*[-*]\s+(.*)$/;
const NUMBERED = /^\s*\d{1,3}[.)]\s+(.*)$/;

function inline(text: string, citations: Citation[], onCite: (c: Citation) => void, keyBase: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyBase}-${i}`;
    const cite = /^\[(\d+)\]$/.exec(part);
    if (cite) {
      const citation = citations.find((c) => c.n === Number(cite[1]));
      if (!citation) return null; // never show a marker that maps to no source
      return <CitationChip key={key} citation={citation} onCite={onCite} />;
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={key}>{inline(part.slice(2, -2), citations, onCite, key)}</strong>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={key} className="rounded bg-slate-100 px-1 text-sm">{part.slice(1, -1)}</code>;
    return part;
  });
}

/**
 * Plain-text answer with paragraphs, bullet and numbered lists, bold, code and clickable citation chips.
 * While `streaming`, a blinking caret follows the last line.
 */
export default function Answer({
  text,
  citations,
  onCite,
  streaming = false,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
  streaming?: boolean;
}) {
  const lines = text.split("\n");
  let lastLine = -1;
  lines.forEach((line, index) => {
    if (line.trim()) lastLine = index;
  });
  const caret = (index: number) => (streaming && index === lastLine ? <span className="stream-caret" aria-hidden="true" /> : null);

  const blocks: ReactNode[] = [];
  let items: { text: string; line: number }[] = [];
  let ordered = false;
  const flush = (key: number) => {
    if (!items.length) return;
    const list = items;
    items = [];
    const rendered = list.map((item, i) => (
      <li key={i}>
        {inline(item.text, citations, onCite, `li-${key}-${i}`)}
        {caret(item.line)}
      </li>
    ));
    blocks.push(
      ordered ? (
        <ol key={`ol-${key}`} className="ml-5 list-decimal space-y-1">
          {rendered}
        </ol>
      ) : (
        <ul key={`ul-${key}`} className="ml-5 list-disc space-y-1">
          {rendered}
        </ul>
      ),
    );
  };
  lines.forEach((line, index) => {
    const bullet = BULLET.exec(line);
    const numbered = bullet ? null : NUMBERED.exec(line);
    const match = bullet ?? numbered;
    if (match) {
      const isOrdered = numbered !== null;
      if (items.length && ordered !== isOrdered) flush(index); // a bullet list followed by a numbered one
      ordered = isOrdered;
      items.push({ text: match[1], line: index });
      return;
    }
    flush(index);
    if (line.trim())
      blocks.push(
        <p key={`p-${index}`}>
          {inline(line, citations, onCite, `p-${index}`)}
          {caret(index)}
        </p>,
      );
  });
  flush(text.length);
  return <div className="space-y-2.5 leading-relaxed">{blocks}</div>;
}
