// Turning analysis results into text that leaves the app (copy, Markdown, JSON, CSV).

import { pageLabel } from "./files";
import type { Citation, ExtractionContent, SummaryContent } from "./types";

/** "…25 days [1]." -> "…25 days (leave.md, p.3)." Markers of a source that is gone are dropped. */
export function plainCitations(text: string, citations: Citation[]): string {
  return text.replace(/\s*\[(\d+)\]/g, (_marker, n: string) => {
    const citation = citations.find((c) => c.n === Number(n));
    if (!citation || citation.unavailable) return "";
    return ` (${citation.document_name}, p.${pageLabel(citation.page_number, citation.page_end)})`;
  });
}

export const SUMMARY_TITLES: Record<SummaryContent["kind"], string> = {
  executive: "Executive summary",
  chapters: "Chapter-by-chapter summary",
  faq: "FAQ",
  glossary: "Glossary",
  key_changes: "Key changes",
};

/** A summary as Markdown, with citations written out as "(document, page)". */
export function summaryMarkdown(documentName: string, content: SummaryContent): string {
  const cite = (text: string) => plainCitations(text, content.citations);
  const head = `# ${SUMMARY_TITLES[content.kind]}: ${documentName}\n\n`;
  switch (content.kind) {
    case "executive":
      return head + cite(content.text) + "\n";
    case "chapters":
      return head + content.items.map((i) => `## ${i.title}\n\n${cite(i.summary)}\n`).join("\n");
    case "faq":
      return head + content.items.map((i) => `**${i.question}**\n\n${cite(i.answer)}\n`).join("\n");
    case "glossary":
      return head + content.items.map((i) => `- **${i.term}**: ${cite(i.definition)}`).join("\n") + "\n";
    case "key_changes":
      return head + content.items.map((i) => `- **${cite(i.change)}**\n  Impact: ${cite(i.impact)}`).join("\n") + "\n";
  }
}

// --- Extraction ------------------------------------------------------------------------------------

export interface ExtractionRow {
  field: string;
  /** Null for a field the document says nothing about. */
  value: string | null;
  citation: Citation | null;
  quote: string;
}

/** One row per extracted value; a field with no value still gets a row so nothing silently vanishes. */
export function extractionRows(content: ExtractionContent): ExtractionRow[] {
  const rows: ExtractionRow[] = [];
  for (const field of content.fields) {
    if (field.values.length === 0) rows.push({ field: field.field, value: null, citation: null, quote: "" });
    for (const v of field.values) {
      rows.push({ field: field.field, value: v.value, citation: content.citations.find((c) => c.n === v.n) ?? null, quote: v.quote });
    }
  }
  return rows;
}

const location = (c: Citation | null) => ({
  page: c && !c.unavailable ? pageLabel(c.page_number, c.page_end) : "",
  section: c && !c.unavailable ? c.section_title : "",
});

export function extractionJson(documentName: string, content: ExtractionContent): string {
  const out = {
    document: documentName,
    fields: content.fields.map((f) => ({
      field: f.field,
      values: f.values.map((v) => {
        const citation = content.citations.find((c) => c.n === v.n);
        const known = citation !== undefined && !citation.unavailable;
        return {
          value: v.value,
          page_start: known ? citation.page_number : null,
          page_end: known ? citation.page_end : null,
          section: known ? citation.section_title : "",
          quote: v.quote,
        };
      }),
    })),
  };
  return JSON.stringify(out, null, 2);
}

/**
 * A CSV cell. Values come from documents, so one that starts with a character a spreadsheet reads as
 * a formula (= + - @) is prefixed with an apostrophe, which keeps it text when the file is opened.
 */
export function csvCell(raw: string): string {
  const safe = /^[=+\-@\t\r]/.test(raw) ? `'${raw}` : raw;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

export function extractionCsv(documentName: string, content: ExtractionContent): string {
  const lines = [["Document", "Field", "Value", "Page", "Section", "Quote"].join(",")];
  for (const row of extractionRows(content)) {
    const where = location(row.citation);
    lines.push([documentName, row.field, row.value ?? "", where.page, where.section, row.quote].map(csvCell).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}
