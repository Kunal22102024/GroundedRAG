import { apiBlob } from "../api/client";

/** Download the original document (the endpoint needs the bearer token, so a plain link cannot be used). */
export async function downloadOriginal(documentId: string, name: string): Promise<void> {
  const blob = await apiBlob(`/documents/${documentId}/file?download=true`);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link); // Firefox ignores a click on an anchor that is not in the page
  link.click();
  link.remove();
  // Revoking at once can cancel the download in Firefox and Safari; give the browser time to start it.
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

export const isPdf = (name: string): boolean => name.toLowerCase().endsWith(".pdf");

/** "Policy.PDF" -> "pdf" ("" when there is none). */
export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function pageLabel(start: number, end: number): string {
  return end > start ? `${start}-${end}` : `${start}`;
}
