import { apiBlob } from "../api/client";

/** Download the original document (the endpoint needs the bearer token, so a plain link cannot be used). */
export async function downloadOriginal(documentId: string, name: string): Promise<void> {
  const blob = await apiBlob(`/documents/${documentId}/file?download=true`);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export const isPdf = (name: string): boolean => name.toLowerCase().endsWith(".pdf");

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function pageLabel(start: number, end: number): string {
  return end > start ? `${start}-${end}` : `${start}`;
}
