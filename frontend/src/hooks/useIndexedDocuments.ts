import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { DocumentList } from "../lib/types";

/** Documents that are searchable (the only ones that can be analysed). Shared with the chat page. */
export function useIndexedDocuments() {
  return useQuery({
    queryKey: ["documents", "indexed"],
    queryFn: () => api<DocumentList>("/documents?status=indexed&limit=200"),
  });
}
