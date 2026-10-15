import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import type { EntityDetail, GraphView } from "../lib/types";

function params(values: Record<string, string>): string {
  const query = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => value && query.set(key, value));
  const text = query.toString();
  return text ? `?${text}` : "";
}

/**
 * The knowledge graph of the chosen documents (none chosen: every document the person may read).
 * A graph is built in the background after upload, so the query is repeated while any is pending.
 */
export function useKnowledgeGraph(documentIds: string[], types: string[], search: string) {
  const ids = [...documentIds].sort();
  const kinds = [...types].sort();
  const text = search.trim();
  return useQuery({
    queryKey: ["graph", ids, kinds, text],
    queryFn: () => api<GraphView>(`/graph${params({ document_ids: ids.join(","), types: kinds.join(","), q: text })}`),
    placeholderData: keepPreviousData, // keep drawing the old picture while a filter loads
    staleTime: 30_000,
    refetchInterval: (query) => (query.state.data?.documents.some((d) => d.status === "pending") ? 3000 : false),
  });
}

/** Where one entity is mentioned and how it is related; the passages open in the source drawer. */
export function useEntityDetail(key: string | null, documentIds: string[]) {
  const ids = [...documentIds].sort();
  return useQuery({
    queryKey: ["graph", "entity", key, ids],
    queryFn: () => api<EntityDetail>(`/graph/entity${params({ key: key ?? "", document_ids: ids.join(",") })}`),
    enabled: Boolean(key),
    staleTime: 30_000,
  });
}

/** Queue (re)building the graph of a document, then follow it. */
export function useBuildGraph() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => api<{ status: "pending" }>(`/documents/${documentId}/graph`, { method: "POST" }),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["graph"] }),
  });
}
