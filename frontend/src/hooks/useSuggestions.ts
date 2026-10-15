import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Suggestions } from "../lib/types";

/**
 * Questions to offer before the first question: from the chosen documents (written once per
 * document, in the background: polled while `pending`) and questions several people asked.
 */
export function useSuggestions(documentIds: string[], enabled = true) {
  const ids = [...documentIds].sort();
  return useQuery({
    queryKey: ["suggestions", ids],
    queryFn: () => api<Suggestions>(`/suggestions${ids.length ? `?document_ids=${ids.join(",")}` : ""}`),
    enabled,
    staleTime: 60_000,
    refetchInterval: (query) => (query.state.data?.pending.length ? 3000 : false),
  });
}
