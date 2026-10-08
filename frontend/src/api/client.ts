import type { ChatFilters, ChatHandlers, Citation, DonePayload } from "../lib/types";

const TOKEN_KEY = "grounded_token";

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string | null): void => {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fail(response: Response): Promise<never> {
  if (response.status === 401 && getToken()) {
    setToken(null);
    window.dispatchEvent(new Event("auth:logout"));
  }
  let detail = response.statusText;
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
  } catch {
    /* not JSON */
  }
  throw new ApiError(response.status, detail);
}

/** JSON request with the bearer token; a 401 logs the user out. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders(), ...(init.headers as Record<string, string>) };
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const response = await fetch(`/api${path}`, { ...init, headers });
  if (!response.ok) await fail(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** Fetch a protected file (the original document) as a Blob. */
export async function apiBlob(path: string): Promise<Blob> {
  const response = await fetch(`/api${path}`, { headers: authHeaders() });
  if (!response.ok) await fail(response);
  return response.blob();
}

/** POST /chat and read the Server-Sent Events stream (fetch, because EventSource cannot send a token). */
export async function streamChat(
  body: { question: string; conversation_id?: string; filters?: ChatFilters },
  handlers: ChatHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) await fail(response);

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatch = (name: string, data: Record<string, unknown>) => {
    if (name === "status") handlers.onStatus(String(data.stage), data);
    else if (name === "token") handlers.onToken(String(data.text));
    else if (name === "citations") handlers.onCitations(data.citations as Citation[]);
    else if (name === "done") handlers.onDone(data as unknown as DonePayload);
    else if (name === "error") handlers.onError(String(data.message), String(data.code));
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end = buffer.indexOf("\n\n");
    while (end >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let name = "";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (name && data) dispatch(name, JSON.parse(data));
      end = buffer.indexOf("\n\n");
    }
  }
}
