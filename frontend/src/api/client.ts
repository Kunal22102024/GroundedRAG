import type { ChatFilters, ChatHandlers, Citation, Conflict, DonePayload, FollowUp, QuestionSource, UploadResult } from "../lib/types";

const TOKEN_KEY = "grounded_token";

// Storage can be unavailable (private mode, blocked site data): the app must still load, just
// without remembering the sign-in.
export const getToken = (): string | null => {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
};
export const setToken = (token: string | null): void => {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* not persisted */
  }
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    /** The server's raw `detail`, for callers that need more than the sentence (e.g. per-file upload reasons). */
    public detail?: unknown,
  ) {
    super(message);
  }
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** What the person sees when the server gave no readable explanation. */
function fallbackMessage(status: number, statusText: string): string {
  if (status === 413) return "That is too large to upload. Try fewer or smaller files.";
  if (status === 429) return "Too many requests. Please wait a moment and try again.";
  if (status === 502 || status === 503 || status === 504) return "The server is not reachable right now. Please try again in a moment.";
  return statusText || `Request failed (${status})`;
}

/** Turn FastAPI's `detail` (a string, a validation-error list or an upload report) into a sentence. */
export function describeDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const lines = detail
      .map((item) => {
        if (typeof item !== "object" || item === null) return null;
        const { msg, loc } = item as { msg?: unknown; loc?: unknown };
        if (typeof msg !== "string") return null;
        const field = Array.isArray(loc) ? loc.filter((p) => typeof p === "string" && !["body", "query", "path"].includes(p)).join(".") : "";
        const text = msg.replace(/^Value error, /, "");
        return field ? `${field}: ${text}` : text;
      })
      .filter((line): line is string => line !== null);
    return lines.length ? lines.join("; ") : null;
  }
  if (typeof detail === "object" && detail !== null) {
    const { message, rejected } = detail as { message?: unknown; rejected?: unknown };
    const head = typeof message === "string" ? message : null;
    const files = Array.isArray(rejected)
      ? rejected
          .map((r) => (typeof r === "object" && r !== null ? `${(r as { filename?: string }).filename ?? "file"}: ${(r as { reason?: string }).reason ?? "not accepted"}` : null))
          .filter((line): line is string => line !== null)
      : [];
    if (head || files.length) return [head, ...files].filter(Boolean).join(" ");
  }
  return null;
}

async function fail(response: Response): Promise<never> {
  if (response.status === 401 && getToken()) {
    setToken(null);
    window.dispatchEvent(new Event("auth:logout"));
  }
  let message = fallbackMessage(response.status, response.statusText);
  let raw: unknown;
  try {
    const body: unknown = await response.json();
    raw = typeof body === "object" && body !== null ? (body as { detail?: unknown }).detail : undefined;
    const described = describeDetail(raw);
    if (described) message = described;
  } catch {
    /* not JSON */
  }
  throw new ApiError(response.status, message, raw);
}

export interface UploadProgress {
  loaded: number;
  total: number;
}

/**
 * POST /documents with upload progress. `fetch` cannot report how much of a request body has been
 * sent, so this uses XMLHttpRequest; errors are turned into the same `ApiError` as every other call.
 */
export function uploadDocuments(form: FormData, onProgress: (progress: UploadProgress) => void, signal?: AbortSignal): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents");
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress({ loaded: event.loaded, total: event.total });
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResult);
        } catch {
          reject(new ApiError(xhr.status, "The server sent a response that could not be read."));
        }
        return;
      }
      fail(new Response(xhr.responseText, { status: xhr.status, statusText: xhr.statusText })).catch(reject);
    };
    xhr.onerror = () => reject(new ApiError(0, "The upload failed: the server could not be reached."));
    xhr.onabort = () => reject(new DOMException("The upload was cancelled.", "AbortError"));
    if (signal) {
      if (signal.aborted) return reject(new DOMException("The upload was cancelled.", "AbortError"));
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(form);
  });
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

/** One Server-Sent Events block: its `event` name and its `data` lines joined with newlines. */
export function parseSseBlock(block: string): { name: string; data: string } | null {
  let name = "";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // blank, or a comment used as a keep-alive
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") name = value;
    else if (field === "data") data.push(value);
  }
  return name && data.length ? { name, data: data.join("\n") } : null;
}

/**
 * POST `path` (under /api) and read the Server-Sent Events stream (fetch, because EventSource cannot
 * send a token or a POST body). `onEvent` gets every event in order. Resolves only after a `done` or
 * `error` event; a stream that simply ends before one is an error, so the caller is never left
 * waiting on a result that will not come.
 */
export async function streamSse(
  path: string,
  body: unknown,
  onEvent: (name: string, data: Record<string, unknown>) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) await fail(response);

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  const handleBlock = (block: string) => {
    const event = parseSseBlock(block);
    if (!event) return;
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(event.data) as Record<string, unknown>;
    } catch {
      throw new ApiError(0, "The server sent an answer that could not be read. Please try again.");
    }
    if (event.name === "done" || event.name === "error") finished = true;
    onEvent(event.name, data);
  };
  const drain = () => {
    let end = buffer.indexOf("\n\n");
    while (end >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      handleBlock(block);
      end = buffer.indexOf("\n\n");
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // A "\r\n" split between two reads stays a lone "\r" until the next read completes it.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      drain();
    }
    buffer = (buffer + decoder.decode()).replace(/\r\n/g, "\n");
    drain();
    if (buffer.trim()) handleBlock(buffer); // a final event the server did not terminate with a blank line
  } finally {
    // Free the connection whatever happened: a parse error or an abort must not leave it open.
    void reader.cancel().catch(() => undefined);
  }
  if (!finished) throw new ApiError(0, "The connection closed before the answer finished. Please try again.");
}

/** POST /chat: the question, and the answer as it streams in. */
export function streamChat(
  body: { question: string; conversation_id?: string; filters?: ChatFilters; source?: QuestionSource },
  handlers: ChatHandlers,
  signal: AbortSignal,
): Promise<void> {
  return streamSse(
    "/chat",
    body,
    (name, data) => {
      if (name === "status") handlers.onStatus(String(data.stage), data);
      else if (name === "token") handlers.onToken(String(data.text));
      else if (name === "citations") handlers.onCitations(data.citations as Citation[]);
      else if (name === "conflicts") handlers.onConflicts(data.conflicts as Conflict[]);
      else if (name === "suggestions") handlers.onSuggestions(data.suggestions as FollowUp[]);
      else if (name === "done") handlers.onDone(data as unknown as DonePayload);
      else if (name === "error") handlers.onError(String(data.message), String(data.code));
    },
    signal,
  );
}
