import { useCallback, useEffect, useRef, useState } from "react";
import { streamSse } from "../api/client";
import type { AnalysisDone } from "../lib/types";

export interface RunState<T> {
  running: boolean;
  /** What the server is doing now, in words (empty when idle). */
  message: string;
  result: T | null;
  /** True when the result was saved earlier and no new model call was made. */
  cached: boolean;
  error: string | null;
}

const IDLE = { running: false, message: "", result: null, cached: false, error: null } as const;

/** The sentence for a `status` event of a summary or an extraction. */
export function describeStatus(data: Record<string, unknown>): string {
  if (data.stage === "rate_limited") return "The AI service is busy, retrying shortly…";
  if (typeof data.detail === "string" && data.detail) return `${data.detail}…`;
  return "Working…";
}

/**
 * Runs one streamed analysis (`POST /api/<path>`) at a time and follows it: progress text while it
 * runs, then the result or the error. Starting another run, calling `cancel` or leaving the page
 * ends the one in progress, and a run that is no longer the current one never touches the state.
 */
export function useAnalysisRun<T>(path: string) {
  const [state, setState] = useState<RunState<T>>(IDLE);
  const controller = useRef<AbortController | null>(null);
  const current = useRef(0);

  const cancel = useCallback(() => {
    current.current += 1;
    controller.current?.abort();
    controller.current = null;
    setState((s) => (s.running ? { ...s, running: false, message: "" } : s));
  }, []);

  const reset = useCallback(() => {
    current.current += 1;
    controller.current?.abort();
    controller.current = null;
    setState(IDLE);
  }, []);

  useEffect(
    () => () => {
      current.current += 1;
      controller.current?.abort();
    },
    [],
  );

  const run = useCallback(
    async (body: unknown): Promise<T | null> => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      const id = ++current.current;
      const mine = () => current.current === id;
      setState({ running: true, message: "Starting…", result: null, cached: false, error: null });

      let result: T | null = null;
      try {
        await streamSse(
          path,
          body,
          (name, data) => {
            if (!mine()) return;
            if (name === "status") setState((s) => ({ ...s, message: describeStatus(data) }));
            else if (name === "done") {
              const done = data as unknown as AnalysisDone<T>;
              result = done.content;
              setState({ running: false, message: "", result: done.content, cached: Boolean(done.cached), error: null });
            } else if (name === "error") {
              setState({ running: false, message: "", result: null, cached: false, error: String(data.message) });
            }
          },
          abort.signal,
        );
      } catch (e) {
        if (mine() && !abort.signal.aborted) {
          setState({ running: false, message: "", result: null, cached: false, error: e instanceof Error ? e.message : "Request failed" });
        }
      }
      return mine() ? result : null;
    },
    [path],
  );

  return { ...state, run, cancel, reset };
}
