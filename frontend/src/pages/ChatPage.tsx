import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, api, streamChat } from "../api/client";
import Answer from "../components/chat/Answer";
import FilterBar from "../components/chat/FilterBar";
import SourceDrawer from "../components/source/SourceDrawer";
import {
  REFUSAL_MESSAGE,
  type ChatFilters,
  type Citation,
  type Conversation,
  type ConversationDetail,
  type DonePayload,
  type Message,
} from "../lib/types";

interface StreamState {
  question: string;
  text: string;
  status: string;
  citations: Citation[];
  error: string | null;
  done: DonePayload | null;
}

const STATUS_TEXT: Record<string, string> = {
  searching: "Searching the documents…",
  rewriting: "Understanding your follow-up…",
  generating: "Writing the answer…",
  rate_limited: "The AI service is busy, retrying shortly…",
  retrying: "Double-checking the citations…",
};

function RefusalNotice() {
  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
      <b>Not found in the selected documents.</b>
      <p>{REFUSAL_MESSAGE}</p>
    </div>
  );
}

function FeedbackButtons({ message }: { message: Message }) {
  const [rating, setRating] = useState(message.feedback);
  const rate = useMutation({
    mutationFn: (value: "up" | "down") => api<void>(`/messages/${message.id}/feedback`, { method: "POST", body: JSON.stringify({ rating: value }) }),
    onSuccess: (_, value) => setRating(value),
  });
  const button = (value: "up" | "down", label: string) => (
    <button
      onClick={() => rate.mutate(value)}
      aria-label={value === "up" ? "Good answer" : "Bad answer"}
      className={`rounded border px-2 py-0.5 text-sm ${rating === value ? "border-indigo-500 bg-indigo-50" : "hover:bg-slate-100"}`}
    >
      {label}
    </button>
  );
  return <div className="flex gap-1">{button("up", "👍")}{button("down", "👎")}</div>;
}

function AssistantBubble({ message, onCite }: { message: Message; onCite: (c: Citation) => void }) {
  const refusal = message.content === REFUSAL_MESSAGE;
  return (
    <div className="max-w-3xl space-y-2 rounded-xl border bg-white p-4">
      {refusal ? <RefusalNotice /> : <Answer text={message.content} citations={message.citations ?? []} onCite={onCite} />}
      {message.flagged && <p className="text-xs text-amber-700">Some citations could not be verified, or the answer was cut short.</p>}
      <FeedbackButtons message={message} />
    </div>
  );
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<ChatFilters>({});
  const [input, setInput] = useState("");
  const [stream, setStream] = useState<StreamState | null>(null);
  const [drawer, setDrawer] = useState<Citation | null>(null);
  const controller = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const conversations = useQuery({ queryKey: ["conversations"], queryFn: () => api<Conversation[]>("/conversations") });
  const detail = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api<ConversationDetail>(`/conversations/${conversationId}`),
    enabled: Boolean(conversationId),
  });
  const messages = conversationId ? (detail.data?.messages ?? []) : [];

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, stream?.text, stream?.status]);
  useEffect(() => {
    setDrawer(null);
    setStream(null);
  }, [conversationId]);
  useEffect(() => () => controller.current?.abort(), []);

  const remove = useMutation({
    mutationFn: (id: string) => api<void>(`/conversations/${id}`, { method: "DELETE" }),
    onSuccess: (_, id) => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      if (id === conversationId) navigate("/chat");
    },
  });

  const update = (patch: Partial<StreamState>) => setStream((s) => (s ? { ...s, ...patch } : s));

  const send = async (question: string) => {
    const abort = new AbortController();
    controller.current = abort;
    setInput("");
    setStream({ question, text: "", status: "searching", citations: [], error: null, done: null });
    const clean: ChatFilters = Object.fromEntries(
      Object.entries(filters).filter(([, v]) => (Array.isArray(v) ? v.length > 0 : Boolean(v))),
    );
    try {
      await streamChat(
        { question, conversation_id: conversationId, filters: Object.keys(clean).length ? clean : undefined },
        {
          onStatus: (stage, data) => setStream((s) => (s ? { ...s, status: stage, text: data.reset ? "" : s.text } : s)),
          onToken: (t) => setStream((s) => (s ? { ...s, text: s.text + t } : s)),
          onCitations: (citations) => update({ citations }),
          onDone: (done) => {
            update({ done, text: done.answer, status: "done" });
            void (async () => {
              await queryClient.invalidateQueries({ queryKey: ["conversations"] });
              await queryClient.fetchQuery({
                queryKey: ["conversation", done.conversation_id],
                queryFn: () => api<ConversationDetail>(`/conversations/${done.conversation_id}`),
                staleTime: 0,
              });
              if (!conversationId) navigate(`/chat/${done.conversation_id}`);
              else setStream(null);
            })();
          },
          onError: (message) => update({ error: message, status: "error" }),
        },
        abort.signal,
      );
    } catch (e) {
      if (abort.signal.aborted) return;
      update({ error: e instanceof ApiError || e instanceof Error ? e.message : "Request failed", status: "error" });
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (question && !(stream && !stream.error && stream.status !== "done")) void send(question);
  };
  const busy = stream !== null && !stream.error && stream.status !== "done";

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 flex-col border-r bg-white md:flex">
        <Link to="/chat" className="m-3 rounded bg-indigo-600 py-2 text-center text-sm font-medium text-white hover:bg-indigo-700">
          + New chat
        </Link>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {conversations.data?.map((c) => (
            <div key={c.id} className={`group flex items-center rounded ${c.id === conversationId ? "bg-indigo-50" : "hover:bg-slate-100"}`}>
              <Link to={`/chat/${c.id}`} className="min-w-0 flex-1 truncate px-2 py-2 text-sm" title={c.title}>
                {c.title}
              </Link>
              <button
                onClick={() => window.confirm("Delete this conversation?") && remove.mutate(c.id)}
                aria-label="Delete conversation"
                className="hidden px-2 text-slate-400 hover:text-red-600 group-hover:block"
              >
                ✕
              </button>
            </div>
          ))}
          {conversations.data?.length === 0 && <p className="px-2 py-3 text-sm text-slate-500">No conversations yet.</p>}
        </div>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col">
        <FilterBar filters={filters} onChange={setFilters} />
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && !stream && (
            <div className="mx-auto mt-16 max-w-md text-center text-slate-500">
              <h2 className="text-lg font-semibold text-slate-700">Ask about your documents</h2>
              <p className="mt-1 text-sm">Answers come only from the uploaded documents and cite their sources. Upload files in the Library first.</p>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="ml-auto max-w-2xl rounded-xl bg-indigo-600 px-4 py-2 text-white">
                {m.content}
              </div>
            ) : (
              <AssistantBubble key={m.id} message={m} onCite={setDrawer} />
            ),
          )}
          {stream && (
            <>
              <div className="ml-auto max-w-2xl rounded-xl bg-indigo-600 px-4 py-2 text-white">{stream.question}</div>
              <div className="max-w-3xl space-y-2 rounded-xl border bg-white p-4">
                {stream.error ? (
                  <p className="rounded bg-red-50 p-2 text-sm text-red-700">{stream.error}</p>
                ) : stream.text ? (
                  stream.text === REFUSAL_MESSAGE ? <RefusalNotice /> : <Answer text={stream.text} citations={stream.citations} onCite={setDrawer} />
                ) : null}
                {busy && <p className="animate-pulse text-sm text-slate-500">{STATUS_TEXT[stream.status] ?? "Working…"}</p>}
              </div>
            </>
          )}
          <div ref={bottom} />
        </div>
        <form onSubmit={submit} className="flex gap-2 border-t bg-white p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={2000}
            placeholder="Ask a question about your documents…"
            className="min-w-0 flex-1 rounded border px-3 py-2"
          />
          {busy ? (
            <button type="button" onClick={() => controller.current?.abort()} className="rounded border px-4 py-2 text-sm">
              Stop
            </button>
          ) : (
            <button disabled={!input.trim()} className="rounded bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
              Send
            </button>
          )}
        </form>
        <SourceDrawer citation={drawer} onClose={() => setDrawer(null)} />
      </section>
    </div>
  );
}
