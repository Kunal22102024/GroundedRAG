import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, streamChat } from "../api/client";
import Answer from "../components/chat/Answer";
import FilterBar from "../components/chat/FilterBar";
import SourceDrawer from "../components/source/SourceDrawer";
import {
  AlertCircleIcon,
  AlertTriangleIcon,
  BookOpenIcon,
  ChatIcon,
  CheckIcon,
  CopyIcon,
  FilterIcon,
  LogoMark,
  MenuIcon,
  PlusIcon,
  SearchIcon,
  SendIcon,
  SparklesIcon,
  StopIcon,
  ThumbDownIcon,
  ThumbUpIcon,
  TrashIcon,
  XIcon,
} from "../components/ui/icons";
import { Skeleton } from "../components/ui/primitives";
import { useToast } from "../components/ui/Toast";
import { timeAgo } from "../lib/format";
import {
  REFUSAL_MESSAGE,
  type ChatFilters,
  type Citation,
  type Conversation,
  type ConversationDetail,
  type DocumentList,
  type DonePayload,
  type Message,
} from "../lib/types";
import { btnSecondary, iconBtn } from "../lib/ui";

interface StreamState {
  question: string;
  text: string;
  status: string;
  citations: Citation[];
  error: string | null;
  done: DonePayload | null;
}

const STATUS_TEXT: Record<string, string> = {
  searching: "Searching your documents",
  rewriting: "Understanding your follow-up",
  generating: "Writing the answer",
  rate_limited: "The AI service is busy, retrying shortly",
  retrying: "Double-checking the citations",
  saving: "Saving the conversation",
};

/** States in which nothing more will arrive for the question being shown. */
const FINISHED = new Set(["done", "stopped", "error"]);
const MAX_QUESTION = 2000;

const SUGGESTIONS = [
  { title: "Summarize a policy", prompt: "Summarize the main rules of the leave policy." },
  { title: "Look up an error code", prompt: "What does error ERR_404_NET mean and what should I check first?" },
  { title: "Find a requirement", prompt: "What is the minimum recommended length for a work password?" },
  { title: "Compare figures", prompt: "Which region reported the highest quarterly revenue?" },
];

function AssistantAvatar() {
  return (
    <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-linear-to-br from-indigo-500 to-violet-500 text-white shadow-sm shadow-indigo-500/30">
      <SparklesIcon size={15} />
    </span>
  );
}

function TypingIndicator({ label }: { label: string }) {
  return (
    <div role="status" className="flex items-center gap-2 text-sm text-slate-500">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span key={i} className="h-1.5 w-1.5 animate-typing rounded-full bg-indigo-400" style={{ animationDelay: `${i * 0.15}s` }} />
        ))}
      </span>
      {label}…
    </div>
  );
}

function RefusalNotice() {
  return (
    <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50/70 p-3.5 text-sm text-amber-900">
      <AlertCircleIcon size={18} className="mt-0.5 shrink-0 text-amber-500" />
      <div>
        <p className="font-medium">Not found in the selected documents</p>
        <p className="mt-0.5 text-amber-800/90">{REFUSAL_MESSAGE} Try rephrasing, or widen the filters.</p>
      </div>
    </div>
  );
}

function MessageActions({ message }: { message: Message }) {
  const { toast } = useToast();
  // What the person just chose, until the stored rating (message.feedback) is loaded again.
  const [chosen, setChosen] = useState<"up" | "down" | null | undefined>(undefined);
  const [copied, setCopied] = useState(false);
  useEffect(() => setChosen(undefined), [message.feedback]);
  const rating = chosen === undefined ? message.feedback : chosen;
  const rate = useMutation({
    mutationFn: (value: "up" | "down") => api<void>(`/messages/${message.id}/feedback`, { method: "POST", body: JSON.stringify({ rating: value }) }),
    onSuccess: (_, value) => {
      setChosen(value);
      toast({ tone: "success", title: "Thanks for the feedback" });
    },
    onError: () => toast({ tone: "error", title: "Could not save your rating" }),
  });
  const copy = () => {
    navigator.clipboard
      .writeText(message.content)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => toast({ tone: "error", title: "Could not copy the answer" }));
  };
  const rateButton = (value: "up" | "down") => (
    <button
      onClick={() => rate.mutate(value)}
      aria-label={value === "up" ? "Good answer" : "Bad answer"}
      aria-pressed={rating === value}
      title={value === "up" ? "Good answer" : "Bad answer"}
      className={`${iconBtn} h-7 w-7 ${rating === value ? (value === "up" ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-600") : ""}`}
    >
      {value === "up" ? <ThumbUpIcon size={15} /> : <ThumbDownIcon size={15} />}
    </button>
  );
  return (
    <div className="flex items-center gap-0.5">
      <button onClick={copy} aria-label="Copy answer" title="Copy" className={`${iconBtn} h-7 w-7`}>
        {copied ? <CheckIcon size={15} className="animate-pop text-emerald-600" /> : <CopyIcon size={15} />}
      </button>
      {rateButton("up")}
      {rateButton("down")}
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex animate-fade-up justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-linear-to-br from-indigo-600 to-violet-600 px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-sm shadow-indigo-500/20">
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ message, onCite }: { message: Message; onCite: (c: Citation) => void }) {
  const refusal = message.content === REFUSAL_MESSAGE;
  return (
    <div className="flex animate-fade-up gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-[15px] text-slate-800 shadow-sm">
          {refusal ? <RefusalNotice /> : <Answer text={message.content} citations={message.citations ?? []} onCite={onCite} />}
          {message.flagged && (
            <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-700">
              <AlertTriangleIcon size={13} /> Some citations could not be verified, or the answer was cut short.
            </p>
          )}
        </div>
        {!refusal && (
          <div className="mt-1 pl-1">
            <MessageActions message={message} />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [filters, setFilters] = useState<ChatFilters>({});
  const [showFilters, setShowFilters] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [input, setInput] = useState("");
  const [stream, setStream] = useState<StreamState | null>(null);
  const [drawer, setDrawer] = useState<Citation | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  // Which question owns the screen. A stream that is no longer the current one (another conversation
  // was opened, or the page was left) must not touch the state of the newer one.
  const requestId = useRef(0);
  const stickToBottom = useRef(true);
  const bottom = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const conversations = useQuery({ queryKey: ["conversations"], queryFn: () => api<Conversation[]>("/conversations") });
  const detail = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api<ConversationDetail>(`/conversations/${conversationId}`),
    enabled: Boolean(conversationId),
  });
  const indexed = useQuery({
    queryKey: ["documents", "indexed"],
    queryFn: () => api<DocumentList>("/documents?status=indexed&limit=200"),
  });
  const messages = conversationId ? (detail.data?.messages ?? []) : [];
  const activeFilters = Object.values(filters).filter((v) => (Array.isArray(v) ? v.length : Boolean(v))).length;

  useEffect(() => {
    if (stickToBottom.current) bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, stream?.text, stream?.status]);

  useEffect(() => {
    // A different conversation (or a new chat) was opened: whatever is still being written belongs
    // to the old one, so stop it and let its handlers go quiet.
    requestId.current += 1;
    controller.current?.abort();
    controller.current = null;
    setDrawer(null);
    setStream(null);
    setSidebarOpen(false);
  }, [conversationId]);
  useEffect(
    () => () => {
      requestId.current += 1;
      controller.current?.abort();
    },
    [],
  );

  // The question box grows with its text, up to about eight lines.
  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const remove = useMutation({
    mutationFn: (id: string) => api<void>(`/conversations/${id}`, { method: "DELETE" }),
    onSuccess: (_, id) => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      toast({ tone: "success", title: "Conversation deleted" });
      if (id === conversationId) navigate("/chat");
    },
    onError: () => toast({ tone: "error", title: "Could not delete that conversation" }),
  });

  const send = async (question: string) => {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const id = ++requestId.current;
    const current = () => requestId.current === id;
    const patch = (change: Partial<StreamState> | ((s: StreamState) => Partial<StreamState>)) =>
      setStream((s) => (s && current() ? { ...s, ...(typeof change === "function" ? change(s) : change) } : s));

    stickToBottom.current = true;
    setInput("");
    setStream({ question, text: "", status: "searching", citations: [], error: null, done: null });
    const clean: ChatFilters = Object.fromEntries(
      Object.entries(filters).filter(([, v]) => (Array.isArray(v) ? v.length > 0 : Boolean(v))),
    );

    /** After the answer: reload the stored conversation, then hand over from the live bubble to it. */
    const finish = async (done: DonePayload) => {
      let saved = true;
      try {
        await queryClient.invalidateQueries({ queryKey: ["conversations"] });
        await queryClient.fetchQuery({
          queryKey: ["conversation", done.conversation_id],
          queryFn: () => api<ConversationDetail>(`/conversations/${done.conversation_id}`),
          staleTime: 0,
        });
      } catch {
        saved = false; // the answer is on screen and stored; the next load will show it
      }
      if (!current()) return;
      if (!conversationId) navigate(`/chat/${done.conversation_id}`); // the new conversation's own page loads it
      else if (saved) setStream(null); // the stored messages show the answer now
      else patch({ status: "done" }); // keep what was streamed on screen
    };

    try {
      await streamChat(
        { question, conversation_id: conversationId, filters: Object.keys(clean).length ? clean : undefined },
        {
          onStatus: (stage, data) => patch((s) => ({ status: stage, text: data.reset ? "" : s.text })),
          onToken: (t) => patch((s) => ({ text: s.text + t })),
          onCitations: (citations) => patch({ citations }),
          onDone: (done) => {
            // Stay busy until the stored conversation is loaded: a question sent in between would
            // otherwise start a second conversation or have its bubble cleared.
            patch({ done, text: done.answer, status: "saving" });
            void finish(done);
          },
          onError: (message) => patch({ error: message, status: "error" }),
        },
        abort.signal,
      );
    } catch (e) {
      if (!current()) return; // another conversation was opened meanwhile
      if (abort.signal.aborted) patch({ status: "stopped" }); // the person pressed Stop
      else patch({ error: e instanceof Error ? e.message : "Request failed", status: "error" });
    }
  };

  const busy = stream !== null && !stream.error && !FINISHED.has(stream.status);
  const ask = (raw: string) => {
    const question = raw.trim();
    if (!question || busy) return;
    if (filters.date_from && filters.date_to && filters.date_from > filters.date_to) {
      setFormError("The From date is after the To date. Fix the filters and ask again.");
      setShowFilters(true);
      return;
    }
    setFormError(null);
    void send(question);
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    ask(input);
  };
  const onComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      ask(input);
    }
  };

  const shownConversations = (conversations.data ?? []).filter((c) => c.title.toLowerCase().includes(search.trim().toLowerCase()));
  const title = conversationId ? (detail.data?.title ?? "Conversation") : "New chat";
  const empty = messages.length === 0 && !stream && !(conversationId && (detail.isLoading || detail.isError));
  const docCount = indexed.data?.total ?? 0;

  return (
    <div className="relative flex h-full">
      {/* Conversation list: a column on wide screens, a sliding drawer on small ones. */}
      {sidebarOpen && <div className="fixed inset-0 z-30 animate-fade-in bg-slate-900/30 md:hidden" onClick={() => setSidebarOpen(false)} />}
      <aside
        className={`${sidebarOpen ? "fixed inset-y-0 left-0 z-40 flex animate-slide-in-left shadow-2xl" : "hidden"} w-72 shrink-0 flex-col border-r border-slate-200 bg-white md:static md:z-auto md:flex md:animate-none md:shadow-none`}
        aria-label="Conversations"
      >
        <div className="space-y-3 p-3">
          <div className="flex items-center gap-2">
            <Link
              to="/chat"
              onClick={() => setSidebarOpen(false)}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-linear-to-r from-indigo-600 to-violet-600 py-2.5 text-sm font-medium text-white shadow-sm shadow-indigo-500/25 transition hover:shadow-md active:scale-[0.98]"
            >
              <PlusIcon size={16} /> New chat
            </Link>
            <button onClick={() => setSidebarOpen(false)} aria-label="Close conversations" className={`${iconBtn} md:hidden`}>
              <XIcon size={18} />
            </button>
          </div>
          <div className="relative">
            <SearchIcon size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search conversations"
              aria-label="Search conversations"
              className="w-full rounded-lg border border-transparent bg-slate-100 py-2 pl-9 pr-3 text-sm transition placeholder:text-slate-400 focus:border-indigo-300 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-500/10"
            />
          </div>
        </div>
        <nav className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
          {conversations.isLoading && [0, 1, 2, 3].map((i) => <Skeleton key={i} className="mx-1 my-2 h-9" />)}
          {shownConversations.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center rounded-lg transition-colors ${c.id === conversationId ? "bg-indigo-50 text-indigo-900" : "text-slate-700 hover:bg-slate-100"}`}
            >
              <Link to={`/chat/${c.id}`} className="flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2" title={c.title}>
                <ChatIcon size={15} className={`shrink-0 ${c.id === conversationId ? "text-indigo-500" : "text-slate-400"}`} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm">{c.title}</span>
                  <span className="block text-[11px] text-slate-400">{timeAgo(c.updated_at)}</span>
                </span>
              </Link>
              <button
                onClick={() => window.confirm("Delete this conversation?") && remove.mutate(c.id)}
                aria-label={`Delete conversation: ${c.title}`}
                title="Delete"
                className="mr-1 rounded-md p-1.5 text-slate-400 opacity-0 transition hover:bg-rose-50 hover:text-rose-600 focus:opacity-100 group-focus-within:opacity-100 group-hover:opacity-100"
              >
                <TrashIcon size={14} />
              </button>
            </div>
          ))}
          {conversations.data?.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-slate-500">No conversations yet. Ask your first question.</p>
          )}
          {conversations.data && conversations.data.length > 0 && shownConversations.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-slate-500">Nothing matches "{search}".</p>
          )}
          {conversations.isError && (
            <p role="alert" className="px-3 py-3 text-sm text-rose-600">
              Could not load your conversations.
            </p>
          )}
        </nav>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col bg-linear-to-b from-slate-50 to-white">
        <div className="flex items-center gap-2 border-b border-slate-200/80 bg-white/80 px-3 py-2 backdrop-blur sm:px-4">
          <button onClick={() => setSidebarOpen(true)} aria-label="Show conversations" className={`${iconBtn} md:hidden`}>
            <MenuIcon size={18} />
          </button>
          <h1 className="min-w-0 flex-1 truncate text-sm font-semibold text-slate-800">{title}</h1>
          <button
            onClick={() => setShowFilters((v) => !v)}
            aria-expanded={showFilters}
            className={`${btnSecondary} py-1.5 ${activeFilters ? "border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-50" : ""}`}
          >
            <FilterIcon size={15} />
            Filters
            {activeFilters > 0 && (
              <span className="grid h-5 min-w-5 animate-pop place-items-center rounded-full bg-indigo-600 px-1 text-[11px] font-semibold text-white">{activeFilters}</span>
            )}
          </button>
        </div>
        {showFilters && <FilterBar filters={filters} onChange={setFilters} />}

        <div
          className="min-h-0 flex-1 overflow-y-auto"
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {conversationId && detail.isLoading && (
              <div className="space-y-6">
                <Skeleton className="ml-auto h-10 w-2/3 rounded-2xl" />
                <div className="flex gap-3">
                  <Skeleton className="h-8 w-8 rounded-full" />
                  <Skeleton className="h-24 flex-1 rounded-2xl" />
                </div>
              </div>
            )}
            {conversationId && detail.isError && (
              <div role="alert" className="mx-auto mt-16 max-w-md animate-fade-up rounded-2xl border border-rose-200 bg-rose-50 p-5 text-center text-sm text-rose-700">
                This conversation could not be opened. It may have been deleted.{" "}
                <Link to="/chat" className="font-medium underline">
                  Start a new chat
                </Link>
              </div>
            )}

            {empty && (
              <div className="flex animate-fade-up flex-col items-center pt-10 text-center sm:pt-16">
                <span className="animate-float">
                  <LogoMark size={56} />
                </span>
                <h2 className="mt-5 text-2xl font-semibold tracking-tight text-slate-900">Ask anything about your documents</h2>
                <p className="mt-2 max-w-md text-sm text-slate-500">
                  Answers come only from your uploaded files, and every claim cites its source.
                  {docCount > 0 ? (
                    <span className="mt-2 flex items-center justify-center gap-1.5 text-indigo-600">
                      <BookOpenIcon size={15} /> {docCount} document{docCount === 1 ? "" : "s"} ready to search
                    </span>
                  ) : indexed.data ? (
                    <span className="mt-2 block">
                      No documents yet.{" "}
                      <Link to="/library" className="font-medium text-indigo-600 underline">
                        Upload some in the Library
                      </Link>
                      .
                    </span>
                  ) : null}
                </p>
                <div className="mt-8 grid w-full gap-3 sm:grid-cols-2">
                  {SUGGESTIONS.map((s, i) => (
                    <button
                      key={s.title}
                      onClick={() => ask(s.prompt)}
                      disabled={docCount === 0}
                      className="group animate-fade-up rounded-xl border border-slate-200 bg-white p-4 text-left shadow-xs transition-all hover:-translate-y-0.5 hover:border-indigo-200 hover:shadow-md disabled:pointer-events-none disabled:opacity-50"
                      style={{ animationDelay: `${120 + i * 60}ms` }}
                    >
                      <p className="flex items-center gap-2 text-sm font-medium text-slate-800">
                        <SparklesIcon size={15} className="text-indigo-500 transition-transform group-hover:rotate-12" /> {s.title}
                      </p>
                      <p className="mt-1 line-clamp-2 text-sm text-slate-500">{s.prompt}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m) => (m.role === "user" ? <UserBubble key={m.id} text={m.content} /> : <AssistantBubble key={m.id} message={m} onCite={setDrawer} />))}

            {stream && (
              <>
                <UserBubble text={stream.question} />
                <div className="flex animate-fade-up gap-3" aria-live="polite" aria-busy={busy}>
                  <AssistantAvatar />
                  <div className="min-w-0 flex-1 space-y-2 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-[15px] text-slate-800 shadow-sm">
                    {stream.error ? (
                      <p role="alert" className="flex items-start gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">
                        <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> {stream.error}
                      </p>
                    ) : stream.text ? (
                      stream.text === REFUSAL_MESSAGE ? (
                        <RefusalNotice />
                      ) : (
                        <Answer text={stream.text} citations={stream.citations} onCite={setDrawer} streaming={stream.status === "generating"} />
                      )
                    ) : null}
                    {stream.status === "stopped" && <p className="text-sm text-slate-500">Stopped. This answer was not saved.</p>}
                    {busy && !(stream.status === "generating" && stream.text) && <TypingIndicator label={STATUS_TEXT[stream.status] ?? "Working"} />}
                  </div>
                </div>
              </>
            )}
            <div ref={bottom} />
          </div>
        </div>

        <div className="border-t border-slate-200/80 bg-white/90 px-3 pb-3 pt-2 backdrop-blur sm:px-4">
          <form onSubmit={submit} className="mx-auto max-w-3xl">
            {formError && (
              <p role="alert" className="mb-2 animate-shake rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {formError}
              </p>
            )}
            <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-500/10">
              <textarea
                ref={textarea}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onComposerKey}
                maxLength={MAX_QUESTION}
                rows={1}
                aria-label="Your question"
                placeholder="Ask a question about your documents…"
                className="max-h-[200px] min-h-[40px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-slate-900 placeholder:text-slate-400 focus:outline-none"
              />
              {busy ? (
                <button
                  type="button"
                  onClick={() => controller.current?.abort()}
                  aria-label="Stop"
                  title="Stop"
                  className="grid h-10 w-10 shrink-0 animate-scale-in place-items-center rounded-xl bg-slate-900 text-white transition hover:bg-slate-700 active:scale-95"
                >
                  <StopIcon size={16} />
                </button>
              ) : (
                <button
                  disabled={!input.trim()}
                  aria-label="Send"
                  title="Send"
                  className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-linear-to-br from-indigo-600 to-violet-600 text-white shadow-sm shadow-indigo-500/30 transition hover:shadow-md active:scale-95 disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
                >
                  <SendIcon size={16} />
                </button>
              )}
            </div>
            <p className="mt-1.5 flex justify-between px-1 text-[11px] text-slate-400">
              <span>Enter to send · Shift + Enter for a new line</span>
              {input.length > MAX_QUESTION - 300 && (
                <span className={input.length >= MAX_QUESTION ? "text-rose-500" : ""}>
                  {input.length}/{MAX_QUESTION}
                </span>
              )}
            </p>
          </form>
        </div>
        <SourceDrawer citation={drawer} onClose={() => setDrawer(null)} />
      </section>
    </div>
  );
}
