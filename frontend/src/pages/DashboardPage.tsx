import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link, Navigate } from "react-router-dom";
import { api } from "../api/client";
import { BarList, Histogram, Sparkline, StackedBars } from "../components/dashboard/charts";
import { AlertCircleIcon, ChartIcon, RefreshIcon, ThumbDownIcon, ThumbUpIcon } from "../components/ui/icons";
import { Skeleton } from "../components/ui/primitives";
import { useAuth } from "../hooks/useAuth";
import { timeAgo } from "../lib/format";
import type { MetricsSummary, QueryEventItem } from "../lib/types";
import { btnGhost, card } from "../lib/ui";

type Scope = "all" | "chat" | "analysis";
type Problem = "failed" | "refused" | "flagged";

const RANGES = [
  { days: 1, label: "24 h" },
  { days: 7, label: "7 days" },
  { days: 30, label: "30 days" },
];
const SCOPES: { id: Scope; label: string }[] = [
  { id: "all", label: "All" },
  { id: "chat", label: "Chat" },
  { id: "analysis", label: "Analysis" },
];
// Targets from TDD section 12, for the latency bars.
const STAGE_TARGET_MS: Record<string, number> = { embed: 300, dense: 150, keyword: 150, ttft: 1200 };
const GUARD_LABEL: Record<string, string> = {
  relevant: "Relevant (passed)",
  lexical_override: "Exact identifier (override)",
  below_threshold: "Below threshold (refused)",
  no_candidates: "Nothing found (refused)",
};
const PAGE = 25;

const pct = (v: number | null | undefined, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
const ms = (v: number | null | undefined) => (v == null ? "—" : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`);
const num = (v: number) => (v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : v >= 10_000 ? `${Math.round(v / 1000)}k` : v.toLocaleString());

function Segmented<T extends string | number>({ options, value, onChange, label }: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div role="group" aria-label={label} className="flex rounded-lg bg-slate-100 p-0.5 text-xs font-medium">
      {options.map((o) => (
        <button key={String(o.id)} onClick={() => onChange(o.id)} aria-pressed={value === o.id} className={`rounded-md px-2.5 py-1 transition ${value === o.id ? "bg-white text-indigo-700 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Kpi({ label, value, sub, spark, sparkLabel, tone }: { label: string; value: string; sub?: string; spark?: (number | null)[]; sparkLabel?: string; tone?: string }) {
  return (
    <div className={`${card} flex animate-fade-up flex-col gap-1 p-4`}>
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className="text-2xl font-semibold tabular-nums tracking-tight text-slate-900">{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
      {spark && <Sparkline values={spark} label={sparkLabel ?? label} className={tone} />}
    </div>
  );
}

function Section({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className={`${card} animate-fade-up space-y-4 p-4 sm:p-5`} aria-label={title}>
      <header>
        <h2 className="font-semibold text-slate-900">{title}</h2>
        <p className="text-sm text-slate-500">{description}</p>
      </header>
      {children}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className="font-semibold tabular-nums text-slate-800">{value}</p>
    </div>
  );
}

function ProblemQueries({ scope }: { scope: Scope }) {
  const [outcome, setOutcome] = useState<Problem>("failed");
  const [open, setOpen] = useState<string | null>(null);
  const list = useInfiniteQuery({
    queryKey: ["metrics", "events", outcome, scope],
    initialPageParam: "",
    queryFn: ({ pageParam }) =>
      api<QueryEventItem[]>(`/metrics/events?outcome=${outcome}&kind=${scope}&limit=${PAGE}${pageParam ? `&before=${encodeURIComponent(pageParam)}` : ""}`),
    getNextPageParam: (last) => (last.length === PAGE ? last[last.length - 1].created_at : undefined),
  });
  const rows = list.data?.pages.flat() ?? [];
  const reason = (e: QueryEventItem) =>
    e.outcome === "failed"
      ? (e.error_code ?? "failed")
      : e.outcome === "refused"
        ? (GUARD_LABEL[e.guard_reason ?? ""] ?? "Refused by the model")
        : [e.citations_removed && `${e.citations_removed} invalid citation(s) removed`, e.uncited && "no citation", e.partial && "cut short", e.retried && "retried"].filter(Boolean).join(", ");

  return (
    <section className={`${card} animate-fade-up overflow-hidden`} aria-label="Problem queries">
      <header className="flex flex-wrap items-center gap-3 border-b border-slate-100 p-4 sm:px-5">
        <div className="min-w-0 flex-1">
          <h2 className="font-semibold text-slate-900">Failed and problem queries</h2>
          <p className="text-sm text-slate-500">What went wrong, newest first. Questions are visible to administrators only.</p>
        </div>
        <Segmented
          label="Kind of problem"
          value={outcome}
          onChange={(v) => setOutcome(v)}
          options={[
            { id: "failed", label: "Failed" },
            { id: "refused", label: "Refused" },
            { id: "flagged", label: "Flagged" },
          ]}
        />
      </header>
      {list.isLoading ? (
        <div className="space-y-2 p-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-10" />
          ))}
        </div>
      ) : rows.length === 0 ? (
        <p className="p-6 text-center text-sm text-slate-500">Nothing here in this period. Good.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[44rem] text-left text-sm">
            <thead className="bg-slate-50/70 text-xs font-medium text-slate-500">
              <tr>
                <th className="px-4 py-2.5">When</th>
                <th className="px-4 py-2.5">Question</th>
                <th className="px-4 py-2.5">Why</th>
                <th className="px-4 py-2.5 text-right">Similarity</th>
                <th className="px-4 py-2.5 text-right">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((e) => (
                <tr key={e.id} className="align-top hover:bg-slate-50/60">
                  <td className="whitespace-nowrap px-4 py-2.5 text-slate-500">
                    <span title={new Date(e.created_at).toLocaleString()}>{timeAgo(e.created_at)}</span>
                    <span className="block text-[11px] text-slate-400">
                      {e.kind}
                      {e.user_email ? ` · ${e.user_email}` : ""}
                    </span>
                  </td>
                  <td className="max-w-md px-4 py-2.5 text-slate-800">
                    {e.question ? (
                      <button onClick={() => setOpen(open === e.id ? null : e.id)} className={`text-left ${open === e.id ? "" : "line-clamp-2"}`} title="Show the whole question">
                        {e.question}
                      </button>
                    ) : (
                      <span className="text-slate-400">{e.document_ids?.length ? `${e.document_ids.length} document(s)` : "—"}</span>
                    )}
                    {e.kind === "chat" && e.conversation_id && e.outcome !== "failed" && (
                      <Link to={`/chat/${e.conversation_id}`} className="mt-0.5 block text-[11px] text-indigo-600 hover:underline">
                        Open conversation
                      </Link>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-slate-600">
                    <span className={`inline-flex rounded-md px-2 py-0.5 text-xs ${e.outcome === "failed" ? "bg-rose-50 text-rose-700" : e.outcome === "refused" ? "bg-amber-50 text-amber-800" : "bg-violet-50 text-violet-700"}`}>
                      {reason(e)}
                    </span>
                    {e.source !== "typed" && <span className="ml-1 text-[11px] text-slate-400">via {e.source}</span>}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-slate-600">{e.best_similarity == null ? "—" : e.best_similarity.toFixed(2)}</td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-right tabular-nums text-slate-600">{ms(e.total_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {list.hasNextPage && (
            <div className="border-t border-slate-100 p-3 text-center">
              <button onClick={() => void list.fetchNextPage()} disabled={list.isFetchingNextPage} className={btnGhost}>
                {list.isFetchingNextPage ? "Loading…" : "Load more"}
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/** Live evaluation of the app: relevance, groundedness, abstention, latency, cost and failures (admins only). */
export default function DashboardPage() {
  const { user } = useAuth();
  const [days, setDays] = useState(7);
  const [scope, setScope] = useState<Scope>("all");
  const summary = useQuery({
    queryKey: ["metrics", "summary", days, scope],
    queryFn: () => api<MetricsSummary>(`/metrics/summary?days=${days}&kind=${scope}`),
    refetchInterval: 60_000,
    enabled: user?.role === "admin",
  });
  if (user && user.role !== "admin") return <Navigate to="/chat" replace />;

  const m = summary.data;
  const daily = m?.daily ?? [];
  const stages = (m?.latency.stages ?? []).map((s) => ({
    label: `${s.stage}${STAGE_TARGET_MS[s.stage] ? ` (target ${ms(STAGE_TARGET_MS[s.stage])})` : ""}`,
    value: s.p95 ?? 0,
    display: `p50 ${ms(s.p50)} · p95 ${ms(s.p95)}`,
    tone: STAGE_TARGET_MS[s.stage] && (s.p95 ?? 0) > STAGE_TARGET_MS[s.stage] ? "bg-amber-400" : undefined,
  }));
  const g = m?.groundedness;

  return (
    <div className="h-full overflow-y-auto bg-linear-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-6">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-0 basis-full sm:basis-auto sm:flex-1">
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-slate-900">
              <ChartIcon size={22} className="text-indigo-500" /> Evaluation
            </h1>
            <p className="text-sm text-slate-500">How the app performs on real questions: every chat turn and analysis run, failures included.</p>
          </div>
          <Segmented label="Period" value={days} onChange={setDays} options={RANGES.map((r) => ({ id: r.days, label: r.label }))} />
          <Segmented label="What" value={scope} onChange={setScope} options={SCOPES} />
          <button onClick={() => void summary.refetch()} className={btnGhost} aria-label="Refresh">
            <RefreshIcon size={15} className={summary.isFetching ? "animate-spin" : ""} /> Refresh
          </button>
        </div>

        {summary.isError && (
          <p role="alert" className="flex items-start gap-2 rounded-xl bg-rose-50 p-3.5 text-sm text-rose-700">
            <AlertCircleIcon size={17} className="mt-0.5 shrink-0" /> Could not load the figures. {summary.error instanceof Error ? summary.error.message : ""}
          </p>
        )}

        {!m ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 8 }, (_, i) => (
              <Skeleton key={i} className="h-28 rounded-2xl" />
            ))}
          </div>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Kpi label="Queries" value={num(m.totals.queries)} sub={`${m.totals.answered} answered · ${m.totals.refused} refused · ${m.totals.failed} failed`} spark={daily.map((d) => d.queries)} />
              <Kpi label="Abstention rate" value={pct(m.totals.abstention_rate)} sub={`${m.abstention.guard_refusals} by the guard · ${m.abstention.model_refusals} by the model`} spark={daily.map((d) => d.abstention_rate)} tone="text-amber-500" />
              <Kpi label="Citation precision" value={pct(g?.citation_precision, 1)} sub={`${g?.invalid_citations_removed ?? 0} invalid citation(s) removed`} />
              <Kpi label="Answers with citations" value={pct(g?.with_citations_rate)} sub={`${g?.answers ?? 0} chat answers`} />
              <Kpi label="Latency p95" value={ms(m.latency.p95_ms)} sub={`p50 ${ms(m.latency.p50_ms)} · max ${ms(m.latency.max_ms)}`} spark={daily.map((d) => d.p95_ms)} tone="text-sky-500" />
              <Kpi label="Failure rate" value={pct(m.totals.failure_rate, 1)} sub={`${m.totals.failed} failed`} spark={daily.map((d) => d.failure_rate)} tone="text-rose-500" />
              <Kpi
                label="Tokens"
                value={num(m.cost.input_tokens + m.cost.output_tokens)}
                sub={m.cost.priced ? `≈ $${(m.cost.usd ?? 0).toFixed(4)}` : "Set LLM_*_USD_PER_MTOK to see cost"}
                spark={daily.map((d) => d.input_tokens + d.output_tokens)}
                tone="text-violet-500"
              />
              <Kpi label="Asked from a suggestion" value={pct(m.suggestions.usage_rate)} sub={`follow-ups clicked on ${pct(m.suggestions.followup_click_rate)} of answers`} />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Section title="Retrieval relevance" description="Best dense similarity of each chat question; below the red line the question is refused without calling the model.">
                <Histogram bins={m.relevance.histogram} threshold={m.threshold} />
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Stat label="Best, answered (mean)" value={m.relevance.best_similarity_answered_mean?.toFixed(3) ?? "—"} />
                  <Stat label="Best, answered (median)" value={m.relevance.best_similarity_answered_p50?.toFixed(3) ?? "—"} />
                  <Stat label="Best, refused (mean)" value={m.relevance.best_similarity_refused_mean?.toFixed(3) ?? "—"} />
                  <Stat label="All retrieved (mean)" value={m.relevance.mean_similarity_answered?.toFixed(3) ?? "—"} />
                </div>
                <BarList
                  label="Guard decisions"
                  items={Object.entries(m.relevance.guard_reasons).map(([reason, count]) => ({
                    label: GUARD_LABEL[reason] ?? reason,
                    value: count,
                    tone: reason === "relevant" || reason === "lexical_override" ? undefined : "bg-amber-400",
                  }))}
                />
              </Section>

              <Section title="Groundedness and citation accuracy" description="From the citation checks every answer goes through: invalid markers are removed, an answer without a valid citation is retried once.">
                <BarList
                  label="Citation checks"
                  max={1}
                  items={[
                    { label: "Answers with a valid citation", value: g?.with_citations_rate ?? 0, display: pct(g?.with_citations_rate) },
                    { label: "Citation precision (valid ÷ all markers)", value: g?.citation_precision ?? 0, display: pct(g?.citation_precision, 1) },
                    { label: "Answers retried for their citations", value: g?.retried_rate ?? 0, display: pct(g?.retried_rate), tone: "bg-amber-400" },
                    { label: "Answers left without a citation", value: g?.uncited_rate ?? 0, display: pct(g?.uncited_rate), tone: "bg-rose-400" },
                    { label: "Answers cut short", value: g?.truncated_rate ?? 0, display: pct(g?.truncated_rate), tone: "bg-rose-400" },
                    { label: "Answers whose sources disagree", value: g?.conflict_rate ?? 0, display: pct(g?.conflict_rate), tone: "bg-amber-400" },
                  ]}
                />
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Stat label="Citations per answer" value={g?.avg_citations?.toFixed(1) ?? "—"} />
                  <Stat label="Documents per answer" value={g?.avg_documents?.toFixed(1) ?? "—"} />
                  <div className="flex items-center gap-2 rounded-lg bg-emerald-50 px-3 py-2 text-emerald-700">
                    <ThumbUpIcon size={15} /> <span className="font-semibold tabular-nums">{g?.feedback_up ?? 0}</span>
                  </div>
                  <div className="flex items-center gap-2 rounded-lg bg-rose-50 px-3 py-2 text-rose-700">
                    <ThumbDownIcon size={15} /> <span className="font-semibold tabular-nums">{g?.feedback_down ?? 0}</span>
                  </div>
                </div>
              </Section>

              <Section title="Latency" description="Per stage, answered and refused requests (failures excluded). Amber: p95 above the TDD target.">
                {stages.length ? <BarList label="Stage latency" items={stages} /> : <p className="text-sm text-slate-400">No timed stages in this period.</p>}
                <StackedBars
                  label="p95 latency per day"
                  format={(v) => ms(v)}
                  columns={daily.map((d) => ({ label: d.day, parts: [{ name: "p95", value: d.p95_ms ?? 0, className: "bg-sky-400" }] }))}
                />
              </Section>

              <Section title="Token cost" description="Model tokens per kind of request and per day (input below, output on top).">
                <BarList
                  label="Tokens by kind"
                  items={m.cost.by_kind.map((k) => ({
                    label: `${k.kind} · ${k.events} request(s)`,
                    value: k.input_tokens + k.output_tokens,
                    display: `${num(k.input_tokens)} in · ${num(k.output_tokens)} out${k.usd != null ? ` · $${k.usd.toFixed(4)}` : ""}`,
                  }))}
                />
                <StackedBars
                  label="Tokens per day"
                  format={num}
                  columns={daily.map((d) => ({
                    label: d.day,
                    parts: [
                      { name: "input", value: d.input_tokens, className: "bg-indigo-400" },
                      { name: "output", value: d.output_tokens, className: "bg-violet-500" },
                    ],
                  }))}
                />
              </Section>
            </div>
          </>
        )}

        <ProblemQueries scope={scope} />
      </div>
    </div>
  );
}
