// Small charts for the evaluation dashboard: plain SVG and CSS, no chart library.

/** A tiny line of daily values (no axes), for the KPI cards. */
export function Sparkline({ values, label, className = "text-indigo-500" }: { values: (number | null)[]; label: string; className?: string }) {
  const points = values.map((v) => v ?? 0);
  if (points.length < 2) return <div className="h-7" aria-hidden="true" />;
  const max = Math.max(...points, 1e-9);
  const step = 100 / (points.length - 1);
  const xy = points.map((v, i) => `${(i * step).toFixed(2)},${(26 - (v / max) * 24).toFixed(2)}`);
  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className={`h-7 w-full ${className}`} role="img" aria-label={label}>
      <polygon points={`0,28 ${xy.join(" ")} 100,28`} fill="currentColor" opacity="0.12" />
      <polyline points={xy.join(" ")} fill="none" stroke="currentColor" strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

export interface Bar {
  label: string;
  value: number;
  /** Text shown at the end of the bar (defaults to the value). */
  display?: string;
  /** Tailwind background class for the bar. */
  tone?: string;
  hint?: string;
}

/** Horizontal bars, scaled to `max` (default: the largest value). */
export function BarList({ items, max, label }: { items: Bar[]; max?: number; label: string }) {
  const top = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  if (!items.length) return <p className="text-sm text-slate-400">No data yet.</p>;
  return (
    <ul className="space-y-2" aria-label={label}>
      {items.map((item) => (
        <li key={item.label} title={item.hint}>
          <div className="mb-0.5 flex items-baseline justify-between gap-3 text-xs">
            <span className="truncate text-slate-600">{item.label}</span>
            <span className="shrink-0 font-medium tabular-nums text-slate-800">{item.display ?? item.value}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full transition-[width] duration-500 ${item.tone ?? "bg-linear-to-r from-indigo-500 to-violet-500"}`}
              style={{ width: `${Math.max(0, Math.min(100, (item.value / top) * 100))}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Best-similarity histogram (0..1) with the relevance threshold marked. */
export function Histogram({ bins, threshold }: { bins: { from: number; count: number }[]; threshold: number }) {
  const max = Math.max(...bins.map((b) => b.count), 1);
  const total = bins.reduce((sum, b) => sum + b.count, 0);
  const width = 100 / Math.max(bins.length, 1);
  return (
    <figure className="space-y-1">
      <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="h-36 w-full" role="img" aria-label={`Best similarity of ${total} questions; the threshold is ${threshold}`}>
        {bins.map((b, i) => {
          const h = (b.count / max) * 46;
          const below = b.from + 0.05 <= threshold + 1e-9;
          return (
            <rect key={b.from} x={i * width + 0.4} y={50 - h} width={width - 0.8} height={h} rx="0.6" className={below ? "fill-slate-400" : "fill-indigo-500"}>
              <title>{`${b.from.toFixed(2)}–${(b.from + 0.05).toFixed(2)}: ${b.count}`}</title>
            </rect>
          );
        })}
        <line x1={threshold * 100} x2={threshold * 100} y1="0" y2="50" className="stroke-rose-500" strokeWidth="0.6" strokeDasharray="1.5 1" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption className="flex justify-between text-[11px] text-slate-400">
        <span>0</span>
        <span className="text-rose-500">threshold {threshold.toFixed(2)}</span>
        <span>1.0</span>
      </figcaption>
    </figure>
  );
}

export interface StackedColumn {
  label: string;
  parts: { value: number; className: string; name: string }[];
}

/** Vertical stacked columns (e.g. tokens per day). */
export function StackedBars({ columns, label, format = (v: number) => v.toLocaleString() }: { columns: StackedColumn[]; label: string; format?: (v: number) => string }) {
  const totals = columns.map((c) => c.parts.reduce((sum, p) => sum + p.value, 0));
  const max = Math.max(...totals, 1e-9);
  if (!columns.length) return <p className="text-sm text-slate-400">No data yet.</p>;
  return (
    <div role="img" aria-label={label} className="flex h-36 items-end justify-center gap-1 border-b border-slate-200">
      {columns.map((column, i) => (
        // Capped width: a period with a single day must not become one full-width block.
        <div key={column.label} className="group flex h-full min-w-0 max-w-10 flex-1 flex-col justify-end" title={`${column.label}: ${column.parts.map((p) => `${p.name} ${format(p.value)}`).join(", ")}`}>
          <div className="flex flex-col-reverse overflow-hidden rounded-t" style={{ height: `${(totals[i] / max) * 100}%` }}>
            {column.parts.map((part) => (
              <div key={part.name} className={part.className} style={{ height: totals[i] ? `${(part.value / totals[i]) * 100}%` : 0 }} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
