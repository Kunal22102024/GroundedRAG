import { useEffect, useRef, type ReactNode } from "react";
import { fileExtension } from "../../lib/files";
import type { AccessLevel, DocStatus } from "../../lib/types";
import { XIcon } from "./icons";

export function Spinner({ size = 16, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={`animate-spin ${className}`} aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

type Tone = "indigo" | "emerald" | "amber" | "rose" | "slate";
const BAR: Record<Tone, string> = {
  indigo: "bg-linear-to-r from-indigo-500 to-violet-500",
  emerald: "bg-emerald-500",
  amber: "bg-amber-500",
  rose: "bg-rose-500",
  slate: "bg-slate-400",
};

/** A determinate bar for a known fraction (0..1), or an indeterminate sweep when `value` is undefined. */
export function ProgressBar({ value, tone = "indigo", label }: { value?: number; tone?: Tone; label: string }) {
  const percent = value === undefined ? undefined : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
      className="relative h-1.5 w-full overflow-hidden rounded-full bg-slate-100"
    >
      {percent === undefined ? (
        <div className={`absolute inset-y-0 w-2/5 animate-indeterminate rounded-full ${BAR[tone]}`} />
      ) : (
        <div className={`h-full rounded-full transition-[width] duration-300 ease-out ${BAR[tone]}`} style={{ width: `${percent}%` }} />
      )}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

const STATUS: Record<DocStatus, { label: string; pill: string; dot: string; live: boolean }> = {
  queued: { label: "Queued", pill: "bg-slate-100 text-slate-700 ring-slate-200", dot: "bg-slate-400", live: true },
  processing: { label: "Indexing", pill: "bg-amber-50 text-amber-800 ring-amber-200", dot: "bg-amber-500", live: true },
  indexed: { label: "Ready", pill: "bg-emerald-50 text-emerald-700 ring-emerald-200", dot: "bg-emerald-500", live: false },
  failed: { label: "Failed", pill: "bg-rose-50 text-rose-700 ring-rose-200", dot: "bg-rose-500", live: false },
};

export function StatusPill({ status, title }: { status: DocStatus; title?: string }) {
  const s = STATUS[status];
  return (
    <span title={title} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${s.pill}`}>
      <span className="relative flex h-2 w-2">
        {s.live && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${s.dot}`} />}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${s.dot}`} />
      </span>
      {s.label}
    </span>
  );
}

const ACCESS: Record<AccessLevel, string> = {
  Public: "bg-sky-50 text-sky-700 ring-sky-200",
  Internal: "bg-slate-100 text-slate-600 ring-slate-200",
  Restricted: "bg-rose-50 text-rose-700 ring-rose-200",
};

export function AccessBadge({ level }: { level: AccessLevel }) {
  return <span className={`inline-flex rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${ACCESS[level]}`}>{level}</span>;
}

const TYPE_STYLE: Record<string, string> = {
  pdf: "from-rose-500 to-red-600",
  docx: "from-blue-500 to-indigo-600",
  md: "from-violet-500 to-purple-600",
  markdown: "from-violet-500 to-purple-600",
  html: "from-orange-400 to-amber-600",
  htm: "from-orange-400 to-amber-600",
  txt: "from-slate-400 to-slate-600",
};

/** A small coloured tile with the file's extension, so a list of documents scans at a glance. */
export function FileTypeBadge({ name, size = "md" }: { name: string; size?: "sm" | "md" }) {
  const ext = fileExtension(name);
  const box = size === "sm" ? "h-8 w-8 text-[9px]" : "h-10 w-10 text-[10px]";
  return (
    <span
      aria-hidden="true"
      className={`grid shrink-0 place-items-center rounded-lg bg-linear-to-br font-bold uppercase tracking-wide text-white shadow-sm ${box} ${TYPE_STYLE[ext] ?? "from-slate-400 to-slate-600"}`}
    >
      {ext.slice(0, 4) || "file"}
    </span>
  );
}

/** A centred dialog: fades the page, scales in, closes on Escape or a click outside, focuses `initialFocus`. */
export function Modal({
  open,
  onClose,
  label,
  children,
  initialFocus,
}: {
  open: boolean;
  onClose: () => void;
  label: string;
  children: ReactNode;
  initialFocus?: React.RefObject<HTMLElement | null>;
}) {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    initialFocus?.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [open, initialFocus]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 animate-fade-in bg-slate-900/40 backdrop-blur-[2px]" onClick={onClose} />
      <div role="dialog" aria-modal="true" aria-label={label} className="relative w-full max-w-md animate-scale-in rounded-2xl bg-white p-6 shadow-2xl">
        <button onClick={onClose} aria-label="Close" className="absolute right-3 top-3 rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700">
          <XIcon size={16} />
        </button>
        {children}
      </div>
    </div>
  );
}
