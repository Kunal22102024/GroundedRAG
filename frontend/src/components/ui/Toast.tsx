import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { AlertCircleIcon, CheckCircleIcon, InfoIcon, XIcon } from "./icons";

type Tone = "success" | "error" | "info";
interface ToastItem {
  id: number;
  title: string;
  description?: string;
  tone: Tone;
}
interface ToastApi {
  toast: (t: { title: string; description?: string; tone?: Tone }) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const STYLE: Record<Tone, { icon: ReactNode; ring: string }> = {
  success: { icon: <CheckCircleIcon className="text-emerald-500" size={20} />, ring: "ring-emerald-100" },
  error: { icon: <AlertCircleIcon className="text-rose-500" size={20} />, ring: "ring-rose-100" },
  info: { icon: <InfoIcon className="text-indigo-500" size={20} />, ring: "ring-indigo-100" },
};

/** Small notifications in the bottom-right corner; they leave by themselves after a few seconds. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setItems((all) => all.filter((t) => t.id !== id)), []);
  const toast = useCallback<ToastApi["toast"]>(
    ({ title, description, tone = "info" }) => {
      const id = nextId.current++;
      setItems((all) => [...all.slice(-3), { id, title, description, tone }]);
      setTimeout(() => dismiss(id), tone === "error" ? 7000 : 4500);
    },
    [dismiss],
  );
  const value = useMemo(() => ({ toast }), [toast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-full max-w-sm flex-col gap-2">
        {items.map((t) => (
          <div
            key={t.id}
            role={t.tone === "error" ? "alert" : "status"}
            className={`pointer-events-auto flex animate-toast-in items-start gap-3 rounded-xl bg-white p-3.5 shadow-lg ring-1 ${STYLE[t.tone].ring}`}
          >
            <span className="mt-0.5 shrink-0">{STYLE[t.tone].icon}</span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-slate-900">{t.title}</p>
              {t.description && <p className="mt-0.5 break-words text-sm text-slate-500">{t.description}</p>}
            </div>
            <button onClick={() => dismiss(t.id)} aria-label="Dismiss" className="shrink-0 rounded-md p-1 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600">
              <XIcon size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const value = useContext(ToastContext);
  if (!value) throw new Error("useToast must be used inside ToastProvider");
  return value;
}
