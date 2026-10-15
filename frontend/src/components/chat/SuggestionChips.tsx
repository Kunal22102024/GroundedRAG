import type { ComponentType } from "react";
import type { FollowUp, SuggestionType } from "../../lib/types";
import { ChevronRightIcon, LayersIcon, ListIcon } from "../ui/icons";

const GROUPS: { type: SuggestionType; label: string; Icon: ComponentType<{ size?: number; className?: string }> }[] = [
  { type: "follow_up", label: "Go deeper", Icon: ChevronRightIcon },
  { type: "sub_question", label: "Break it down", Icon: ListIcon },
  { type: "related", label: "Related", Icon: LayersIcon },
];

/** Follow-up, sub- and related questions under an answer; a click asks one in the same conversation. */
export default function SuggestionChips({
  suggestions,
  onAsk,
  disabled = false,
}: {
  suggestions: FollowUp[];
  onAsk: (question: string) => void;
  disabled?: boolean;
}) {
  const groups = GROUPS.map((g) => ({ ...g, items: suggestions.filter((s) => s.type === g.type) })).filter((g) => g.items.length);
  if (!groups.length) return null;
  return (
    <section aria-label="Suggested next questions" className="mt-3 animate-fade-up space-y-2 border-t border-slate-100 pt-3">
      {groups.map(({ type, label, Icon, items }) => (
        <div key={type} className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            <Icon size={12} /> {label}
          </span>
          {items.map((s) => (
            <button
              key={s.question}
              type="button"
              disabled={disabled}
              onClick={() => onAsk(s.question)}
              className="rounded-full border border-indigo-200 bg-indigo-50/60 px-3 py-1 text-left text-[13px] text-indigo-800 transition hover:-translate-y-px hover:border-indigo-300 hover:bg-indigo-100/70 disabled:pointer-events-none disabled:opacity-50"
            >
              {s.question}
            </button>
          ))}
        </div>
      ))}
    </section>
  );
}
