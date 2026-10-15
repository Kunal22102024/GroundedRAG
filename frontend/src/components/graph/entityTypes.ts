// How each kind of entity is drawn and named. Full class names, so Tailwind finds them.

export interface EntityStyle {
  label: string;
  /** SVG circle fill. */
  fill: string;
  /** Legend and chip dot. */
  dot: string;
}

export const ENTITY_STYLES: Record<string, EntityStyle> = {
  person: { label: "Person", fill: "fill-sky-500", dot: "bg-sky-500" },
  organization: { label: "Organization", fill: "fill-violet-500", dot: "bg-violet-500" },
  role: { label: "Role", fill: "fill-amber-500", dot: "bg-amber-500" },
  concept: { label: "Concept", fill: "fill-indigo-500", dot: "bg-indigo-500" },
  course: { label: "Course", fill: "fill-emerald-500", dot: "bg-emerald-500" },
  policy: { label: "Policy", fill: "fill-rose-500", dot: "bg-rose-500" },
  process: { label: "Process", fill: "fill-cyan-500", dot: "bg-cyan-500" },
  product: { label: "Product", fill: "fill-fuchsia-500", dot: "bg-fuchsia-500" },
  location: { label: "Place", fill: "fill-lime-600", dot: "bg-lime-600" },
  event: { label: "Event", fill: "fill-orange-500", dot: "bg-orange-500" },
  other: { label: "Other", fill: "fill-slate-400", dot: "bg-slate-400" },
};

export const ENTITY_TYPE_ORDER = Object.keys(ENTITY_STYLES);

export function entityStyle(type: string): EntityStyle {
  return ENTITY_STYLES[type] ?? ENTITY_STYLES.other;
}

/** "calculus_ii_requires" -> "calculus ii requires": relation labels are stored as snake_case. */
export function relationText(label: string): string {
  return label.replace(/_/g, " ");
}
