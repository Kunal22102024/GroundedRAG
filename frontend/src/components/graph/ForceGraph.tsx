import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import type { GraphEdge, GraphNode } from "../../lib/types";
import { entityStyle, relationText } from "./entityTypes";
import { COOLING, HEIGHT, MIN_ALPHA, WIDTH, linkIndexes, seedLayout, settle, tick, type LayoutNode } from "./layout";

interface View {
  x: number;
  y: number;
  w: number;
  h: number;
}

const FULL: View = { x: 0, y: 0, w: WIDTH, h: HEIGHT };
const MIN_WIDTH = 120; // most zoomed in
const LABELLED = 24; // the best connected nodes always carry their name
const DRAG_THRESHOLD = 4; // pixels: less than this is a click

export interface SelectedEdge {
  source: string;
  target: string;
  label: string;
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** The knowledge graph drawn as plain SVG: a force layout that cools down, wheel zoom, drag to pan, drag a node to move it. */
export default function ForceGraph({
  nodes,
  edges,
  selectedId,
  selectedEdge,
  onSelectNode,
  onSelectEdge,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedId: string | null;
  selectedEdge: SelectedEdge | null;
  onSelectNode: (id: string | null) => void;
  onSelectEdge: (edge: SelectedEdge) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const layout = useRef<Map<string, LayoutNode>>(new Map());
  const alpha = useRef(0);
  const frame = useRef<number | null>(null);
  const [, redraw] = useState(0);
  const [view, setView] = useState<View>(FULL);
  const [hovered, setHovered] = useState<string | null>(null);
  const pan = useRef<{ x: number; y: number; view: View } | null>(null);
  const drag = useRef<{ id: string; startX: number; startY: number; moved: boolean } | null>(null);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const visibleEdges = useMemo(() => edges.filter((e) => byId.has(e.source) && byId.has(e.target) && e.source !== e.target), [edges, byId]);
  const labelled = useMemo(() => new Set([...nodes].sort((a, b) => b.degree - a.degree).slice(0, LABELLED).map((n) => n.id)), [nodes]);
  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    const link = (from: string, to: string) => map.set(from, (map.get(from) ?? new Set<string>()).add(to));
    for (const e of visibleEdges) {
      link(e.source, e.target);
      link(e.target, e.source);
    }
    return map;
  }, [visibleEdges]);

  const visibleEdgesRef = useRef(visibleEdges);
  visibleEdgesRef.current = visibleEdges;

  const run = useCallback(() => {
    if (frame.current !== null) return;
    const step = () => {
      const list = [...layout.current.values()];
      const pairs = linkIndexes(list, visibleEdgesRef.current);
      for (let i = 0; i < 2 && alpha.current > MIN_ALPHA; i++) {
        tick(list, pairs, alpha.current);
        alpha.current *= COOLING;
      }
      redraw((n) => n + 1);
      frame.current = alpha.current > MIN_ALPHA ? requestAnimationFrame(step) : null;
    };
    frame.current = requestAnimationFrame(step);
  }, []);

  // New data: keep what is already placed, start the new nodes next to their neighbours, warm up again.
  useEffect(() => {
    layout.current = seedLayout(layout.current, nodes.map((n) => ({ id: n.id, degree: n.degree })), visibleEdges);
    alpha.current = 1;
    if (prefersReducedMotion()) {
      settle([...layout.current.values()], visibleEdges);
      alpha.current = 0;
      redraw((n) => n + 1);
    } else {
      run();
    }
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
    };
  }, [nodes, visibleEdges, run]);

  // Wheel zoom around the pointer. Attached by hand: React's onWheel is passive and cannot stop the page scrolling.
  useEffect(() => {
    const element = svg.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const box = element.getBoundingClientRect();
      const px = (event.clientX - box.left) / box.width;
      const py = (event.clientY - box.top) / box.height;
      setView((v) => {
        const w = clamp(v.w * (event.deltaY > 0 ? 1.15 : 1 / 1.15), MIN_WIDTH, WIDTH);
        const h = (w / WIDTH) * HEIGHT;
        return { x: clamp(v.x + (v.w - w) * px, 0, WIDTH - w), y: clamp(v.y + (v.h - h) * py, 0, HEIGHT - h), w, h };
      });
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, []);

  const toGraph = (clientX: number, clientY: number) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return { x: 0, y: 0 };
    return { x: view.x + ((clientX - box.left) / box.width) * view.w, y: view.y + ((clientY - box.top) / box.height) * view.h };
  };

  const startPan = (event: ReactPointerEvent<SVGRectElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    pan.current = { x: event.clientX, y: event.clientY, view };
  };
  const movePan = (event: ReactPointerEvent<SVGRectElement>) => {
    const start = pan.current;
    const box = svg.current?.getBoundingClientRect();
    if (!start || !box) return;
    const dx = ((event.clientX - start.x) / box.width) * start.view.w;
    const dy = ((event.clientY - start.y) / box.height) * start.view.h;
    setView({ ...start.view, x: clamp(start.view.x - dx, 0, WIDTH - start.view.w), y: clamp(start.view.y - dy, 0, HEIGHT - start.view.h) });
  };
  const endPan = (event: ReactPointerEvent<SVGRectElement>) => {
    const start = pan.current;
    pan.current = null;
    if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) < DRAG_THRESHOLD) onSelectNode(null); // a click on the background
  };

  const startDrag = (event: ReactPointerEvent<SVGGElement>, id: string) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { id, startX: event.clientX, startY: event.clientY, moved: false };
  };
  const moveDrag = (event: ReactPointerEvent<SVGGElement>) => {
    const current = drag.current;
    const node = current && layout.current.get(current.id);
    if (!current || !node) return;
    if (!current.moved && Math.hypot(event.clientX - current.startX, event.clientY - current.startY) < DRAG_THRESHOLD) return;
    current.moved = true;
    const point = toGraph(event.clientX, event.clientY);
    node.x = clamp(point.x, 0, WIDTH);
    node.y = clamp(point.y, 0, HEIGHT);
    node.pinned = true;
    alpha.current = Math.max(alpha.current, 0.3);
    run();
    redraw((n) => n + 1);
  };
  const endDrag = (id: string) => {
    const current = drag.current;
    drag.current = null;
    const node = layout.current.get(id);
    if (node) node.pinned = false;
    if (current && !current.moved) onSelectNode(id);
  };

  const onKey = (event: KeyboardEvent<SVGGElement>, id: string) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelectNode(id);
    }
  };

  const focus = selectedId ?? hovered;
  const near = focus ? neighbours.get(focus) : undefined;
  const zoomedIn = view.w < WIDTH * 0.55;
  const parallel = new Map<string, number>();
  const pairSize = new Map<string, number>();
  for (const e of visibleEdges) {
    const key = [e.source, e.target].sort().join("\u0000");
    pairSize.set(key, (pairSize.get(key) ?? 0) + 1);
  }

  return (
    <svg
      ref={svg}
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      role="group"
      aria-label={`Knowledge graph with ${nodes.length} entities and ${visibleEdges.length} connections. Use the entity list to browse it with the keyboard.`}
      className="aspect-[1000/640] w-full touch-none select-none rounded-xl bg-slate-50/60"
      onKeyDown={(event) => event.key === "Escape" && onSelectNode(null)}
    >
      <defs>
        <marker id="kg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" className="fill-slate-400" />
        </marker>
        <marker id="kg-arrow-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" className="fill-indigo-500" />
        </marker>
      </defs>
      <rect x={0} y={0} width={WIDTH} height={HEIGHT} className="fill-transparent" style={{ cursor: pan.current ? "grabbing" : "grab" }} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan} />

      <g>
        {visibleEdges.map((edge) => {
          const a = layout.current.get(edge.source);
          const b = layout.current.get(edge.target);
          if (!a || !b) return null;
          const pair = [edge.source, edge.target].sort().join("\u0000");
          const index = parallel.get(pair) ?? 0;
          parallel.set(pair, index + 1);
          const count = pairSize.get(pair) ?? 1;
          // Several relations between the same two entities are drawn as separate curves.
          const bend = (index - (count - 1) / 2) * 26 * (edge.source < edge.target ? 1 : -1);
          const mx = (a.x + b.x) / 2;
          const my = (a.y + b.y) / 2;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const length = Math.max(Math.hypot(dx, dy), 1);
          const cx = mx + (-dy / length) * bend;
          const cy = my + (dx / length) * bend;
          const toward = (from: LayoutNode, ax: number, ay: number, gap: number) => {
            const ex = ax - from.x;
            const ey = ay - from.y;
            const el = Math.max(Math.hypot(ex, ey), 1);
            return { x: from.x + (ex / el) * gap, y: from.y + (ey / el) * gap };
          };
          const start = toward(a, cx, cy, a.r + 1);
          const end = toward(b, cx, cy, b.r + 3);
          const active = edge.source === focus || edge.target === focus;
          const chosen = selectedEdge && selectedEdge.source === edge.source && selectedEdge.target === edge.target && selectedEdge.label === edge.label;
          const d = `M${start.x} ${start.y} Q${cx} ${cy} ${end.x} ${end.y}`;
          const dim = focus && !active ? "opacity-25" : "";
          return (
            <g key={`${edge.source}|${edge.target}|${edge.label}`} className={dim}>
              <path d={d} fill="none" className={chosen ? "stroke-indigo-600" : active ? "stroke-indigo-400" : "stroke-slate-300"} strokeWidth={chosen ? 3 : active ? 2 : 1.3} markerEnd={`url(#${active || chosen ? "kg-arrow-on" : "kg-arrow"})`} />
              <path d={d} fill="none" stroke="transparent" strokeWidth={12} style={{ cursor: "pointer" }} onClick={() => onSelectEdge({ source: edge.source, target: edge.target, label: edge.label })}>
                <title>{`${byId.get(edge.source)?.name} ${relationText(edge.label)} ${byId.get(edge.target)?.name}${edge.count > 1 ? ` (${edge.count})` : ""}`}</title>
              </path>
              {(active || chosen || zoomedIn) && (
                <text x={cx / 2 + (a.x + b.x) / 4} y={cy / 2 + (a.y + b.y) / 4 - 3} textAnchor="middle" className="fill-slate-500 text-[9px]" style={{ paintOrder: "stroke", stroke: "white", strokeWidth: 3, pointerEvents: "none" }}>
                  {relationText(edge.label)}
                </text>
              )}
            </g>
          );
        })}
      </g>

      <g>
        {nodes.map((node) => {
          const place = layout.current.get(node.id);
          if (!place) return null;
          const style = entityStyle(node.type);
          const selected = node.id === selectedId;
          const dim = focus && node.id !== focus && !near?.has(node.id);
          const showLabel = selected || labelled.has(node.id) || zoomedIn || node.id === focus || Boolean(near?.has(node.id));
          return (
            <g
              key={node.id}
              transform={`translate(${place.x} ${place.y})`}
              tabIndex={0}
              role="button"
              aria-label={`${node.name}, ${style.label}, ${node.degree} connection${node.degree === 1 ? "" : "s"}`}
              aria-pressed={selected}
              className={`cursor-pointer outline-none focus-visible:[&>circle]:stroke-indigo-700 ${dim ? "opacity-30" : ""}`}
              onPointerDown={(event) => startDrag(event, node.id)}
              onPointerMove={moveDrag}
              onPointerUp={() => endDrag(node.id)}
              onPointerCancel={() => endDrag(node.id)}
              onPointerEnter={() => setHovered(node.id)}
              onPointerLeave={() => setHovered((h) => (h === node.id ? null : h))}
              onFocus={() => setHovered(node.id)}
              onBlur={() => setHovered((h) => (h === node.id ? null : h))}
              onKeyDown={(event) => onKey(event, node.id)}
            >
              <circle r={place.r} className={`${style.fill} ${selected ? "stroke-indigo-700" : "stroke-white"}`} strokeWidth={selected ? 3 : 2} />
              {showLabel && (
                <text y={place.r + 12} textAnchor="middle" className="fill-slate-800 text-[11px] font-medium" style={{ paintOrder: "stroke", stroke: "white", strokeWidth: 3, pointerEvents: "none" }}>
                  {node.name.length > 28 ? `${node.name.slice(0, 27)}…` : node.name}
                </text>
              )}
              <title>{`${node.name} · ${style.label}`}</title>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
