// A small force-directed layout for the knowledge graph: no library, no worker (the Content-Security-Policy
// allows neither a CDN nor a blob worker). Pure functions on plain objects, so the layout can be
// checked under Node without a browser.

export const WIDTH = 1000;
export const HEIGHT = 640;

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  /** Held in place while the pointer drags it. */
  pinned: boolean;
}

export interface NodeSeed {
  id: string;
  degree: number;
}

export interface LinkSeed {
  source: string;
  target: string;
}

const MARGIN = 24;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;

/** Bigger circles for better connected entities, within a readable range. */
export function radius(degree: number): number {
  return 7 + Math.min(14, Math.sqrt(Math.max(0, degree)) * 3.2);
}

/**
 * Positions for `seeds`: a node that was already placed keeps its place, a new one starts next to
 * a neighbour that is placed (so adding documents does not scramble the picture), else on a circle.
 */
export function seedLayout(
  previous: Map<string, LayoutNode>,
  seeds: NodeSeed[],
  links: LinkSeed[],
): Map<string, LayoutNode> {
  const placed = new Map<string, LayoutNode>();
  const neighbours = new Map<string, string[]>();
  const remember = (from: string, to: string) => neighbours.set(from, [...(neighbours.get(from) ?? []), to]);
  for (const { source, target } of links) {
    remember(source, target);
    remember(target, source);
  }
  seeds.forEach((seed, index) => {
    const old = previous.get(seed.id);
    if (old) {
      placed.set(seed.id, { ...old, r: radius(seed.degree), vx: 0, vy: 0 });
      return;
    }
    const near = (neighbours.get(seed.id) ?? []).map((id) => placed.get(id) ?? previous.get(id)).find(Boolean);
    const angle = (index / Math.max(1, seeds.length)) * Math.PI * 2;
    const spread = Math.min(WIDTH, HEIGHT) * 0.35;
    placed.set(seed.id, {
      id: seed.id,
      x: near ? near.x + Math.cos(angle) * 40 : CENTER_X + Math.cos(angle) * spread,
      y: near ? near.y + Math.sin(angle) * 40 : CENTER_Y + Math.sin(angle) * spread,
      vx: 0,
      vy: 0,
      r: radius(seed.degree),
      pinned: false,
    });
  });
  return placed;
}

const REPULSION = 2200;
const LINK_STRENGTH = 0.07;
const GRAVITY = 0.012;
const DAMPING = 0.6;

/** Advance the layout one step. `alpha` (1 down to about 0) scales the forces as the picture cools. */
export function tick(nodes: LayoutNode[], links: [number, number][], alpha: number): void {
  const count = nodes.length;
  for (let i = 0; i < count; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < count; j++) {
      const b = nodes[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d2 = Math.max(dx * dx + dy * dy, 100);
      if (d2 > 160_000) continue; // beyond 400 px they do not feel each other
      const force = (REPULSION * alpha) / d2;
      a.vx -= dx * force;
      a.vy -= dy * force;
      b.vx += dx * force;
      b.vy += dy * force;
      const d = Math.sqrt(d2);
      const overlap = a.r + b.r + 8 - d;
      if (overlap > 0) {
        const push = (overlap / d) * 0.5;
        a.vx -= dx * push;
        a.vy -= dy * push;
        b.vx += dx * push;
        b.vy += dy * push;
      }
    }
  }
  for (const [i, j] of links) {
    const a = nodes[i];
    const b = nodes[j];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
    const rest = 70 + a.r + b.r;
    const pull = ((d - rest) / d) * LINK_STRENGTH;
    a.vx += dx * pull;
    a.vy += dy * pull;
    b.vx -= dx * pull;
    b.vy -= dy * pull;
  }
  for (const node of nodes) {
    if (node.pinned) {
      node.vx = node.vy = 0;
      continue;
    }
    node.vx += (CENTER_X - node.x) * GRAVITY;
    node.vy += (CENTER_Y - node.y) * GRAVITY;
    node.vx *= DAMPING;
    node.vy *= DAMPING;
    node.x = Math.min(WIDTH - MARGIN, Math.max(MARGIN, node.x + node.vx));
    node.y = Math.min(HEIGHT - MARGIN, Math.max(MARGIN, node.y + node.vy));
  }
}

/** Index pairs of the links whose two ends are both in `nodes` (self-links and repeats dropped). */
export function linkIndexes(nodes: LayoutNode[], links: LinkSeed[]): [number, number][] {
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const seen = new Set<string>();
  const pairs: [number, number][] = [];
  for (const { source, target } of links) {
    const a = index.get(source);
    const b = index.get(target);
    if (a === undefined || b === undefined || a === b) continue;
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pairs.push([a, b]);
  }
  return pairs;
}

export const COOLING = 0.985;
export const MIN_ALPHA = 0.03;

/** Run the whole simulation at once (reduced motion, and tests). */
export function settle(nodes: LayoutNode[], links: LinkSeed[], alpha = 1): void {
  const pairs = linkIndexes(nodes, links);
  let current = alpha;
  while (current > MIN_ALPHA) {
    tick(nodes, pairs, current);
    current *= COOLING;
  }
}
