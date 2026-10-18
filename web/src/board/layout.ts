// The board's geometry, as tools/gen_layout.py writes it (ticket_to_ride/data/layout/*.json)
// and /api/boards/{map} serves it, joined with the engine's own board data.

export type CardColor = 'black' | 'blue' | 'green' | 'orange' | 'purple' | 'red' | 'white' | 'yellow';
export type RouteColor = CardColor | 'gray';

/** A car space: centre x, centre y, and the angle of its long axis in degrees. */
export type Car = [number, number, number];

export interface LayoutCity {
  name: string;
  x: number;
  y: number;
  /** Plaque anchor x, y and alignment: the fraction of the plaque left of and above the anchor. */
  label: [number, number, number, number];
}

export interface LayoutSegment {
  cities: [string, string];
  color: RouteColor;
  /** The car spaces' length: it flexes with the room the route has. */
  length: number;
  cars: Car[];
}

export interface Layout {
  map: string;
  size: [number, number];
  square: number;
  car_width: number;
  city_radius: number;
  label_size: number;
  label_pad: [number, number];
  /** Letter spacing on the plaques, in ems. */
  label_tracking: number;
  /** The score track's 100 squares, top-left corners, clockwise from the top-left corner. */
  track: [number, number][];
  decor: {
    /** Centre and radius. */
    compass?: [number, number, number];
    /** Top-left corner and scale. */
    legend?: [number, number, number];
    /** Lettering on the water: text, x, y, size, degrees. */
    water?: [string, number, number, number, number][];
  };
  cities: LayoutCity[];
  segments: LayoutSegment[];
  basemap: {
    land: string;
    lakes: string;
    rivers: string;
    borders: string;
    states: string;
    /** Mountain glyphs: x, y, scale. */
    peaks: [number, number, number][];
    /** A coarse coastline for the tickets' small maps. */
    thumb: string;
  };
}

/** What the engine says about a segment; its index is the engine's segment index. */
export interface EngineSegment {
  a: string;
  b: string;
  length: number;
  color: RouteColor;
}

export interface Segment extends EngineSegment {
  index: number;
  cars: Car[];
  /** Length of each car space. */
  carLength: number;
  /** The other track of a double route, if any. */
  sibling: number | null;
}

export interface BoardModel {
  map: string;
  layout: Layout;
  /** In engine order. */
  segments: Segment[];
  cities: Map<string, LayoutCity>;
  /** routePoints[n - 1] scores a route of length n. */
  routePoints: number[];
}

const pairKey = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);

/** Joins the engine's segments to the layout's tracks by city pair and colour, so neither
 * side's ordering matters. A segment the layout does not draw is a stale layout: it throws. */
export function joinBoard(layout: Layout, segments: EngineSegment[], routePoints: number[]): BoardModel {
  const pool = new Map<string, LayoutSegment[]>();
  for (const s of layout.segments) {
    const key = `${pairKey(...s.cities)}|${s.color}`;
    pool.set(key, [...(pool.get(key) ?? []), s]);
  }
  const byPair = new Map<string, number[]>();
  const joined = segments.map((seg, index): Segment => {
    const key = `${pairKey(seg.a, seg.b)}|${seg.color}`;
    const drawn = pool.get(key)?.shift();
    if (!drawn || drawn.cars.length !== seg.length) {
      throw new Error(`layout ${layout.map} does not draw ${seg.a}-${seg.b} (${seg.color}, ${seg.length})`);
    }
    const pair = pairKey(seg.a, seg.b);
    byPair.set(pair, [...(byPair.get(pair) ?? []), index]);
    return { ...seg, index, cars: drawn.cars, carLength: drawn.length, sibling: null };
  });
  for (const idxs of byPair.values()) {
    if (idxs.length === 2) {
      const [i, j] = idxs as [number, number];
      joined[i]!.sibling = j;
      joined[j]!.sibling = i;
    }
  }
  const left = [...pool.values()].flat();
  if (left.length) throw new Error(`layout ${layout.map} draws segments the engine lacks: ${left.length}`);
  return {
    map: layout.map,
    layout,
    segments: joined,
    cities: new Map(layout.cities.map((c) => [c.name, c])),
    routePoints,
  };
}

/** A model straight from the layout, in its own order: for drawing a board with no game. */
export function layoutOnly(layout: Layout, routePoints: number[]): BoardModel {
  const segments = layout.segments.map((s) => ({
    a: s.cities[0],
    b: s.cities[1],
    length: s.cars.length,
    color: s.color,
  }));
  return joinBoard(layout, segments, routePoints);
}
