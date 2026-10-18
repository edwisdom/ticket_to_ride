import type { Car } from './layout';

/** The two ends of a car space's long axis. */
export function carEnds([x, y, deg]: Car, length: number): [[number, number], [number, number]] {
  const r = (deg * Math.PI) / 180;
  const dx = (Math.cos(r) * length) / 2;
  const dy = (Math.sin(r) * length) / 2;
  return [
    [x - dx, y - dy],
    [x + dx, y + dy],
  ];
}

/** Couplings between consecutive cars of a train: the closest pair of ends of each two. */
export function couplings(cars: Car[], length: number): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  for (let i = 0; i + 1 < cars.length; i++) {
    const a = carEnds(cars[i]!, length);
    const b = carEnds(cars[i + 1]!, length);
    let best: [number, number, number, number] = [...a[0], ...b[0]];
    let bestD = Infinity;
    for (const p of a) {
      for (const q of b) {
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < bestD) {
          bestD = d;
          best = [p[0], p[1], q[0], q[1]];
        }
      }
    }
    out.push(best);
  }
  return out;
}

/** Where a plaque goes: its anchor and alignment from the layout, its size from the text. */
export function plaqueRect(
  label: [number, number, number, number],
  textWidth: number,
  size: number,
  pad: [number, number],
): { x: number; y: number; w: number; h: number } {
  const [lx, ly, ax, ay] = label;
  const w = textWidth + 2 * pad[0];
  const h = size + 2 * pad[1];
  return { x: lx - ax * w, y: ly - ay * h, w, h };
}

/** Markers on the score track: seats sharing a square fan out instead of hiding each other. */
export function scoreMarkers(
  scores: number[],
  track: [number, number][],
  square: number,
): { seat: number; x: number; y: number }[] {
  const bySquare = new Map<number, number[]>();
  scores.forEach((score, seat) => {
    const sq = ((score % track.length) + track.length) % track.length;
    bySquare.set(sq, [...(bySquare.get(sq) ?? []), seat]);
  });
  const out: { seat: number; x: number; y: number }[] = [];
  for (const [sq, seats] of bySquare) {
    const [x, y] = track[sq]!;
    seats.forEach((seat, k) => {
      const spread = (k - (seats.length - 1) / 2) * (square * 0.24);
      out.push({ seat, x: x + square / 2 + spread, y: y + square / 2 - spread * 0.4 });
    });
  }
  return out.sort((a, b) => a.seat - b.seat);
}
