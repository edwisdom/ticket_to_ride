import { describe, expect, it } from 'vitest';
import { carEnds, couplings, plaqueRect, scoreMarkers } from './geometry';
import type { Car } from './layout';

describe('couplings', () => {
  it('joins the facing ends of consecutive cars', () => {
    const cars: Car[] = [
      [0, 0, 0],
      [50, 0, 0],
      [100, 0, 90],
    ];
    const [first, second] = couplings(cars, 40);
    expect(first).toEqual([20, 0, 30, 0]);
    // The third car is vertical: its nearer end is either of (100, +-20).
    expect(second![0]).toBeCloseTo(70);
    expect(Math.abs(second![3])).toBeCloseTo(20);
  });

  it('has one coupling fewer than cars', () => {
    expect(couplings([[0, 0, 0]], 40)).toEqual([]);
    expect(couplings([], 40)).toEqual([]);
  });
});

describe('carEnds', () => {
  it('spans the car along its angle', () => {
    const [p, q] = carEnds([10, 10, 90], 20);
    expect(p[0]).toBeCloseTo(10);
    expect(p[1]).toBeCloseTo(0);
    expect(q[1]).toBeCloseTo(20);
  });
});

describe('plaqueRect', () => {
  it('aligns the plaque on its anchor', () => {
    // Anchored at its left middle, east of the city.
    expect(plaqueRect([100, 50, 0, 0.5], 40, 13, [4, 2.5])).toEqual({ x: 100, y: 41, w: 48, h: 18 });
    // Anchored at its bottom centre, north of the city.
    expect(plaqueRect([100, 50, 0.5, 1], 40, 13, [4, 2.5])).toEqual({ x: 76, y: 32, w: 48, h: 18 });
  });
});

describe('scoreMarkers', () => {
  const track = Array.from({ length: 100 }, (_, i) => [i * 10, 0] as [number, number]);

  it('puts each seat on its score, wrapping past 99', () => {
    const [a, b] = scoreMarkers([3, 104], track, 10);
    expect(a).toMatchObject({ seat: 0, x: 35, y: 5 });
    expect(b).toMatchObject({ seat: 1, x: 45, y: 5 });
  });

  it('fans out seats on the same square', () => {
    const markers = scoreMarkers([7, 7, 7], track, 10);
    expect(new Set(markers.map((m) => m.x)).size).toBe(3);
    expect(markers.map((m) => m.seat)).toEqual([0, 1, 2]);
  });
});
