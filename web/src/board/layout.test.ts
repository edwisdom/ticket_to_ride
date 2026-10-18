import { describe, expect, it } from 'vitest';
import mini from '../../../ticket_to_ride/data/layout/mini.json';
import usa from '../../../ticket_to_ride/data/layout/usa.json';
import { joinBoard, layoutOnly, type EngineSegment, type Layout } from './layout';

const layouts = { usa: usa as unknown as Layout, mini: mini as unknown as Layout };
const POINTS = [1, 2, 4, 7, 10, 15];

describe('joinBoard', () => {
  it('matches every engine segment whatever the order, reversed or not', () => {
    for (const layout of Object.values(layouts)) {
      const engine: EngineSegment[] = layout.segments
        .map((s) => ({ a: s.cities[1], b: s.cities[0], length: s.cars.length, color: s.color }))
        .reverse();
      const model = joinBoard(layout, engine, POINTS);
      expect(model.segments).toHaveLength(engine.length);
      model.segments.forEach((seg, i) => {
        expect(seg.index).toBe(i);
        expect(seg.cars).toHaveLength(seg.length);
        expect(new Set([seg.a, seg.b])).toEqual(new Set([engine[i]!.a, engine[i]!.b]));
      });
    }
  });

  it('pairs the two tracks of each double route', () => {
    const model = layoutOnly(layouts.usa, POINTS);
    const doubles = model.segments.filter((s) => s.sibling !== null);
    expect(doubles.length % 2).toBe(0);
    for (const s of doubles) {
      const other = model.segments[s.sibling!]!;
      expect(other.sibling).toBe(s.index);
      expect(new Set([other.a, other.b])).toEqual(new Set([s.a, s.b]));
    }
    // The USA map has 100 segments, 22 of them in doubles (RULES.md).
    expect(model.segments).toHaveLength(100);
    expect(doubles).toHaveLength(44);
  });

  it('refuses a segment the layout does not draw', () => {
    const engine: EngineSegment[] = [{ a: 'Seattle', b: 'Miami', length: 6, color: 'red' }];
    expect(() => joinBoard(layouts.mini, engine, POINTS)).toThrow(/does not draw Seattle-Miami/);
  });

  it('refuses a layout with segments the engine lacks', () => {
    const [first, ...rest] = layouts.mini.segments;
    const engine = rest.map((s) => ({ a: s.cities[0], b: s.cities[1], length: s.cars.length, color: s.color }));
    expect(first).toBeDefined();
    expect(() => joinBoard(layouts.mini, engine, POINTS)).toThrow(/draws segments the engine lacks/);
  });
});
