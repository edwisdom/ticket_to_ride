import { describe, expect, it } from 'vitest';
import mini from '../../../ticket_to_ride/data/layout/mini.json';
import usa from '../../../ticket_to_ride/data/layout/usa.json';
import { fromServer } from './boards';
import miniBoard from './fixtures/board-mini.json';
import usaBoard from './fixtures/board-usa.json';
import type { Board } from './protocol';

// The fixtures are GET /api/boards/{map} with the layout left out; it is attached here from
// the checked-in layouts, as the server does.
const cases = [
  { board: miniBoard as unknown as Board, layout: mini },
  { board: usaBoard as unknown as Board, layout: usa },
];

describe('fromServer', () => {
  it('draws every engine segment at its own index', () => {
    for (const { board, layout } of cases) {
      const model = fromServer({ ...board, layout });
      expect(model.segments).toHaveLength(board.segments.length);
      for (const s of board.segments) {
        const seg = model.segments[s.id]!;
        expect(new Set([seg.a, seg.b])).toEqual(new Set([board.cities[s.a], board.cities[s.b]]));
        expect(seg.color).toBe(s.color === null ? 'gray' : board.colors[s.color]);
        expect(seg.cars).toHaveLength(s.length);
        expect(seg.sibling).toBe(s.sibling);
      }
    }
  });

  it('takes card kinds, tickets and route points in the engine order', () => {
    const model = fromServer({ ...cases[0]!.board, layout: mini });
    expect(model.cardKinds).toEqual(['black', 'blue', 'green', 'orange', 'red', 'yellow', 'locomotive']);
    expect(model.routePoints).toEqual([1, 2, 4, 7, 10]);
    const t = cases[0]!.board.tickets[0]!;
    expect(model.tickets[0]).toEqual({ a: model.cityNames[t.a], b: model.cityNames[t.b], points: t.points });
  });

  it('refuses a board without a layout', () => {
    expect(() => fromServer(cases[0]!.board)).toThrow(/no layout for mini/);
  });
});
