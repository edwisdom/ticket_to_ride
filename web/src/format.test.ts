import { describe, expect, it } from 'vitest';
import type { GameSummary } from './api/protocol';
import { elo, outcome, plural, versus, when } from './format';

const rating = (v: number, lo: number | null, hi: number | null) => ({ elo: v, lo, hi, games: 10, score: 0.5 });

describe('elo', () => {
  it('signs the rating and its interval', () => {
    expect(elo(rating(21.4, 8, 33))).toBe('+21 [+8, +33]');
    expect(elo(rating(-100, -119, -82))).toBe('−100 [−119, −82]');
    expect(elo(rating(0, null, null))).toBe('0');
    expect(elo(undefined)).toBe('unrated');
  });
});

describe('plural', () => {
  it('agrees with the count', () => {
    expect(plural(1, 'point')).toBe('1 point');
    expect(plural(0, 'point')).toBe('0 points');
    expect(plural(2, 'locomotive')).toBe('2 locomotives');
  });
});

describe('when', () => {
  const now = new Date('2026-10-17T12:00:00Z');
  it('says how long ago', () => {
    expect(when('2026-10-17T11:59:30Z', now)).toBe('just now');
    expect(when('2026-10-17T11:45:00Z', now)).toBe('15 min ago');
    expect(when('2026-10-17T09:00:00Z', now)).toBe('3 h ago');
    expect(when('2026-10-16T09:00:00Z', now)).toBe('yesterday');
  });
});

const game = (scores: [number | null, number | null], human: number | null): GameSummary => ({
  id: 'g',
  map: 'usa',
  mode: human === null ? 'spectate' : 'play',
  rated: true,
  assist: 'parity-1',
  status: scores[0] === null ? 'active' : 'finished',
  n_players: 2,
  seats: [
    { seat: 0, player: { id: 'p', kind: human === 0 ? 'human' : 'bot', name: human === 0 ? 'Ada' : 'H2', spec: null }, score: scores[0], rank: null },
    { seat: 1, player: { id: 'b', kind: 'bot', name: 'H3', spec: 'h3' }, score: scores[1], rank: null },
  ],
  human_seat: human,
  ply: 0,
  created: '',
  finished: null,
});

describe('versus and outcome', () => {
  it('names the opponent, or both bots', () => {
    expect(versus(game([null, null], 0))).toBe('vs H3');
    expect(versus(game([null, null], null))).toBe('H2 vs H3');
  });

  it("gives the person's result", () => {
    expect(outcome(game([112, 98], 0))).toBe('Won 112–98');
    expect(outcome(game([90, 98], 0))).toBe('Lost 90–98');
    expect(outcome(game([98, 98], 0))).toBe('Drew 98–98');
    expect(outcome(game([null, null], 0))).toBeNull();
    expect(outcome(game([1, 2], null))).toBeNull();
  });
});
