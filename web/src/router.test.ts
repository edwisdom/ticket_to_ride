import { describe, expect, it } from 'vitest';
import { href, parse } from './router';

describe('parse', () => {
  it('reads every route', () => {
    expect(parse('')).toEqual({ name: 'lobby' });
    expect(parse('#/')).toEqual({ name: 'lobby' });
    expect(parse('#/game/abc123')).toEqual({ name: 'game', id: 'abc123' });
    expect(parse('#/review/abc123')).toEqual({ name: 'review', id: 'abc123', at: null });
    expect(parse('#/review/abc123?at=34')).toEqual({ name: 'review', id: 'abc123', at: 34 });
    expect(parse('#/review/abc123?at=x')).toEqual({ name: 'review', id: 'abc123', at: null });
    expect(parse('#/profile')).toEqual({ name: 'profile', id: null });
    expect(parse('#/profile/p-1')).toEqual({ name: 'profile', id: 'p-1' });
    expect(parse('#/leaderboard?map=mini')).toEqual({ name: 'leaderboard', map: 'mini' });
    expect(parse('#/dev?map=mini')).toEqual({ name: 'dev', map: 'mini' });
  });

  it('falls back on an unknown map and an unknown path', () => {
    expect(parse('#/leaderboard?map=europe')).toEqual({ name: 'leaderboard', map: 'usa' });
    expect(parse('#/nowhere')).toEqual({ name: 'missing', path: '/nowhere' });
    expect(parse('#/game')).toEqual({ name: 'missing', path: '/game' });
  });

  it('round-trips the links it makes', () => {
    expect(parse(href.game('a b'))).toEqual({ name: 'game', id: 'a b' });
    expect(parse(href.review('x'))).toEqual({ name: 'review', id: 'x', at: null });
    expect(parse(href.review('x', 7))).toEqual({ name: 'review', id: 'x', at: 7 });
    expect(parse(href.profile('p'))).toEqual({ name: 'profile', id: 'p' });
    expect(parse(href.leaderboard('mini'))).toEqual({ name: 'leaderboard', map: 'mini' });
  });
});
