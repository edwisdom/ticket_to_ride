import { describe, expect, it } from 'vitest';
import type { Candidate, Explanation } from '../api/protocol';
import { axis, gloss, num, ranked, ties, words } from './explain';

const candidate = (id: number, total: number | null, terms: [string, number][] = []): Candidate => ({
  action: { kind: 'pass', id, label: `move ${id}` },
  total,
  rule: null,
  terms: terms.map(([name, value]) => ({ name, value })),
  facts: [],
});

const explanation = (chosen: number, candidates: Candidate[]): Explanation => ({
  agent: 'h3',
  tier: 'h3',
  chosen,
  rule: null,
  context: [],
  candidates,
});

describe('ranked', () => {
  it('puts the move played first, then the best totals, the ruled-out last', () => {
    const e = explanation(2, [candidate(1, 3), candidate(2, 1), candidate(3, null), candidate(4, 5)]);
    expect(ranked(e).map((c) => c.action.id)).toEqual([2, 4, 1, 3]);
    expect(ranked(e, 2).map((c) => c.action.id)).toEqual([2, 4]);
  });
});

describe('ties', () => {
  it('counts the others that scored what the move played did', () => {
    expect(ties(explanation(1, [candidate(1, 0), candidate(2, 0), candidate(3, 0), candidate(4, -1)]))).toBe(2);
    expect(ties(explanation(1, [candidate(1, 2), candidate(2, 1)]))).toBe(0);
    expect(ties(explanation(1, [candidate(1, null), candidate(2, null)]))).toBe(0);
  });
});

describe('axis', () => {
  it('puts zero where the bars need it', () => {
    // Gains only: zero at the left, the widest candidate filling the width.
    expect(axis([candidate(1, 0, [['a', 2], ['b', 2]]), candidate(2, 0, [['a', 1]])])).toEqual({ zero: 0, unit: 25 });
    // Both: zero splits the width between the largest gain and the largest cost.
    expect(axis([candidate(1, 0, [['a', 3], ['b', -1]])])).toEqual({ zero: 25, unit: 25 });
    expect(axis([candidate(1, null)])).toEqual({ zero: 0, unit: 100 });
  });
});

describe('num, words and gloss', () => {
  it('formats', () => {
    expect(num(1.7530642)).toBe('1.75');
    expect(num(-0.05)).toBe('−0.05');
    expect(num(15)).toBe('15');
    expect(num(2.5, true)).toBe('+2.50');
    expect(words('plan_car_value')).toBe('plan car value');
  });

  it('looks a name up in every part of the glossary', () => {
    const g = { terms: { a: 'term a' }, facts: { b: 'fact b' }, context: { c: 'context c' }, rules: { d: 'rule d' } };
    expect(['a', 'b', 'c', 'd', 'e'].map((n) => gloss(g, n))).toEqual(['term a', 'fact b', 'context c', 'rule d', 'e']);
    expect(gloss(null, 'x')).toBe('x');
  });
});
