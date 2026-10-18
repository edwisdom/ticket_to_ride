// Laying out a bot's reasoning: pure helpers, so the arithmetic is tested and the panel only
// draws.
import type { Candidate, Explanation, Glossary } from '../api/protocol';

/** A number as the panel shows it: whole when it is, else to two places, signed on request. */
export function num(v: number, signed = false): string {
  const r = Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2);
  return signed && v > 0 ? `+${r}` : r.replace('-', '−');
}

/** The candidates to show: the one played first, then the best of the rest by total, those
 * ruled out last. */
export function ranked(explanation: Explanation, limit = 6): Candidate[] {
  const chosen = explanation.candidates.find((c) => c.action.id === explanation.chosen);
  const rest = explanation.candidates
    .filter((c) => c !== chosen)
    .sort((a, b) => (b.total ?? -Infinity) - (a.total ?? -Infinity));
  return [...(chosen ? [chosen] : []), ...rest].slice(0, limit);
}

/** How many other candidates scored exactly what the move played did. */
export function ties(explanation: Explanation): number {
  const chosen = explanation.candidates.find((c) => c.action.id === explanation.chosen);
  if (chosen?.total == null) return 0;
  return explanation.candidates.filter((c) => c !== chosen && c.total === chosen.total).length;
}

/** Where zero sits and how wide a point is, for drawing every candidate's terms as bars on
 * one scale (0-100): at the left when nothing is negative, the right when nothing is positive,
 * else in the middle. */
export function axis(candidates: Candidate[]): { zero: number; unit: number } {
  let pos = 0;
  let neg = 0;
  for (const c of candidates) {
    let p = 0;
    let n = 0;
    for (const t of c.terms) {
      if (t.value > 0) p += t.value;
      else n -= t.value;
    }
    pos = Math.max(pos, p);
    neg = Math.max(neg, n);
  }
  if (!pos && !neg) return { zero: 0, unit: 100 };
  const zero = (100 * neg) / (pos + neg);
  return { zero, unit: 100 / (pos + neg) };
}

/** What a name means, from whichever part of the glossary has it. */
export function gloss(glossary: Glossary | null, name: string): string {
  if (!glossary) return name;
  return glossary.terms[name] ?? glossary.facts[name] ?? glossary.context[name] ?? glossary.rules[name] ?? name;
}

/** "plan_car_value" -> "plan car value". */
export const words = (name: string) => name.replaceAll('_', ' ');
