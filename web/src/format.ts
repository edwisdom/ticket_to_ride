import type { GameSummary, MapRating, Rating } from './api/protocol';

/** "1 point", "2 points". */
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Whole, with a typographic minus: ratings are absolute, so no plus sign. */
const whole = (v: number) => `${Math.round(v)}`.replace('-', '−');

/** A rating on the H3-anchored scale, with its interval when there is one. */
export function elo(rating: Rating | null | undefined): string {
  if (!rating) return 'unrated';
  const lo = rating.lo;
  const hi = rating.hi;
  return lo === null || hi === null ? whole(rating.elo) : `${whole(rating.elo)} [${whole(lo)}, ${whole(hi)}]`;
}

export function ratingOn(ratings: MapRating[], map: string): Rating | undefined {
  return ratings.find((r) => r.map === map)?.rating;
}

export const MAP_NAMES: Record<string, string> = { usa: 'USA', mini: 'Mini' };

/** "3 min ago", "yesterday", or the date. */
export function when(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  const s = (now.getTime() - then.getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 2 * 86400) return 'yesterday';
  return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** Who the game is between, as the lobby lists it: the person's opponent, or both bots. */
export function versus(game: GameSummary): string {
  if (game.human_seat !== null) {
    const them = game.seats.filter((s) => s.seat !== game.human_seat).map((s) => s.player.name);
    return `vs ${them.join(', ')}`;
  }
  return game.seats.map((s) => s.player.name).join(' vs ');
}

/** The person's result in a finished game: "Won 112–98". */
export function outcome(game: GameSummary): string | null {
  if (game.status !== 'finished' || game.human_seat === null) return null;
  const mine = game.seats.find((s) => s.seat === game.human_seat);
  const best = Math.max(...game.seats.filter((s) => s.seat !== game.human_seat).map((s) => s.score ?? 0));
  if (!mine || mine.score === null) return null;
  const word = mine.score > best ? 'Won' : mine.score < best ? 'Lost' : 'Drew';
  return `${word} ${mine.score}–${best}`;
}
