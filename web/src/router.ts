// Hash routes: the server serves index.html for any path, but a hash needs nothing from it.
import type { MapName } from './api/protocol';

export type Route =
  | { name: 'lobby' }
  | { name: 'game'; id: string }
  | { name: 'review'; id: string; at: number | null }
  | { name: 'profile'; id: string | null }
  | { name: 'leaderboard'; map: MapName }
  | { name: 'dev'; map: string }
  | { name: 'missing'; path: string };

const MAPS: readonly MapName[] = ['usa', 'mini'];

export function parse(hash: string): Route {
  const [path = '', query = ''] = hash.replace(/^#/, '').split('?');
  const params = new URLSearchParams(query);
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  const [head, arg] = parts;
  if (!head) return { name: 'lobby' };
  if (head === 'game' && arg) return { name: 'game', id: arg };
  if (head === 'review' && arg) {
    const at = Number(params.get('at'));
    return { name: 'review', id: arg, at: params.has('at') && Number.isInteger(at) && at >= 0 ? at : null };
  }
  if (head === 'profile') return { name: 'profile', id: arg ?? null };
  if (head === 'leaderboard') {
    const map = params.get('map');
    return { name: 'leaderboard', map: MAPS.includes(map as MapName) ? (map as MapName) : 'usa' };
  }
  if (head === 'dev') return { name: 'dev', map: params.get('map') ?? 'usa' };
  return { name: 'missing', path };
}

const q = encodeURIComponent;

export const href = {
  lobby: () => '#/',
  game: (id: string) => `#/game/${q(id)}`,
  review: (id: string, at?: number) => `#/review/${q(id)}${at === undefined ? '' : `?at=${at}`}`,
  profile: (id?: string) => (id ? `#/profile/${q(id)}` : '#/profile'),
  leaderboard: (map: MapName = 'usa') => `#/leaderboard?map=${map}`,
};
