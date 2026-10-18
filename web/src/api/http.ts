// The REST half of the protocol (docs/WEB_PROTOCOL.md). Every call goes through `call`, so an
// error always arrives as an HttpError carrying the server's `{code, message}`.
import type {
  Agent,
  Board,
  GameSummary,
  Glossary,
  Leaderboard,
  MapName,
  Meta,
  NewGame,
  Player,
  Profile,
  Review,
} from './protocol';

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: init?.body ? { 'content-type': 'application/json' } : undefined,
  });
  if (!res.ok) {
    let code = String(res.status);
    let message = res.statusText;
    try {
      const detail = (await res.json())?.detail;
      if (detail && typeof detail === 'object' && 'code' in detail) {
        ({ code, message } = detail);
      } else if (detail) {
        message = typeof detail === 'string' ? detail : JSON.stringify(detail);
      }
    } catch {
      // Not JSON: keep the status line.
    }
    throw new HttpError(res.status, code, message);
  }
  return (await res.json()) as T;
}

const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
const q = encodeURIComponent;

export const api = {
  meta: () => call<Meta>('/meta'),
  board: (map: MapName) => call<Board>(`/boards/${map}`),
  glossary: () => call<Glossary>('/glossary'),
  agents: () => call<Agent[]>('/agents'),
  newPlayer: (name: string) => call<Player>('/players', post({ name })),
  profile: (id: string) => call<Profile>(`/players/${q(id)}`),
  leaderboard: (map: MapName) => call<Leaderboard>(`/leaderboard?map=${map}`),
  newGame: (body: NewGame) => call<GameSummary>('/games', post(body)),
  games: (playerId: string) => call<GameSummary[]>(`/games?player_id=${q(playerId)}`),
  game: (id: string) => call<GameSummary>(`/games/${q(id)}`),
  review: (id: string) => call<Review>(`/games/${q(id)}/review`),
};
