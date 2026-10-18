import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Action, ClientMessage, GameSummary, LogEntry, ServerMessage, View } from '../api/protocol';
import { ANSWER_MS, GameSession, SPEEDS } from './session.svelte';

const draw = (id: number): Action => ({ kind: 'draw', id, slot: null, card: null, label: 'Draw from the deck' });

function view(ply: number, current: number): View {
  return {
    viewer: 0,
    ply,
    turn: ply,
    phase: 'main',
    current,
    draws_left: 2,
    final_turns: null,
    owners: [],
    closed: [],
    faceup: [],
    deck: 100,
    discard: [],
    ticket_deck: 20,
    seats: [],
    offer: null,
    unseen: null,
    deck_cards: null,
  };
}

const summary: GameSummary = {
  id: 'g1',
  map: 'mini',
  mode: 'play',
  rated: true,
  assist: 'parity-1',
  status: 'active',
  n_players: 2,
  seats: [],
  human_seat: 0,
  ply: 0,
  created: '',
  finished: null,
};

const entry = (ply: number, seat: number): LogEntry => ({ ply, seat, action: draw(1), text: `move ${ply}` });

function event(ply: number, seat: number, legal: Action[] = []): ServerMessage {
  return { type: 'event', entry: entry(ply, seat), view: view(ply + 1, legal.length ? 0 : 1), legal, explanation: null, think_ms: 3 };
}

describe('GameSession', () => {
  let sent: ClientMessage[];
  let closed: boolean;
  let reconnects: number;
  let session: GameSession;
  let deliver: (msg: ServerMessage) => void;

  beforeEach(() => {
    vi.useFakeTimers();
    sent = [];
    closed = false;
    reconnects = 0;
    session = new GameSession('g1', (_id, onMessage) => {
      deliver = onMessage;
      return {
        send: (msg) => (sent.push(msg), true),
        reconnect: () => reconnects++,
        close: () => (closed = true),
      };
    });
    session.setSpeed('normal');
    deliver({ type: 'snapshot', game: summary, view: view(0, 0), legal: [draw(5)], log: [] });
  });

  afterEach(() => vi.useRealTimers());

  it('starts from the snapshot', () => {
    expect(session.view?.ply).toBe(0);
    expect(session.legal.map((a) => a.id)).toEqual([5]);
    expect(session.viewer).toBe(0);
  });

  it('shows the bot moves one at a time, at the chosen pace', () => {
    deliver(event(1, 1));
    deliver(event(2, 1));
    deliver(event(3, 1, [draw(7)]));
    expect(session.log.map((e) => e.ply)).toEqual([1]);
    expect(session.behind).toBe(2);
    // The viewer's move arrives last, but it cannot be played before it is on screen.
    expect(session.legal).toEqual([]);
    vi.advanceTimersByTime(SPEEDS.normal);
    expect(session.log.map((e) => e.ply)).toEqual([1, 2]);
    vi.advanceTimersByTime(SPEEDS.normal);
    expect(session.log.map((e) => e.ply)).toEqual([1, 2, 3]);
    expect(session.legal.map((a) => a.id)).toEqual([7]);
    expect(session.behind).toBe(0);
  });

  it("holds the viewer's own move only briefly", () => {
    deliver(event(0, 0));
    deliver(event(1, 1));
    expect(session.log).toHaveLength(1);
    vi.advanceTimersByTime(400);
    expect(session.log).toHaveLength(2);
  });

  it('sends a legal action at the ply on screen, once', () => {
    session.act(draw(9));
    expect(sent).toEqual([]);
    session.act(draw(5));
    session.act(draw(5));
    expect(sent).toEqual([{ type: 'act', ply: 0, action: 5 }]);
    expect(session.sent).toEqual({ ply: 0, action: 5 });
    deliver({ type: 'ack', ply: 0, action: 5, duplicate: false });
    expect(session.sent).not.toBeNull();
    deliver(event(0, 0));
    expect(session.sent).toBeNull();
  });

  it('shows the end after the last move, and hangs up', () => {
    deliver(event(1, 1));
    deliver({ type: 'over', game: { ...summary, status: 'finished' }, view: view(2, 0), scores: [], seed: '42' });
    expect(session.over).toBeNull();
    vi.advanceTimersByTime(SPEEDS.normal);
    expect(session.over?.seed).toBe('42');
    expect(session.summary?.status).toBe('finished');
    expect(closed).toBe(true);
  });

  it('starts over from a reconnection snapshot', () => {
    deliver(event(1, 1));
    deliver(event(2, 1));
    deliver({ type: 'snapshot', game: summary, view: view(5, 0), legal: [draw(8)], log: [entry(4, 1)] });
    vi.advanceTimersByTime(10 * SPEEDS.normal);
    expect(session.view?.ply).toBe(5);
    expect(session.log.map((e) => e.ply)).toEqual([4]);
    expect(session.legal.map((a) => a.id)).toEqual([8]);
  });

  it('pauses, and steps while paused', () => {
    session.togglePause();
    deliver(event(1, 1));
    deliver(event(2, 1));
    expect(session.log).toHaveLength(0);
    session.step();
    expect(session.log.map((e) => e.ply)).toEqual([1]);
    session.togglePause();
    expect(session.log.map((e) => e.ply)).toEqual([1, 2]);
  });

  it('reconnects when a move goes unanswered, and not when it is answered', () => {
    session.act(draw(5));
    vi.advanceTimersByTime(ANSWER_MS);
    expect(reconnects).toBe(1);
    deliver({ type: 'snapshot', game: summary, view: view(0, 0), legal: [draw(5)], log: [] });
    session.act(draw(5));
    deliver(event(0, 0));
    vi.advanceTimersByTime(ANSWER_MS);
    expect(reconnects).toBe(1);
  });

  it('reports an error and lets the viewer try again', () => {
    session.act(draw(5));
    deliver({ type: 'error', code: 'illegal_action', message: 'no such move', ply: 0 });
    expect(session.error).toBe('no such move');
    expect(session.sent).toBeNull();
  });
});
