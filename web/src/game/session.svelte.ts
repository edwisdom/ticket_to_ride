// One game as the client sees it: the socket, and the position on screen.
//
// Moves stream in as fast as the bots make them; the session shows them at the viewer's
// chosen pace, so `view`, `log` and `legal` describe what is on screen, which may trail
// what has arrived. `legal` is non-empty only when the viewer is to move at the position
// shown, so nothing can be played against a position the viewer has not seen.
import { GameSocket, type SocketStatus } from '../api/socket';
import type {
  Action,
  ClientMessage,
  Event as GameEvent,
  Explanation,
  GameSummary,
  LogEntry,
  Over,
  ServerMessage,
  View,
} from '../api/protocol';

export type Speed = 'slow' | 'normal' | 'fast' | 'instant';

/** How long each speed holds a move on screen before showing the next, in ms. */
export const SPEEDS: Record<Speed, number> = { slow: 1800, normal: 900, fast: 350, instant: 0 };

/** A move stays on screen for the chosen hold, or briefly when it was the viewer's own: they
 * know what they did, but should see it land before the reply. */
export function holdFor(entry: LogEntry, viewer: number | null, hold: number): number {
  return entry.seat === viewer ? Math.min(hold, 400) : hold;
}

export interface Connection {
  send(msg: ClientMessage): boolean;
  reconnect(): void;
  close(): void;
}

/** How long a move may go unanswered before the session reconnects for a fresh snapshot;
 * `act` is idempotent on its ply, so a move that did land is not played twice. */
export const ANSWER_MS = 5000;

type Connect = (
  id: string,
  onMessage: (msg: ServerMessage) => void,
  onStatus: (status: SocketStatus) => void,
) => Connection;

const SPEED_KEY = 'ttr.speed';

function storedSpeed(): Speed {
  try {
    const s = localStorage.getItem(SPEED_KEY);
    if (s && s in SPEEDS) return s as Speed;
  } catch {
    // No storage: the default.
  }
  return 'normal';
}

export class GameSession {
  summary = $state<GameSummary | null>(null);
  view = $state<View | null>(null);
  legal = $state<Action[]>([]);
  log = $state<LogEntry[]>([]);
  /** The last move shown. */
  last = $state<LogEntry | null>(null);
  /** The latest bot decision shown, and its reasoning (spectating). */
  explanation = $state<{ entry: LogEntry; explanation: Explanation; thinkMs: number } | null>(null);
  over = $state<Over | null>(null);
  status = $state<SocketStatus>('connecting');
  error = $state<string | null>(null);
  /** The viewer's move in flight: sent, its event not yet arrived. */
  sent = $state<{ ply: number; action: number } | null>(null);
  speed = $state<Speed>(storedSpeed());
  paused = $state(false);
  /** Moves arrived but not yet shown. */
  behind = $state(0);

  private queue: (GameEvent | Over)[] = [];
  private timer: ReturnType<typeof setTimeout> | undefined;
  private watchdog: ReturnType<typeof setTimeout> | undefined;
  private connection: Connection;

  constructor(
    readonly id: string,
    connect: Connect = (id, onMessage, onStatus) => new GameSocket(id, onMessage, onStatus),
  ) {
    this.connection = connect(
      id,
      (msg) => this.receive(msg),
      (status) => (this.status = status),
    );
  }

  /** The seat whose hidden information this viewer sees; null when it sees everything. */
  get viewer(): number | null {
    return this.summary?.human_seat ?? null;
  }

  get spectating(): boolean {
    return this.summary?.mode === 'spectate';
  }

  receive(msg: ServerMessage) {
    switch (msg.type) {
      case 'snapshot':
        // Every (re)connection starts over from where the game stands.
        clearTimeout(this.timer);
        this.timer = undefined;
        this.queue = [];
        this.behind = 0;
        this.summary = msg.game;
        this.view = msg.view;
        this.legal = msg.legal;
        this.log = msg.log;
        this.last = msg.log.at(-1) ?? null;
        this.sent = null;
        this.over = null;
        break;
      case 'event':
      case 'over':
        this.queue.push(msg);
        this.behind = this.queue.length;
        this.pump();
        break;
      case 'ack':
        if (msg.duplicate && this.sent?.ply === msg.ply) this.sent = null;
        break;
      case 'error':
        this.error = msg.message;
        this.sent = null;
        if (msg.code === 'not_found') this.connection.close();
        break;
    }
  }

  /** Play one of `legal`. */
  act(action: Action) {
    if (!this.view || this.sent || !this.legal.some((a) => a.id === action.id)) return;
    this.error = null;
    if (this.connection.send({ type: 'act', ply: this.view.ply, action: action.id })) {
      this.sent = { ply: this.view.ply, action: action.id };
      clearTimeout(this.watchdog);
      this.watchdog = setTimeout(() => this.sent && this.connection.reconnect(), ANSWER_MS);
    } else {
      this.error = 'Not connected; reconnecting.';
    }
  }

  setSpeed(speed: Speed) {
    this.speed = speed;
    try {
      localStorage.setItem(SPEED_KEY, speed);
    } catch {
      // No storage: this page only.
    }
  }

  togglePause() {
    this.paused = !this.paused;
    this.pump();
  }

  /** Show the next move now, paused or not. */
  step() {
    clearTimeout(this.timer);
    this.timer = undefined;
    const next = this.queue.shift();
    if (next) this.show(next);
    this.behind = this.queue.length;
  }

  close() {
    clearTimeout(this.timer);
    clearTimeout(this.watchdog);
    this.connection.close();
  }

  private pump() {
    if (this.timer !== undefined || this.paused) return;
    const next = this.queue.shift();
    this.behind = this.queue.length;
    if (!next) return;
    this.show(next);
    const hold = next.type === 'event' ? holdFor(next.entry, this.viewer, SPEEDS[this.speed]) : 0;
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.pump();
    }, hold);
  }

  private show(msg: GameEvent | Over) {
    if (msg.type === 'over') {
      this.over = msg;
      this.summary = msg.game;
      this.view = msg.view;
      this.legal = [];
      this.connection.close();
      return;
    }
    this.view = msg.view;
    this.legal = msg.legal;
    this.log = [...this.log, msg.entry];
    this.last = msg.entry;
    if (msg.explanation) this.explanation = { entry: msg.entry, explanation: msg.explanation, thinkMs: msg.think_ms };
    if (this.sent && msg.entry.ply >= this.sent.ply) this.sent = null;
  }
}
