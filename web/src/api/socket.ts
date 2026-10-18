// The game socket: WS /ws/games/{id}. It reconnects on its own -- the server answers every
// connection with a fresh snapshot, and `act` is idempotent on its ply, so nothing is lost
// or applied twice -- until it is closed for good.
import type { ClientMessage, ServerMessage } from './protocol';

export type SocketStatus = 'connecting' | 'open' | 'reconnecting' | 'closed';

export class GameSocket {
  private ws: WebSocket | null = null;
  private retries = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private stopped = false;

  constructor(
    private readonly id: string,
    private readonly onMessage: (msg: ServerMessage) => void,
    private readonly onStatus: (status: SocketStatus) => void,
  ) {
    this.open();
  }

  private open() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${scheme}://${location.host}/ws/games/${encodeURIComponent(this.id)}`);
    this.ws = ws;
    this.onStatus(this.retries ? 'reconnecting' : 'connecting');
    ws.onopen = () => {
      this.retries = 0;
      this.onStatus('open');
    };
    ws.onmessage = (e) => this.onMessage(JSON.parse(e.data as string) as ServerMessage);
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.stopped) {
        this.onStatus('closed');
        return;
      }
      this.onStatus('reconnecting');
      this.timer = setTimeout(() => this.open(), Math.min(10_000, 500 * 2 ** this.retries++));
    };
  }

  /** False when there is no open connection to send on. */
  send(msg: ClientMessage): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  /** Drop the connection and open a new one, for a fresh snapshot. */
  reconnect() {
    this.ws?.close();
  }

  /** Stop for good: the game is over or gone, or the page is leaving. */
  close() {
    this.stopped = true;
    clearTimeout(this.timer);
    if (this.ws) this.ws.close();
    else this.onStatus('closed');
  }
}
