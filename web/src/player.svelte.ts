// Who is playing. There are no accounts yet (PLAN.md §14.2): a person is a player id the
// server issued, kept in this browser.
import type { Player } from './api/protocol';

const KEY = 'ttr.player';

function load(): Player | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Player) : null;
  } catch {
    return null;
  }
}

export const me = $state<{ player: Player | null }>({ player: load() });

export function remember(player: Player) {
  me.player = player;
  try {
    localStorage.setItem(KEY, JSON.stringify(player));
  } catch {
    // No storage: this page only.
  }
}

export function forget() {
  me.player = null;
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Nothing kept.
  }
}
