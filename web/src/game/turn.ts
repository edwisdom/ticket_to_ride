// The viewer's turn in words. Train pieces -- the plastic trains a seat places on routes --
// are easy to mix up with train cards, which show a railcar too, and both the final round and
// whether a route can be claimed turn on pieces alone. The wording here keeps the two apart.
import type { Action, View } from '../api/protocol';
import type { BoardModel } from '../board/layout';
import { plural } from '../format';

/** "1 train piece", "6 train pieces". */
export const pieces = (n: number) => plural(n, 'train piece');

/** "a or b", "a, b, or c". */
const or = (items: string[]) =>
  items.length < 3 ? items.join(' or ') : `${items.slice(0, -1).join(', ')}, or ${items.at(-1)}`;

/** Why no route can be claimed, when train pieces are the reason: every open route is longer
 * than the supply. Null when some open route would fit -- then it is the cards. */
function tooFewPieces(model: BoardModel, view: View, have: number): string | null {
  const open = model.segments.filter((s) => view.owners[s.index] === null && !view.closed.includes(s.index));
  if (!open.length) return null;
  const shortest = Math.min(...open.map((s) => s.length));
  if (have >= shortest) return null;
  if (have === 0) return 'No route can be claimed: you have no train pieces left.';
  return `No route can be claimed: you have ${pieces(have)} left, and the shortest open route takes ${shortest}.`;
}

/** The viewer's own turn: the moves the server allows -- a claim only when there is one --
 * and, when train pieces rule every route out, why. */
export function yourTurn(
  legal: Action[],
  model: BoardModel,
  view: View,
  have: number,
): { prompt: string; why: string | null } {
  const moves: string[] = [];
  if (legal.some((a) => a.kind === 'draw')) moves.push('draw two cards');
  if (legal.some((a) => a.kind === 'claim')) moves.push('claim a route');
  if (legal.some((a) => a.kind === 'draw_tickets')) moves.push('draw tickets');
  const prompt = moves.length ? `Your turn: ${or(moves)}.` : 'No move is possible: pass.';
  return { prompt, why: moves.includes('claim a route') ? null : tooFewPieces(model, view, have) };
}

/** What claiming a route of `length` spaces does to a supply of `have` train pieces. It `ends`
 * when it leaves `endTrigger` or fewer before the final round has begun: the claim starts the
 * final round. */
export function claimNote(
  length: number,
  have: number,
  endTrigger: number,
  finalRound: boolean,
): { ends: boolean; text: string } {
  const left = have - length;
  const ends = !finalRound && left <= endTrigger;
  const cost =
    left === 0
      ? `Uses your last ${length === 1 ? 'train piece' : pieces(length)}`
      : ends
        ? `Leaves you ${pieces(left)}`
        : `Takes ${length} of your ${pieces(have)}, leaving ${left}`;
  if (!ends) return { ends, text: `${cost}.` };
  const none = left === 0 ? ', and you will have none left to claim with' : '';
  return { ends, text: `${cost}: everyone, you included, gets one more turn${none}.` };
}
