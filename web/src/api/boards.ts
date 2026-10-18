// Boards come from GET /api/boards/{map}: the engine's arrays in the engine's order, with the
// layout tools/gen_layout.py draws. Never from a map TOML, whose order mini does not share.
import { joinBoard, type BoardModel, type CardKind, type Layout, type RouteColor } from '../board/layout';
import { api } from './http';
import type { Board, MapName } from './protocol';

/** A board's model from the server's description of it. */
export function fromServer(board: Board): BoardModel {
  if (!board.layout) throw new Error(`the server has no layout for ${board.name}: run make layout`);
  const city = (i: number) => board.cities[i]!;
  const model = joinBoard(board.layout as unknown as Layout, {
    segments: board.segments.map((s) => ({
      a: city(s.a),
      b: city(s.b),
      length: s.length,
      color: (s.color === null ? 'gray' : board.colors[s.color]) as RouteColor,
    })),
    // The server's route_points is indexed by length, from zero.
    routePoints: board.route_points.slice(1),
    cardKinds: [...board.colors, 'locomotive'] as CardKind[],
    cityNames: board.cities,
    tickets: board.tickets.map((t) => ({ a: city(t.a), b: city(t.b), points: t.points })),
    trainsPerPlayer: board.trains_per_player,
    endTriggerTrains: board.end_trigger_trains,
  });
  for (const s of board.segments) {
    if (model.segments[s.id]?.sibling !== s.sibling) throw new Error(`${board.name}: segment ${s.id}'s sibling disagrees`);
  }
  return model;
}

const boards = new Map<MapName, Promise<BoardModel>>();

/** Each map's board, fetched once. */
export function loadBoard(map: MapName): Promise<BoardModel> {
  let board = boards.get(map);
  if (!board) {
    board = api.board(map).then(fromServer);
    board.catch(() => boards.delete(map));
    boards.set(map, board);
  }
  return board;
}
