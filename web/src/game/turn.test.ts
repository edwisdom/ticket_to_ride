import { describe, expect, it } from 'vitest';
import mini from '../../../ticket_to_ride/data/layout/mini.json';
import { fromServer } from '../api/boards';
import miniBoard from '../api/fixtures/board-mini.json';
import type { Action, Board, View } from '../api/protocol';
import { claimNote, pieces, yourTurn } from './turn';

const model = fromServer({ ...(miniBoard as unknown as Board), layout: mini });
const draw: Action = { kind: 'draw', id: 1, slot: null, card: null, label: 'Draw from the deck' };
const claim: Action = { kind: 'claim', id: 2, segment: 0, color: 0, colored: 4, locomotives: 0, label: 'Claim' };
const tickets: Action = { kind: 'draw_tickets', id: 3, dealt: null, tickets: null, label: 'Draw tickets' };
const pass: Action = { kind: 'pass', id: 4, label: 'Pass' };

function view(owners: (number | null)[] = model.segments.map(() => null)): View {
  return {
    viewer: 0,
    ply: 40,
    turn: 30,
    phase: 'main',
    current: 0,
    draws_left: 2,
    final_turns: null,
    owners,
    closed: [],
    faceup: [],
    deck: 20,
    discard: [],
    ticket_deck: 10,
    seats: [],
    offer: null,
    unseen: null,
    deck_cards: null,
  };
}

describe('pieces', () => {
  it('counts train pieces', () => {
    expect(pieces(1)).toBe('1 train piece');
    expect(pieces(0)).toBe('0 train pieces');
  });
});

describe('yourTurn', () => {
  const prompt = (legal: Action[], have = 20, v = view()) => yourTurn(legal, model, v, have).prompt;

  it('names the moves the server allows', () => {
    expect(prompt([draw, claim, tickets])).toBe('Your turn: draw two cards, claim a route, or draw tickets.');
    expect(prompt([draw, tickets])).toBe('Your turn: draw two cards or draw tickets.');
    expect(prompt([claim, tickets])).toBe('Your turn: claim a route or draw tickets.');
    expect(prompt([pass])).toBe('No move is possible: pass.');
  });

  it('says why no route can be claimed when the train pieces have run out', () => {
    expect(yourTurn([draw, tickets], model, view(), 0)).toEqual({
      prompt: 'Your turn: draw two cards or draw tickets.',
      why: 'No route can be claimed: you have no train pieces left.',
    });
  });

  it('says why when every open route is longer than the pieces left', () => {
    const shortest = Math.min(...model.segments.map((s) => s.length));
    const owners = model.segments.map((s) => (s.length === shortest ? 1 : null));
    expect(yourTurn([draw, tickets], model, view(owners), shortest).why).toBe(
      `No route can be claimed: you have ${pieces(shortest)} left, and the shortest open route takes ${shortest + 1}.`,
    );
  });

  it('gives no reason when the pieces would fit, or a route can be claimed', () => {
    expect(yourTurn([draw, tickets], model, view(), 20).why).toBeNull();
    expect(yourTurn([draw, claim, tickets], model, view(), 1).why).toBeNull();
  });
});

describe('claimNote', () => {
  it('says what a claim takes from the supply', () => {
    expect(claimNote(6, 20, 2, false)).toEqual({ ends: false, text: 'Takes 6 of your 20 train pieces, leaving 14.' });
    expect(claimNote(3, 6, 2, false).ends).toBe(false);
  });

  it('warns that a claim leaving the trigger or fewer starts the final round', () => {
    expect(claimNote(6, 6, 2, false)).toEqual({
      ends: true,
      text: 'Uses your last 6 train pieces: everyone, you included, gets one more turn, and you will have none left to claim with.',
    });
    expect(claimNote(1, 1, 2, false).text).toMatch(/^Uses your last train piece: /);
    expect(claimNote(4, 6, 2, false)).toEqual({
      ends: true,
      text: 'Leaves you 2 train pieces: everyone, you included, gets one more turn.',
    });
    expect(claimNote(5, 6, 2, false).text).toMatch(/^Leaves you 1 train piece: /);
  });

  it('does not warn once the final round is under way', () => {
    expect(claimNote(6, 6, 2, true)).toEqual({ ends: false, text: 'Uses your last 6 train pieces.' });
    expect(claimNote(4, 6, 2, true)).toEqual({ ends: false, text: 'Takes 4 of your 6 train pieces, leaving 2.' });
  });
});
