<!--
  A finished game, move by move, with everything revealed: every hand, every ticket, the
  deck's composition, and at each bot decision its reasoning, recomputed by the server.
-->
<script lang="ts">
  import { untrack } from 'svelte';
  import { loadBoard } from '../api/boards';
  import { api } from '../api/http';
  import type { Glossary, Review } from '../api/protocol';
  import Board from '../board/Board.svelte';
  import type { BoardModel } from '../board/layout';
  import CardChips from '../cards/CardChips.svelte';
  import TrainCard from '../cards/TrainCard.svelte';
  import GameLog from '../game/GameLog.svelte';
  import Reasoning from '../game/Reasoning.svelte';
  import ScoreSheet from '../game/ScoreSheet.svelte';
  import SeatPanel from '../game/SeatPanel.svelte';
  import { MAP_NAMES, versus } from '../format';
  import { href } from '../router';

  let { id, start = null }: { id: string; start?: number | null } = $props();

  let review = $state<Review | null>(null);
  let model = $state<BoardModel | null>(null);
  let glossary = $state<Glossary | null>(null);
  let failed = $state<string | null>(null);
  /** Moves shown: 0 is the deal, plies.length the end. */
  let at = $state(untrack(() => start ?? 0));
  let pointing = $state<[string, string] | null>(null);

  $effect(() => {
    api.review(id).then(
      async (r) => {
        model = await loadBoard(r.game.map);
        review = r;
      },
      (e: Error) => (failed = e.message),
    );
    api.glossary().then((g) => (glossary = g));
  });

  const total = $derived(review?.plies.length ?? 0);
  // The position is in the address, so a link can point at a move.
  $effect(() => {
    if (review) history.replaceState(null, '', href.review(id, at));
  });
  const ply = $derived(review && at > 0 ? review.plies[at - 1] : null);
  const view = $derived(ply?.view ?? review?.initial ?? null);
  const log = $derived(review ? review.plies.slice(0, at).map((p) => p.entry) : []);
  const closed = $derived(model && view ? model.segments.map((s) => view.closed.includes(s.index)) : []);
  const recent = $derived(ply?.entry.action.kind === 'claim' ? ply.entry.action.segment : null);
  const name = (seat: number) => review?.game.seats.find((s) => s.seat === seat)?.player.name ?? `Seat ${seat + 1}`;

  const go = (n: number) => (at = Math.max(0, Math.min(total, n)));
  /** The next or previous decision by a bot, to step through the reasoning. */
  function nextDecision(dir: 1 | -1) {
    if (!review) return;
    for (let i = at + dir; i >= 1 && i <= total; i += dir) {
      if (review.plies[i - 1]?.explanation) return go(i);
    }
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.target instanceof HTMLInputElement && e.target.type !== 'range') return;
    if (e.key === 'ArrowRight') go(at + 1);
    else if (e.key === 'ArrowLeft') go(at - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(total);
    else return;
    e.preventDefault();
  }
</script>

<svelte:window {onkeydown} />

{#if failed}
  <p class="notice error">{failed}</p>
{:else if !review || !model || !view}
  <p class="notice">Unrolling the game…</p>
{:else}
  <div class="review">
    <div class="status">
      <strong>{MAP_NAMES[review.game.map]} · {versus(review.game)}</strong>
      <button class="ctl" onclick={() => go(0)} title="The deal (Home)">⏮</button>
      <button class="ctl" onclick={() => nextDecision(-1)} title="The previous bot decision">◂ decision</button>
      <button class="ctl" onclick={() => go(at - 1)} title="Back a move (←)">◀</button>
      <input type="range" min="0" max={total} bind:value={at} aria-label="Move" />
      <button class="ctl" onclick={() => go(at + 1)} title="Forward a move (→)">▶</button>
      <button class="ctl" onclick={() => nextDecision(1)} title="The next bot decision">decision ▸</button>
      <button class="ctl" onclick={() => go(total)} title="The end (End)">⏭</button>
      <span class="where">{at === 0 ? 'The deal' : at === total ? 'The end' : `Move ${at} of ${total}`}</span>
    </div>

    <div class="board-area" style="--aspect: {model.layout.size[0] / model.layout.size[1]}">
      <div class="board-fit">
        <Board {model} owner={view.owners} {closed} scores={view.seats.map((s) => s.score)} highlight={pointing ?? []} {recent} />
      </div>
      {#if at === total}
        <div class="over panel">
          <header><h3>Final scores</h3><span class="seed">seed {review.seed}</span></header>
          <div class="body">
            <ScoreSheet {model} scores={review.scores} seats={review.game.seats} view={view} mine={review.game.human_seat} />
          </div>
        </div>
      {/if}
    </div>

    <aside class="side">
      {#if ply && !ply.explanation}<p class="move">{ply.entry.text}</p>{/if}
      {#if ply?.explanation}
        <Reasoning explanation={ply.explanation} entry={ply.entry} thinkMs={ply.think_ms} {glossary} playerName={name(ply.entry.seat)} />
      {/if}
      {#each view.seats as seat (seat.seat)}
        <SeatPanel
          {model}
          {seat}
          player={review.game.seats.find((s) => s.seat === seat.seat)?.player}
          current={view.current === seat.seat && at < total}
          mine={seat.seat === review.game.human_seat}
          showTickets
          onhoverticket={(c) => (pointing = c)}
        />
      {/each}
      <section class="panel">
        <header><h3>Cards</h3></header>
        <div class="body table">
          <div class="faceup">
            {#each view.faceup as card, slot (slot)}
              {#if card !== null}<TrainCard color={model.cardKinds[card]!} width={44} />{/if}
            {/each}
          </div>
          {#if view.deck_cards}
            <div class="line"><span class="what">Deck</span><CardChips kinds={model.cardKinds} counts={view.deck_cards} /></div>
          {/if}
          <div class="line"><span class="what">Discards</span><CardChips kinds={model.cardKinds} counts={view.discard} /></div>
        </div>
      </section>
      <section class="panel log">
        <header><h3>Log</h3><a class="back" href={href.lobby()}>lobby</a></header>
        <GameLog {log} current={ply?.entry.ply ?? null} />
      </section>
    </aside>
  </div>
{/if}

<style>
  .notice {
    color: var(--paper);
    padding: 40px;
  }
  .review {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 420px;
    grid-template-rows: auto minmax(0, 1fr);
    grid-template-areas:
      'status status'
      'board side';
    gap: 10px;
    padding: 10px 12px 12px;
    height: calc(100vh - 44px);
  }
  .status {
    grid-area: status;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--paper);
  }
  .status strong {
    font: 700 15px var(--caps);
    margin-right: 10px;
  }
  .status input[type='range'] {
    flex: 1;
    accent-color: var(--gilt);
  }
  .ctl {
    font-size: 12.5px;
    padding: 3px 9px;
  }
  .where {
    min-width: 110px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .board-area {
    grid-area: board;
    position: relative;
    container-type: size;
    display: grid;
    place-items: center;
    min-height: 0;
  }
  .board-fit {
    width: min(100cqw, calc(100cqh * var(--aspect)));
  }
  .side {
    grid-area: side;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
    overflow-y: auto;
  }
  .side > :global(.panel) {
    flex: none;
  }
  .move {
    margin: 0;
    color: var(--gilt-soft);
    font-weight: 600;
  }
  .table {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .faceup {
    display: flex;
    gap: 6px;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
  }
  .what {
    font: 600 12px var(--caps);
    color: var(--ink-soft);
    width: 58px;
    flex: none;
  }
  .log {
    flex: 1 1 160px !important;
    display: flex;
    flex-direction: column;
    min-height: 140px;
  }
  .back {
    font-size: 12px;
    color: var(--paper-deep);
  }
  .over {
    position: absolute;
    z-index: 6;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: min(720px, 94%);
    max-height: 94%;
    overflow-y: auto;
  }
  .seed {
    font-size: 12px;
    color: var(--paper-deep);
  }
</style>
