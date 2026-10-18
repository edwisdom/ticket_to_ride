<!--
  A game, played or watched. Everything on screen is the session's view of it; every move is
  one of the server's decoded legal actions, so the client knows no rules.
-->
<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { loadBoard } from '../api/boards';
  import { api } from '../api/http';
  import type { Action, ClaimAction, Glossary, KeepAction, PassAction, SeatView } from '../api/protocol';
  import Board from '../board/Board.svelte';
  import type { BoardModel } from '../board/layout';
  import { seatColor } from '../board/palette';
  import CardChips from '../cards/CardChips.svelte';
  import TicketCard from '../cards/TicketCard.svelte';
  import TrainCard from '../cards/TrainCard.svelte';
  import GameLog from '../game/GameLog.svelte';
  import Market from '../game/Market.svelte';
  import PaymentPicker from '../game/PaymentPicker.svelte';
  import Reasoning from '../game/Reasoning.svelte';
  import ScoreSheet from '../game/ScoreSheet.svelte';
  import SeatPanel from '../game/SeatPanel.svelte';
  import { GameSession, SPEEDS, type Speed } from '../game/session.svelte';
  import TicketOffer from '../game/TicketOffer.svelte';
  import { yourTurn } from '../game/turn';
  import { href } from '../router';

  let { id }: { id: string } = $props();

  // The app re-creates this screen for another game, so the id is read once.
  const session = new GameSession(untrack(() => id));
  // Development builds expose the session to the console, to drive a game from a script.
  if (import.meta.env.DEV) (globalThis as unknown as { session: GameSession }).session = session;
  onDestroy(() => session.close());

  let model = $state<BoardModel | null>(null);
  let failed = $state<string | null>(null);
  let glossary = $state<Glossary | null>(null);
  $effect(() => {
    const map = session.summary?.map;
    if (map && !model) loadBoard(map).then((m) => (model = m), (e: Error) => (failed = e.message));
  });
  $effect(() => {
    if (session.spectating && !glossary) api.glossary().then((g) => (glossary = g));
  });

  let picking = $state<number | null>(null);
  let pointing = $state<[string, string] | null>(null);
  let showOver = $state(true);

  const view = $derived(session.view);
  const summary = $derived(session.summary);
  const viewer = $derived(session.viewer);
  const legal = $derived(session.sent ? [] : session.legal);
  const claims = $derived(legal.filter((a): a is ClaimAction => a.kind === 'claim'));
  const keeps = $derived(legal.filter((a): a is KeepAction => a.kind === 'keep'));
  const pass = $derived(legal.find((a): a is PassAction => a.kind === 'pass'));
  const claimable = $derived(new Set(claims.map((a) => a.segment)));
  const closed = $derived(model && view ? model.segments.map((s) => view.closed.includes(s.index)) : []);
  const recent = $derived(session.last?.action.kind === 'claim' ? session.last.action.segment : null);
  const name = (seat: number) => summary?.seats.find((s) => s.seat === seat)?.player.name ?? `Seat ${seat + 1}`;
  const others = $derived(view ? view.seats.filter((s) => s.seat !== viewer) : []);
  const mine = $derived(view && viewer !== null ? view.seats[viewer] : undefined);

  // The viewer's own turn: the moves it allows and, when train pieces rule out every route, why.
  const own = $derived(
    view?.phase === 'main' && legal.length && model && mine ? yourTurn(legal, model, view, mine.trains) : null,
  );
  const prompt = $derived.by(() => {
    if (!view || session.over) return null;
    if (legal.length) {
      if (view.phase === 'initial_tickets') return 'Choose your opening tickets.';
      if (view.phase === 'ticket_keep') return 'Choose which tickets to keep.';
      if (view.phase === 'draw_second') return 'Draw one more card: from the deck, or a face-up card that is not a locomotive.';
      return own?.prompt ?? 'Your turn.';
    }
    if (session.sent) return 'Sending…';
    return `${name(view.current)} is ${view.phase === 'initial_tickets' || view.phase === 'ticket_keep' ? 'choosing tickets' : 'to move'}…`;
  });

  function act(action: Action) {
    picking = null;
    session.act(action);
  }

  function select(segment: number) {
    if (claimable.has(segment)) picking = segment;
  }

  const SPEED_LABELS: Record<Speed, string> = { slow: 'Slow', normal: 'Normal', fast: 'Fast', instant: 'Instant' };
</script>

{#if failed}
  <p class="notice error">{failed}</p>
{:else if !model || !view || !summary}
  <p class="notice">{session.error ?? 'Setting out…'}</p>
{:else}
  <div class="game" class:watching={session.spectating}>
    <div class="status">
      <div class="say">
        <span class="prompt" class:yours={legal.length > 0}>{prompt ?? 'Game over.'}</span>
        {#if own?.why}<span class="why">{own.why}</span>{/if}
      </div>
      {#if view.final_turns !== null && !session.over}
        <span class="final">Final round: {view.final_turns} turn{view.final_turns === 1 ? '' : 's'} left</span>
      {/if}
      {#if session.status !== 'open' && !session.over}
        <span class="conn">{session.status === 'reconnecting' ? 'Reconnecting…' : 'Connecting…'}</span>
      {/if}
      {#if session.error}<span class="error-chip">{session.error}</span>{/if}
      <span class="spacer"></span>
      {#if pass}<button onclick={() => act(pass)}>Pass</button>{/if}
      {#if session.spectating}
        <button class="ctl" onclick={() => session.togglePause()}>{session.paused ? 'Play' : 'Pause'}</button>
        <button class="ctl" onclick={() => session.step()} disabled={session.behind === 0}>Step</button>
      {/if}
      {#if session.behind > 0}<span class="behind">{session.behind} to show</span>{/if}
      <div class="choices speed" title="How fast the bots' moves are shown">
        {#each Object.keys(SPEEDS) as s (s)}
          <button class:on={session.speed === s} onclick={() => session.setSpeed(s as Speed)}>{SPEED_LABELS[s as Speed]}</button>
        {/each}
      </div>
    </div>

    <div class="board-area" style="--aspect: {model.layout.size[0] / model.layout.size[1]}">
      <div class="board-fit">
        <Board
          {model}
          owner={view.owners}
          {closed}
          scores={view.seats.map((s) => s.score)}
          highlight={pointing ?? []}
          selectable={claimable.size ? claimable : null}
          {recent}
          onselect={select}
        />
      </div>
      {#if picking !== null && model.segments[picking]}
        <PaymentPicker
          {model}
          segment={model.segments[picking]!}
          options={claims.filter((c) => c.segment === picking)}
          trains={mine?.trains ?? 0}
          finalRound={view.final_turns !== null}
          onpick={act}
          onclose={() => (picking = null)}
        />
      {/if}
      {#if keeps.length && view.offer}
        {#key view.ply}
          <TicketOffer
            {model}
            offer={view.offer}
            {keeps}
            opening={view.phase === 'initial_tickets'}
            onkeep={act}
            onhoverticket={(c) => (pointing = c)}
          />
        {/key}
      {/if}
      {#if session.over && showOver}
        <div class="over panel">
          <header><h3>Final scores</h3><button class="quiet close" onclick={() => (showOver = false)}>Close</button></header>
          <div class="body">
            <ScoreSheet {model} scores={session.over.scores} seats={summary.seats} view={session.over.view} mine={viewer} />
            <div class="after">
              <a class="button" href={href.review(id)}>Review the game</a>
              <a class="button quiet" href={href.lobby()}>Back to the lobby</a>
              <span class="muted">Seed {session.over.seed}</span>
            </div>
          </div>
        </div>
      {/if}
    </div>

    {#snippet seats(list: SeatView[], board: BoardModel)}
      {#each list as seat (seat.seat)}
        <SeatPanel
          model={board}
          {seat}
          player={summary.seats.find((s) => s.seat === seat.seat)?.player}
          current={view.current === seat.seat && !session.over}
          showTickets={view.viewer === null}
          onhoverticket={(c) => (pointing = c)}
        />
      {/each}
    {/snippet}

    {#if session.spectating}
      <div class="seats">{@render seats(view.seats, model)}</div>
    {/if}

    <aside class="side">
      {#if !session.spectating}{@render seats(others, model)}{/if}
      <Market {model} {view} {legal} last={session.last} onact={act} />
      {#if session.spectating && session.explanation}
        <Reasoning
          explanation={session.explanation.explanation}
          entry={session.explanation.entry}
          thinkMs={session.explanation.thinkMs}
          {glossary}
          playerName={name(session.explanation.entry.seat)}
        />
      {/if}
      {#if !session.spectating && view.unseen}
        <section class="panel assist">
          <header><h3>On the table</h3></header>
          <div class="body">
            <div class="line" title="Every card discarded so far: spent on routes.">
              <span class="what">Discards</span><CardChips kinds={model.cardKinds} counts={view.discard} />
            </div>
            <div class="line" title="Cards you cannot account for: the deck, and what the others drew blind.">
              <span class="what">Unseen</span><CardChips kinds={model.cardKinds} counts={view.unseen} />
            </div>
          </div>
        </section>
      {/if}
      <section class="panel log">
        <header><h3>Log</h3></header>
        <GameLog log={session.log} />
      </section>
    </aside>

    {#if mine}
      <div class="mine panel">
        <div class="me-head">
          <span class="seat-dot" style="background: {seatColor(mine.seat).fill}"></span>
          <strong>{name(mine.seat)}</strong>
          <span><b>{mine.score}</b> points</span>
          <span
            class="supply"
            class:low={mine.trains <= model.endTriggerTrains}
            title="The plastic trains you place on routes, one per space. Ending a turn with {model.endTriggerTrains} or fewer starts the final round."
          ><b>{mine.trains}</b> of {model.trainsPerPlayer} train pieces left</span>
          <span><b>{mine.hand_size}</b> {mine.hand_size === 1 ? 'card' : 'cards'} in hand</span>
          <span class="muted">longest trail {mine.longest_trail}</span>
        </div>
        <div class="hand">
          {#each model.cardKinds as kind, type (kind)}
            {#if (mine.hand?.[type] ?? 0) > 0}
              <TrainCard color={kind} count={mine.hand?.[type] ?? 0} width={58} />
            {/if}
          {/each}
          {#if !mine.hand?.some((n) => n > 0)}<span class="muted">No cards yet.</span>{/if}
        </div>
        <div class="tickets">
          {#each mine.tickets ?? [] as t (t.ticket)}
            {@const ticket = model.tickets[t.ticket]}
            {#if ticket}
              <TicketCard
                layout={model.layout}
                a={ticket.a}
                b={ticket.b}
                points={ticket.points}
                status={t.connected ? 'done' : session.over ? 'failed' : 'open'}
                remaining={t.remaining}
                width={138}
                onhover={(c) => (pointing = c)}
              />
            {/if}
          {/each}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .notice {
    color: var(--paper);
    padding: 40px;
  }
  .game {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    grid-template-rows: auto minmax(0, 1fr) auto;
    grid-template-areas:
      'status side'
      'board side'
      'mine side';
    gap: 10px;
    padding: 10px 12px 12px;
    height: calc(100vh - 44px);
  }
  .game.watching {
    grid-template-columns: minmax(0, 1fr) 420px;
    grid-template-areas:
      'status side'
      'board side'
      'seats side';
  }
  .seats {
    grid-area: seats;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 10px;
    align-items: start;
  }
  .watching .side :global(.reasoning) {
    flex: 1 1 260px;
  }
  .status {
    grid-area: status;
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--paper);
    min-height: 32px;
  }
  .say {
    display: flex;
    flex-direction: column;
  }
  .prompt {
    font: 600 16px var(--sans);
  }
  .prompt.yours {
    color: var(--gilt-soft);
  }
  .why {
    font-size: 13.5px;
    color: var(--paper-deep);
  }
  /* The chip and the controls keep their size; a long prompt wraps instead. */
  .final,
  .speed {
    flex: none;
  }
  .final {
    font: 700 13px var(--caps);
    color: #ffd27a;
    border: 1px solid #ffd27a;
    border-radius: 4px;
    padding: 1px 8px;
  }
  .conn,
  .behind {
    font-size: 13px;
    color: var(--paper-deep);
  }
  .error-chip {
    font-size: 13px;
    color: #ffb4a8;
  }
  .spacer {
    flex: 1;
  }
  .speed button,
  .ctl {
    font-size: 12.5px;
    padding: 3px 9px;
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
  .log {
    flex: 1 1 140px !important;
    display: flex;
    flex-direction: column;
    min-height: 120px;
  }
  .assist .line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    padding: 2px 0;
    cursor: help;
  }
  .what {
    font: 600 12px var(--caps);
    color: var(--ink-soft);
    width: 58px;
    flex: none;
  }
  .mine {
    grid-area: mine;
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
    gap: 4px 18px;
    padding: 8px 12px;
  }
  .me-head {
    grid-column: 1 / -1;
    display: flex;
    gap: 16px;
    align-items: baseline;
    font-size: 14px;
  }
  .me-head strong {
    font: 700 16px var(--caps);
  }
  .me-head b {
    font-size: 17px;
    /* Old-style figures draw 0 as a small o: "o of 45 train pieces". */
    font-variant-numeric: lining-nums;
  }
  .supply {
    cursor: help;
  }
  .supply.low b {
    color: var(--bad);
  }
  .hand {
    display: flex;
    gap: 6px;
    align-items: flex-end;
  }
  .tickets {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    align-items: flex-end;
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
  .over .close {
    color: var(--cream);
    border-color: var(--gilt);
    padding: 2px 10px;
  }
  .after {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-top: 12px;
  }
</style>
