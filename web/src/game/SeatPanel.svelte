<!--
  One seat: what everyone knows of it -- train pieces left, banked points, hand size, the cards
  known to be in its hand (face-up takes not yet spent), its tickets' count, its longest trail
  -- and, when the viewer may see them, its hand and tickets.
-->
<script lang="ts">
  import type { Player, SeatView } from '../api/protocol';
  import type { BoardModel } from '../board/layout';
  import { seatColor } from '../board/palette';
  import CardChips from '../cards/CardChips.svelte';

  interface Props {
    model: BoardModel;
    seat: SeatView;
    player: Player | undefined;
    /** Its turn. */
    current?: boolean;
    /** The viewer's own seat. */
    mine?: boolean;
    /** Show the tickets: an omniscient view (spectating, review). */
    showTickets?: boolean;
    onhoverticket?: (cities: [string, string] | null) => void;
  }
  let { model, seat, player, current = false, mine = false, showTickets = false, onhoverticket }: Props = $props();

  const color = $derived(seatColor(seat.seat));
  const known = $derived(seat.known.reduce((a, b) => a + b, 0));
</script>

<section class="seat panel" class:current>
  <header style="--seat: {color.fill}">
    <h3>
      <span class="seat-dot" style="background: {color.fill}"></span>
      {player?.name ?? `Seat ${seat.seat + 1}`}{#if mine}<span class="you"> (you)</span>{/if}
    </h3>
    {#if current}<span class="turn">to move</span>{/if}
  </header>
  <div class="body">
    <dl class="stats">
      <div><dt>Points</dt><dd>{seat.score}</dd></div>
      <div
        class="supply"
        class:low={seat.trains <= model.endTriggerTrains}
        title="Train pieces left, of {model.trainsPerPlayer}: the plastic trains placed on routes, one per space. Not cards."
      >
        <dt>Train pieces</dt><dd>{seat.trains}</dd>
      </div>
      <div title="Train cards in hand."><dt>Cards</dt><dd>{seat.hand_size}</dd></div>
      <div><dt>Tickets</dt><dd>{seat.ticket_count}</dd></div>
      <div><dt>Longest</dt><dd>{seat.longest_trail}</dd></div>
    </dl>
    {#if seat.hand && !mine}
      <div class="line"><span class="what">Hand</span><CardChips kinds={model.cardKinds} counts={seat.hand} /></div>
    {:else if !mine}
      <div class="line">
        <span class="what" title="Cards taken face up and not yet spent: everyone saw them.">Known</span>
        {#if known}<CardChips kinds={model.cardKinds} counts={seat.known} />{:else}<span class="muted">none</span>{/if}
        {#if seat.hand_size > known}<span class="muted">+ {seat.hand_size - known} unseen</span>{/if}
      </div>
    {/if}
    {#if showTickets && seat.tickets}
      <ul class="tickets">
        {#each seat.tickets as t (t.ticket)}
          {@const ticket = model.tickets[t.ticket]}
          {#if ticket}
            <li
              class:done={t.connected}
              onmouseenter={() => onhoverticket?.([ticket.a, ticket.b])}
              onmouseleave={() => onhoverticket?.(null)}
            >
              <span>{ticket.a} – {ticket.b}</span>
              <span class="pts">{ticket.points}</span>
              <span class="state">{t.connected ? '✓' : t.remaining === null ? '✗' : `${t.remaining} to go`}</span>
            </li>
          {/if}
        {/each}
      </ul>
    {/if}
  </div>
</section>

<style>
  .seat > header {
    background: linear-gradient(color-mix(in srgb, var(--seat) 55%, #2a0f0b), color-mix(in srgb, var(--seat) 35%, #1a0806));
  }
  .seat.current {
    box-shadow:
      0 0 0 2px var(--gilt),
      0 6px 18px rgb(0 0 0 / 0.35);
  }
  h3 {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .you {
    font-weight: 400;
    opacity: 0.8;
  }
  .turn {
    font: 600 12px var(--sans);
    color: var(--gilt-soft);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    margin: 0 0 6px;
    text-align: center;
  }
  dt {
    font: 600 11px var(--caps);
    color: var(--ink-soft);
  }
  dd {
    margin: 0;
    font: 700 18px var(--sans);
    font-variant-numeric: lining-nums tabular-nums;
  }
  .supply {
    cursor: help;
  }
  .supply.low dd {
    color: var(--bad);
  }
  .line {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 14px;
  }
  .what {
    font: 600 12px var(--caps);
    color: var(--ink-soft);
    width: 44px;
  }
  .tickets {
    list-style: none;
    margin: 6px 0 0;
    padding: 0;
    font-size: 13px;
  }
  .tickets li {
    display: grid;
    grid-template-columns: 1fr 2em 5.5em;
    gap: 6px;
    padding: 2px 0;
    border-top: 1px dotted var(--paper-line);
    cursor: default;
  }
  .tickets li:hover {
    background: rgb(201 163 90 / 0.2);
  }
  .pts {
    text-align: right;
    font-weight: 700;
  }
  .state {
    text-align: right;
    color: var(--ink-soft);
  }
  .tickets li.done .state {
    color: var(--good);
    font-weight: 700;
  }
</style>
