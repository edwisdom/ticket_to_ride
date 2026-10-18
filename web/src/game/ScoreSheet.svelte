<!--
  The end: each seat's score, broken down as the rules settle it, and every ticket revealed.
-->
<script lang="ts">
  import type { Score, SeatInfo, View } from '../api/protocol';
  import type { BoardModel } from '../board/layout';
  import { seatColor } from '../board/palette';

  interface Props {
    model: BoardModel;
    scores: Score[];
    seats: SeatInfo[];
    view: View;
    mine: number | null;
  }
  let { model, scores, seats, view, mine }: Props = $props();

  const ordered = $derived([...scores].sort((a, b) => a.rank - b.rank || a.seat - b.seat));
  const name = (seat: number) => seats.find((s) => s.seat === seat)?.player.name ?? `Seat ${seat + 1}`;
  const headline = $derived.by(() => {
    if (mine === null) return `${name(ordered[0]!.seat)} wins`;
    const me = scores.find((s) => s.seat === mine);
    if (!me) return 'Game over';
    const tied = scores.filter((s) => s.rank === 1).length > 1;
    return me.rank === 1 ? (tied ? 'A draw' : 'You win') : 'You lose';
  });
</script>

<div class="sheet">
  <h2>{headline}</h2>
  <table class="ledger">
    <thead>
      <tr>
        <th></th>
        <th class="num">Routes</th>
        <th class="num">Tickets</th>
        <th class="num">Ticket points</th>
        <th class="num">Longest</th>
        <th class="num">Total</th>
      </tr>
    </thead>
    <tbody>
      {#each ordered as s (s.seat)}
        <tr class:me={s.seat === mine}>
          <td><span class="seat-dot" style="background: {seatColor(s.seat).fill}"></span> {name(s.seat)}</td>
          <td class="num">{s.routes}</td>
          <td class="num">{s.tickets_made} made · {s.tickets_missed} missed</td>
          <td class="num">{s.ticket_points > 0 ? '+' : ''}{s.ticket_points}</td>
          <td class="num">{s.longest_trail}{s.longest_bonus ? ` · +${s.longest_bonus}` : ''}</td>
          <td class="num total">{s.total}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  <div class="tickets">
    {#each ordered as s (s.seat)}
      {@const seat = view.seats[s.seat]}
      <div>
        <h4>{name(s.seat)}'s tickets</h4>
        <ul>
          {#each seat?.tickets ?? [] as t (t.ticket)}
            {@const ticket = model.tickets[t.ticket]}
            {#if ticket}
              <li class:done={t.connected}>
                <span>{ticket.a} – {ticket.b}</span>
                <span class="num">{t.connected ? '+' : '−'}{ticket.points}</span>
              </li>
            {/if}
          {/each}
        </ul>
      </div>
    {/each}
  </div>
</div>

<style>
  .sheet {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  h2 {
    font-size: 24px;
    color: var(--oxblood);
  }
  .total {
    font-weight: 800;
    font-size: 17px;
  }
  .tickets {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
  }
  h4 {
    font-size: 13px;
    color: var(--ink-soft);
  }
  ul {
    list-style: none;
    margin: 4px 0 0;
    padding: 0;
    font-size: 13.5px;
  }
  li {
    display: flex;
    justify-content: space-between;
    color: var(--bad);
    padding: 1px 0;
  }
  li.done {
    color: var(--good);
  }
</style>
