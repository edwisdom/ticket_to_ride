<!--
  A ticket keep: the offer, which to keep, and the legal keep that matches the choice. The
  server lists every subset it allows, so the minimum is its rule, not ours.
-->
<script lang="ts">
  import type { KeepAction } from '../api/protocol';
  import type { BoardModel } from '../board/layout';
  import TicketCard from '../cards/TicketCard.svelte';

  interface Props {
    model: BoardModel;
    offer: number[];
    keeps: KeepAction[];
    opening: boolean;
    onkeep: (action: KeepAction) => void;
    onhoverticket: (cities: [string, string] | null) => void;
  }
  let { model, offer, keeps, opening, onkeep, onhoverticket }: Props = $props();

  let chosen = $state<number[]>([]);
  const fewest = $derived(Math.min(...keeps.map((k) => k.count)));
  const key = (idxs: number[]) => [...idxs].sort((a, b) => a - b).join(',');
  const match = $derived(keeps.find((k) => key(k.offer_indices) === key(chosen)));

  const toggle = (i: number) => (chosen = chosen.includes(i) ? chosen.filter((c) => c !== i) : [...chosen, i]);
</script>

<div class="offer panel" role="dialog" aria-label="Choose tickets to keep">
  <header>
    <h3>{opening ? 'Your opening tickets' : 'New tickets'}</h3>
    <span>keep at least {fewest}</span>
  </header>
  <div class="body">
    <div class="tickets">
      {#each offer as id, i (id)}
        {@const t = model.tickets[id]}
        {#if t}
          <button class="pick" class:on={chosen.includes(i)} onclick={() => toggle(i)} aria-pressed={chosen.includes(i)}>
            <TicketCard layout={model.layout} a={t.a} b={t.b} points={t.points} width={170} onhover={onhoverticket} />
            <span class="tick">{chosen.includes(i) ? 'Keep' : 'Return'}</span>
          </button>
        {/if}
      {/each}
    </div>
    <div class="actions">
      <button disabled={!match} onclick={() => match && onkeep(match)}>
        {match ? `Keep ${match.count}` : `Choose at least ${fewest}`}
      </button>
      <button class="quiet" onclick={() => (chosen = offer.map((_, i) => i))}>All</button>
    </div>
  </div>
</div>

<style>
  .offer {
    position: absolute;
    z-index: 5;
    left: 50%;
    bottom: 4%;
    transform: translateX(-50%);
    max-width: 96%;
  }
  .offer header span {
    font-size: 13px;
    color: var(--paper-deep);
  }
  .tickets {
    display: flex;
    gap: 10px;
  }
  .pick {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 6px;
    background: transparent;
    border: 2px solid transparent;
    color: var(--ink-soft);
  }
  .pick:hover {
    filter: none;
    background: rgb(201 163 90 / 0.2);
  }
  .pick.on {
    border-color: var(--good);
    background: rgb(47 122 58 / 0.12);
    color: var(--good);
  }
  .tick {
    font: 700 13px var(--caps);
  }
  .actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    justify-content: center;
  }
</style>
