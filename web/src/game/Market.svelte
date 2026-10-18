<!--
  The face-up cards and the two draw piles. Each is a button only while the viewer holds a
  legal action for it, so the rules -- no face-up locomotive on a second draw, no tickets
  mid-turn -- stay the server's.
-->
<script lang="ts">
  import type { Action, DrawAction, DrawTicketsAction, LogEntry, View } from '../api/protocol';
  import type { BoardModel } from '../board/layout';
  import CardBack from '../cards/CardBack.svelte';
  import TrainCard from '../cards/TrainCard.svelte';

  interface Props {
    model: BoardModel;
    view: View;
    legal: Action[];
    last: LogEntry | null;
    onact: (action: Action) => void;
  }
  let { model, view, legal, last, onact }: Props = $props();

  const draws = $derived(legal.filter((a): a is DrawAction => a.kind === 'draw'));
  const fromSlot = (slot: number) => draws.find((a) => a.slot === slot);
  const fromDeck = $derived(draws.find((a) => a.slot === null));
  const tickets = $derived(legal.find((a): a is DrawTicketsAction => a.kind === 'draw_tickets'));
  // The slot the last move took from refills: let the new card arrive visibly.
  const refilled = $derived(last?.action.kind === 'draw' ? last.action.slot : null);
  const active = $derived(legal.length > 0);
</script>

<section class="panel market">
  <header><h3>Cards</h3><span class="muted-light">{view.deck} in the deck · {view.ticket_deck} tickets</span></header>
  <div class="body">
    <div class="faceup">
      {#each view.faceup as card, slot (slot)}
        {@const action = fromSlot(slot)}
        {#key `${view.ply}-${slot}-${card}`}
          <button
            class="slot"
            class:fresh={refilled === slot}
            disabled={!action}
            title={action?.label}
            onclick={() => action && onact(action)}
          >
            {#if card !== null}
              <TrainCard color={model.cardKinds[card]!} width={56} muted={active && !action} />
            {:else}
              <span class="empty"></span>
            {/if}
          </button>
        {/key}
      {/each}
    </div>
    <div class="piles">
      <button class="pile" disabled={!fromDeck} title={fromDeck?.label} onclick={() => fromDeck && onact(fromDeck)}>
        <CardBack kind="train" width={44} />
        <span>Deck<br /><b>{view.deck}</b></span>
      </button>
      <button class="pile" disabled={!tickets} title={tickets?.label} onclick={() => tickets && onact(tickets)}>
        <CardBack kind="ticket" width={70} />
        <span>Tickets<br /><b>{view.ticket_deck}</b></span>
      </button>
    </div>
  </div>
</section>

<style>
  .muted-light {
    font-size: 12px;
    color: var(--paper-deep);
  }
  .faceup {
    display: flex;
    gap: 6px;
    justify-content: space-between;
  }
  .slot,
  .pile {
    background: none;
    border: 0;
    padding: 2px;
    border-radius: 7px;
    color: inherit;
  }
  .slot:not(:disabled):hover,
  .pile:not(:disabled):hover {
    background: rgb(201 163 90 / 0.35);
    filter: none;
    transform: translateY(-2px);
  }
  .slot:disabled,
  .pile:disabled {
    opacity: 1;
  }
  .slot.fresh {
    animation: deal 500ms ease-out;
  }
  @keyframes deal {
    from {
      transform: translateY(-14px) rotate(-6deg);
      opacity: 0;
    }
  }
  .empty {
    display: block;
    width: 56px;
    aspect-ratio: 100 / 140;
    border: 2px dashed var(--paper-line);
    border-radius: 6px;
  }
  .piles {
    display: flex;
    gap: 14px;
    margin-top: 8px;
    align-items: center;
  }
  .pile {
    display: flex;
    align-items: center;
    gap: 8px;
    font: 600 13px var(--caps);
    text-align: left;
    color: var(--ink-soft);
  }
  .pile b {
    font: 700 17px var(--sans);
    color: var(--ink);
  }
</style>
