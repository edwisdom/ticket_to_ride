<!--
  Claiming a route: the ways the server says the viewer can pay for it, as the cards each
  would spend, and the train pieces it takes -- with a warning when it starts the final round.
  One click claims; Escape or a click outside cancels.
-->
<script lang="ts">
  import type { ClaimAction } from '../api/protocol';
  import type { BoardModel, CardKind, Segment } from '../board/layout';
  import { SWATCH, SYMBOL } from '../board/palette';
  import { plural } from '../format';
  import { claimNote } from './turn';

  interface Props {
    model: BoardModel;
    segment: Segment;
    options: ClaimAction[];
    /** The claimer's train pieces. */
    trains: number;
    /** The final round is already under way. */
    finalRound: boolean;
    onpick: (action: ClaimAction) => void;
    onclose: () => void;
  }
  let { model, segment, options, trains, finalRound, onpick, onclose }: Props = $props();

  const points = $derived(model.routePoints[segment.length - 1] ?? 0);
  const note = $derived(claimNote(segment.length, trains, model.endTriggerTrains, finalRound));
  const cards = (a: ClaimAction): CardKind[] => [
    ...Array<CardKind>(a.colored).fill(model.cardKinds[a.color]!),
    ...Array<CardKind>(a.locomotives).fill('locomotive'),
  ];

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') onclose();
    const n = Number(e.key);
    if (n >= 1 && n <= options.length) onpick(options[n - 1]!);
  }
</script>

<svelte:window {onkeydown} />

<div class="scrim" role="presentation" onclick={onclose}></div>
<div class="picker panel" role="dialog" aria-label="Pay for {segment.a} to {segment.b}">
  <header>
    <h3>{segment.a} – {segment.b}</h3>
    <span>{segment.length} {SWATCH[segment.color].label.toLowerCase()} · {plural(points, 'point')}</span>
  </header>
  <div class="body">
    {#if note.ends}
      <p class="ends" role="alert"><strong>This starts the final round.</strong> {note.text}</p>
    {:else}
      <p class="cost">{note.text}</p>
    {/if}
    <p>{options.length === 1 ? 'Pay with:' : 'Choose how to pay:'}</p>
    <ul>
      {#each options as a, i (a.id)}
        <li>
          <button class="quiet option" onclick={() => onpick(a)}>
            <span class="key">{i + 1}</span>
            <span class="cards">
              {#each cards(a) as kind, k (k)}
                <svg viewBox="0 0 20 28" width="20" height="28" aria-hidden="true">
                  <rect x="0.6" y="0.6" width="18.8" height="26.8" rx="3" fill={kind === 'locomotive' ? '#d9c08a' : SWATCH[kind].fill} stroke={SWATCH[kind].dark} />
                  <path d={SYMBOL[kind]} transform="translate(10 14) scale(5.5)" fill={kind === 'locomotive' ? '#3a2b18' : SWATCH[kind].ink} fill-rule="evenodd" />
                </svg>
              {/each}
            </span>
            <span class="label">{a.label.split(' with ').at(-1)}</span>
          </button>
        </li>
      {/each}
    </ul>
    <button class="quiet cancel" onclick={onclose}>Cancel</button>
  </div>
</div>

<style>
  .scrim {
    position: absolute;
    inset: 0;
    z-index: 4;
  }
  .picker {
    position: absolute;
    z-index: 5;
    left: 50%;
    top: 12%;
    transform: translateX(-50%);
    min-width: 320px;
    max-width: 92%;
  }
  .picker header span {
    font-size: 13px;
    color: var(--paper-deep);
  }
  p {
    margin: 0 0 6px;
    color: var(--ink-soft);
  }
  .cost {
    color: var(--ink);
  }
  .ends {
    max-width: 380px;
    margin-bottom: 10px;
    padding: 6px 10px;
    color: var(--ink);
    background: #f6d9a8;
    border: 1px solid #b7791f;
    border-left-width: 4px;
    border-radius: 4px;
  }
  .ends strong {
    display: block;
    color: var(--bad);
  }
  ul {
    list-style: none;
    margin: 0 0 8px;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .option {
    width: 100%;
    display: grid;
    grid-template-columns: auto auto 1fr;
    align-items: center;
    gap: 10px;
    color: var(--ink);
    padding: 6px 10px;
  }
  .key {
    font: 700 12px var(--sans);
    color: var(--ink-soft);
    border: 1px solid var(--paper-line);
    border-radius: 4px;
    padding: 0 5px;
  }
  .cards {
    display: flex;
    gap: 2px;
  }
  .label {
    text-align: left;
  }
  .cancel {
    float: right;
  }
</style>
