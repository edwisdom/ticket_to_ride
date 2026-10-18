<!--
  Counts per card type as small chips in the card's colour, each with its symbol: a hand, a
  discard pile, what a seat cannot account for.
-->
<script lang="ts">
  import type { CardKind } from '../board/layout';
  import { SWATCH, SYMBOL } from '../board/palette';

  interface Props {
    kinds: CardKind[];
    counts: number[];
    /** Leave out the types with none. */
    compact?: boolean;
  }
  let { kinds, counts, compact = true }: Props = $props();

  const uid = `chip-${Math.random().toString(36).slice(2, 8)}`;
  const shown = $derived(kinds.map((kind, i) => ({ kind, n: counts[i] ?? 0 })).filter((c) => !compact || c.n > 0));
</script>

<span class="chips">
  {#each shown as { kind, n } (kind)}
    {@const sw = SWATCH[kind]}
    <span class="chip" class:none={n === 0} title="{n} {sw.label.toLowerCase()}">
      <svg viewBox="-10 -10 20 20" width="16" height="16" aria-hidden="true">
        {#if kind === 'locomotive'}
          <defs>
            <linearGradient id="{uid}-rainbow" x1="0" y1="0" x2="1" y2="1">
              {#each ['#c83a30', '#e78a2c', '#f1cd3a', '#4e9a3c', '#2f6db3', '#b05aa6'] as c, i (c)}
                <stop offset={i / 5} stop-color={c} />
              {/each}
            </linearGradient>
          </defs>
          <circle r="9.5" fill="url(#{uid}-rainbow)" stroke="#3b2a1a" stroke-width="1" />
        {:else}
          <circle r="9.5" fill={sw.fill} stroke={sw.dark} stroke-width="1" />
        {/if}
        <path d={SYMBOL[kind]} transform="scale(5.2)" fill={kind === 'locomotive' ? '#2e1d10' : sw.ink} fill-rule="evenodd" />
      </svg>
      <span class="n">{n}</span>
    </span>
  {/each}
  {#if !shown.length}<span class="muted">none</span>{/if}
</span>

<style>
  .chips {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 4px 8px;
    align-items: center;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }
  .chip.none {
    opacity: 0.35;
  }
  .n {
    min-width: 0.7em;
  }
</style>
