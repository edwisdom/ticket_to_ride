<!--
  A bot's decision: what it knew (context), its best candidates with the move played first,
  and each candidate's total broken into its additive terms -- drawn as bars on one scale,
  gains right of zero and costs left -- with the facts behind them. Names explain themselves
  on hover, from the server's glossary.
-->
<script lang="ts">
  import type { Explanation, Glossary, LogEntry } from '../api/protocol';
  import { seatColor } from '../board/palette';
  import { axis, gloss, num, ranked, ties, words } from './explain';
  import { plural } from '../format';

  interface Props {
    explanation: Explanation;
    entry: LogEntry;
    thinkMs?: number;
    glossary: Glossary | null;
    playerName: string;
  }
  let { explanation, entry, thinkMs, glossary, playerName }: Props = $props();

  const shown = $derived(ranked(explanation));
  const tied = $derived(ties(explanation));
  const scale = $derived(axis(shown));

  // One colour per term name, the same in every candidate.
  const PALETTE = ['#2f6db3', '#4e9a3c', '#c83a30', '#e78a2c', '#8b4fa6', '#1f8a8a', '#a07a2a', '#6b6b74', '#b8436b'];
  const colorOf = $derived.by(() => {
    const names = [...new Set(shown.flatMap((c) => c.terms.map((t) => t.name)))];
    return new Map(names.map((n, i) => [n, PALETTE[i % PALETTE.length]!]));
  });

  function bars(terms: { name: string; value: number }[]) {
    let right = 0;
    let left = 0;
    return terms.map((t) => {
      const w = Math.abs(t.value) * scale.unit;
      const x = t.value >= 0 ? scale.zero + right : scale.zero - left - w;
      if (t.value >= 0) right += w;
      else left += w;
      return { ...t, x, w };
    });
  }
</script>

<section class="reasoning panel">
  <header style="--seat: {seatColor(entry.seat).fill}">
    <h3><span class="seat-dot" style="background: {seatColor(entry.seat).fill}"></span> {playerName}'s reasoning</h3>
    {#if thinkMs !== undefined}<span class="ms">{thinkMs} ms</span>{/if}
  </header>
  <div class="body">
    <p class="played">{entry.text}</p>
    {#if tied}
      <p class="rule">A tie: {plural(tied, 'other move')} scored the same as the one played.</p>
    {/if}
    {#if explanation.rule}
      <p class="rule" title={gloss(glossary, explanation.rule)}>Rule: {gloss(glossary, explanation.rule)}</p>
    {/if}
    {#if explanation.context.length}
      <dl class="context">
        {#each explanation.context as c (c.name)}
          <div title={gloss(glossary, c.name)}><dt>{words(c.name)}</dt><dd>{num(c.value)}</dd></div>
        {/each}
      </dl>
    {/if}
    <ol class="candidates">
      {#each shown as c (c.action.id)}
        <li class:chosen={c.action.id === explanation.chosen} class:out={c.total === null}>
          <div class="head">
            <span class="label">{c.action.label}</span>
            <span class="total">{c.total === null ? 'ruled out' : num(c.total)}</span>
          </div>
          {#if c.terms.length}
            <svg class="bar" viewBox="0 0 100 6" preserveAspectRatio="none" aria-hidden="true">
              <line x1={scale.zero} y1="0" x2={scale.zero} y2="6" stroke="#6a5039" stroke-width="0.4" />
              {#each bars(c.terms) as b (b.name)}
                <rect x={b.x} y="0.8" width={Math.max(b.w, 0.4)} height="4.4" fill={colorOf.get(b.name)} opacity={b.value < 0 ? 0.55 : 0.9} />
              {/each}
            </svg>
            <p class="terms">
              {#each c.terms as t (t.name)}
                <span title={gloss(glossary, t.name)}><i style="background: {colorOf.get(t.name)}"></i>{words(t.name)} {num(t.value, true)}</span>
              {/each}
            </p>
          {/if}
          {#if c.rule}<p class="facts" title={gloss(glossary, c.rule)}>{gloss(glossary, c.rule)}</p>{/if}
          {#if c.facts.length}
            <p class="facts">
              {#each c.facts as f (f.name)}<span title={gloss(glossary, f.name)}>{words(f.name)} {num(f.value)}</span>{/each}
            </p>
          {/if}
        </li>
      {/each}
    </ol>
    {#if explanation.candidates.length > shown.length}
      <p class="more">and {explanation.candidates.length - shown.length} more</p>
    {/if}
  </div>
</section>

<style>
  .reasoning {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .reasoning > header {
    background: linear-gradient(color-mix(in srgb, var(--seat) 55%, #2a0f0b), color-mix(in srgb, var(--seat) 35%, #1a0806));
  }
  .ms {
    font-size: 12px;
    color: var(--paper-deep);
  }
  .body {
    overflow-y: auto;
    min-height: 0;
  }
  .played {
    margin: 0 0 6px;
    font-weight: 600;
  }
  .rule {
    margin: 0 0 6px;
    font-size: 13px;
    color: var(--oxblood);
  }
  .context {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 12px;
    margin: 0 0 8px;
    font-size: 12.5px;
  }
  .context div {
    display: flex;
    gap: 4px;
    cursor: help;
  }
  .context dt {
    color: var(--ink-soft);
  }
  .context dd {
    margin: 0;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .candidates {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .candidates li {
    padding: 5px 6px;
    border-top: 1px dotted var(--paper-line);
  }
  li.chosen {
    background: rgb(201 163 90 / 0.22);
    border-radius: 4px;
  }
  li.out {
    opacity: 0.55;
  }
  .head {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 13.5px;
  }
  li.chosen .label::before {
    content: '★ ';
    color: var(--oxblood);
  }
  .total {
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .bar {
    display: block;
    width: 100%;
    height: 9px;
    margin: 3px 0;
  }
  .terms,
  .facts {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 1px 10px;
    font-size: 12px;
  }
  .terms span,
  .facts span,
  .facts {
    cursor: help;
  }
  .terms i {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    margin-right: 3px;
  }
  .facts {
    color: var(--ink-soft);
  }
  .more {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--ink-soft);
  }
</style>
